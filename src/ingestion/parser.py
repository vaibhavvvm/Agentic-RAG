"""
RAG Document Parser
=====================
Converts raw files (PDF, DOCX, HTML, TXT, Markdown, images) into typed
``ParsedElement`` objects that downstream chunkers and processors consume.

Uses ``unstructured`` with the ``hi_res`` strategy for PDFs (tesseract OCR
+ detectron2 layout analysis) and ``fast`` / ``auto`` for other formats.

Output element types
--------------------
* ``text``   — body paragraphs, headings, list items
* ``table``  — HTML-encoded table (passed to TableReformatter)
* ``image``  — base64-encoded image bytes (passed to VisionProcessor)
* ``title``  — document title / section heading
* ``footer`` / ``header`` — page furniture (usually discarded)

Haystack 2.x contract
----------------------
``run(file_path: str | Path, ...) -> {"elements": list[ParsedElement]}``

Usage::

    from src.ingestion.parser import DocumentParser, ParsedElement

    parser = DocumentParser()
    result = parser.run(file_path="docs/manual.pdf")
    for el in result["elements"]:
        print(el.element_type, el.text[:80])
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from haystack import component

from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)

ElementType = Literal["text", "table", "image", "title", "header", "footer", "unknown"]

# MIME types we can handle
_SUPPORTED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/html",
    "text/plain",
    "text/markdown",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
}

# Unstructured element type → our canonical type
_TYPE_MAP: dict[str, ElementType] = {
    "NarrativeText": "text",
    "Text": "text",
    "ListItem": "text",
    "Title": "title",
    "Header": "header",
    "Footer": "footer",
    "Table": "table",
    "Image": "image",
    "FigureCaption": "text",
    "Address": "text",
    "EmailAddress": "text",
    "Formula": "text",
    "CodeSnippet": "text",
    "PageBreak": "unknown",
    "UncategorizedText": "text",
}


# ---------------------------------------------------------------------------
# Output DTO
# ---------------------------------------------------------------------------


@dataclass
class ParsedElement:
    """
    Single logical element extracted from a document.

    Attributes:
        element_type:  Canonical type string.
        text:          Plain or HTML text content.
        page_number:   1-based page number (None if not applicable).
        coordinates:   Bounding box dict from unstructured (may be None).
        image_b64:     Base64-encoded image bytes (only for type=``image``).
        metadata:      Arbitrary extra attributes (source, element_id, etc.).
    """

    element_type: ElementType
    text: str
    page_number: int | None = None
    coordinates: dict[str, Any] | None = None
    image_b64: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_substantive(self) -> bool:
        """Return True if this element carries meaningful content."""
        return (
            self.element_type not in ("header", "footer", "unknown")
            and len(self.text.strip()) > 10
        )


# ---------------------------------------------------------------------------
# Haystack component
# ---------------------------------------------------------------------------


@component
class DocumentParser:
    """
    Multi-format document parser built on ``unstructured``.

    Args:
        strategy:           ``"hi_res"`` (OCR + layout; slow but accurate) or
                            ``"fast"`` (text-layer only; no OCR).
        extract_images:     Whether to base64-encode embedded images.
        max_image_b64_mb:   Images larger than this (MB) are skipped.
        include_page_breaks: Whether to emit ``unknown``-typed page-break elements.
        pdf_infer_table_structure: Use table inference for PDFs (hi_res only).
        languages:          OCR language hints (e.g. ``["eng", "fra"]``).
    """

    def __init__(
        self,
        strategy: Literal["hi_res", "fast", "auto"] = "hi_res",
        extract_images: bool = True,
        max_image_b64_mb: float = 5.0,
        include_page_breaks: bool = False,
        pdf_infer_table_structure: bool = True,
        languages: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.extract_images = extract_images
        self.max_image_b64_bytes = int(max_image_b64_mb * 1024 * 1024)
        self.include_page_breaks = include_page_breaks
        self.pdf_infer_table_structure = pdf_infer_table_structure
        self.languages = languages or ["eng"]
        self._metrics = MetricsCollector.get_instance()

        log.info(
            "DocumentParser initialised",
            extra={"strategy": strategy, "extract_images": extract_images},
        )

    # ------------------------------------------------------------------
    # Haystack run()
    # ------------------------------------------------------------------

    @component.output_types(elements=list)
    def run(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, list[ParsedElement]]:
        """
        Parse a document file and return its structured elements.

        Args:
            file_path: Absolute or relative path to the source file.
            metadata:  Extra key/value pairs merged into every element's
                       metadata (e.g. ``{"source": "upload", "doc_id": "x"}``).

        Returns:
            ``{"elements": list[ParsedElement]}`` ordered by document position.

        Raises:
            FileNotFoundError:    If ``file_path`` does not exist.
            ValueError:           If the file format is unsupported.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        mime, _ = mimetypes.guess_type(str(path))
        if mime and mime not in _SUPPORTED_MIMES:
            log.warning(
                "Potentially unsupported MIME type",
                extra={"path": str(path), "mime": mime},
            )

        extra_meta: dict[str, Any] = {
            "source_file": str(path),
            "file_name": path.name,
            "mime_type": mime or "unknown",
            **(metadata or {}),
        }

        with timed_operation("parser.parse", log, extra={"file": path.name}):
            raw_elements = self._partition(path)

        elements = self._convert(raw_elements, extra_meta)
        self._metrics.record_event("parser.documents_parsed")
        self._metrics.record_event("parser.elements_extracted", len(elements))

        log.info(
            "Document parsed",
            extra={
                "file": path.name,
                "total_elements": len(elements),
                "text_elements": sum(1 for e in elements if e.element_type == "text"),
                "table_elements": sum(1 for e in elements if e.element_type == "table"),
                "image_elements": sum(1 for e in elements if e.element_type == "image"),
            },
        )
        return {"elements": elements}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _partition(self, path: Path) -> list[Any]:
        """
        Call the appropriate ``unstructured`` partitioner.

        Returns raw ``unstructured`` Element objects.
        """
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise ImportError(
                "unstructured is required: pip install 'unstructured[all-docs]'"
            ) from exc

        suffix = path.suffix.lower()
        kwargs: dict[str, Any] = {
            "filename": str(path),
            "strategy": self.strategy,
            "languages": self.languages,
            "include_page_breaks": self.include_page_breaks,
        }

        if suffix == ".pdf":
            kwargs["pdf_infer_table_structure"] = self.pdf_infer_table_structure
            kwargs["extract_images_in_pdf"] = self.extract_images
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".tiff"):
            # Images go directly through vision — return a synthetic element
            return self._wrap_image_file(path)

        return partition(**kwargs)

    def _wrap_image_file(self, path: Path) -> list[Any]:
        """Create a synthetic element list for a standalone image file."""
        class _SyntheticImageElement:
            category = "Image"
            text = ""
            metadata: Any = type("M", (), {
                "page_number": 1,
                "coordinates": None,
                "image_base64": None,
            })()

        el = _SyntheticImageElement()
        if self.extract_images:
            raw = path.read_bytes()
            if len(raw) <= self.max_image_b64_bytes:
                el.metadata.image_base64 = base64.b64encode(raw).decode("ascii")
        el.text = f"[Image file: {path.name}]"
        return [el]

    def _convert(
        self,
        raw_elements: list[Any],
        extra_meta: dict[str, Any],
    ) -> list[ParsedElement]:
        """
        Convert raw ``unstructured`` elements → ``ParsedElement`` objects.

        Applies type mapping, filters page breaks (unless configured), and
        base64-encodes image data.
        """
        results: list[ParsedElement] = []
        for i, el in enumerate(raw_elements):
            cat: str = getattr(el, "category", "UncategorizedText")
            el_type: ElementType = _TYPE_MAP.get(cat, "unknown")

            if el_type == "unknown" and not self.include_page_breaks:
                continue

            # Extract page number
            page_num: int | None = None
            coords: dict[str, Any] | None = None
            if hasattr(el, "metadata"):
                meta = el.metadata
                page_num = getattr(meta, "page_number", None)
                raw_coords = getattr(meta, "coordinates", None)
                if raw_coords is not None:
                    try:
                        coords = {
                            "points": raw_coords.points,
                            "system": raw_coords.system,
                        }
                    except AttributeError:
                        coords = None

            # Extract image data
            img_b64: str | None = None
            if el_type == "image" and self.extract_images:
                img_b64 = self._extract_image_b64(el)

            # Table: prefer HTML representation
            text = el.text or ""
            if el_type == "table":
                text = self._table_html(el) or text

            parsed = ParsedElement(
                element_type=el_type,
                text=text,
                page_number=page_num,
                coordinates=coords,
                image_b64=img_b64,
                metadata={
                    "element_index": i,
                    "unstructured_category": cat,
                    **extra_meta,
                },
            )
            results.append(parsed)

        return results

    def _extract_image_b64(self, el: Any) -> str | None:
        """Extract base64-encoded image from an unstructured Image element."""
        # unstructured stores b64 in metadata.image_base64 for hi_res PDF
        if hasattr(el, "metadata"):
            b64 = getattr(el.metadata, "image_base64", None)
            if b64:
                raw = base64.b64decode(b64)
                if len(raw) <= self.max_image_b64_bytes:
                    return b64
        return None

    @staticmethod
    def _table_html(el: Any) -> str | None:
        """Return the HTML string of a table element if available."""
        if hasattr(el, "metadata"):
            return getattr(el.metadata, "text_as_html", None)
        return None
