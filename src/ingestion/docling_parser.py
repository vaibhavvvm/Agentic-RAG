"""
RAG3 Docling Parser
=====================
Docling-based document parser that converts PDFs/DOCX/HTML into:

  * structured **Markdown** (preserving headings, lists, tables)
  * a list of **image crops** (PNG bytes) for every figure/chart Docling detects

Image crops are returned separately so the LangGraph ingestion pipeline
can route them to the VLM (llama3.2-vision:11b) and into MinIO.

Graceful fallback: if ``docling`` is not installed the parser falls back
to ``unstructured`` (already a dependency of Phase 2) and emits no image
crops — the rest of the pipeline still works, just without visuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.monitoring.logger import get_logger

log = get_logger(__name__)


try:  # pragma: no cover - optional
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    _DOCLING_AVAILABLE = True
except Exception:
    DocumentConverter = None  # type: ignore
    InputFormat = None  # type: ignore
    _DOCLING_AVAILABLE = False


@dataclass
class ParsedImage:
    """One cropped image ready for the VLM + MinIO."""
    bytes_png: bytes
    page: int
    bbox: tuple[float, float, float, float] | None = None
    caption: str = ""
    element_id: str = ""


@dataclass
class ParsedDocument:
    """Canonical output shape consumed by the ingestion LangGraph."""
    markdown: str = ""
    images: list[ParsedImage] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)   # Markdown table strings
    metadata: dict[str, Any] = field(default_factory=dict)


class DoclingParser:
    """
    Parse a document with Docling, producing Markdown + cropped images.

    Example::

        parser = DoclingParser()
        parsed = parser.run(Path("paper.pdf"))
        print(parsed.markdown[:500])
        print(f"{len(parsed.images)} images, {len(parsed.tables)} tables")
    """

    def __init__(self, export_images: bool = True) -> None:
        self._export_images = export_images
        if _DOCLING_AVAILABLE:
            self._converter = DocumentConverter()
        else:
            self._converter = None
            log.warning(
                "docling not installed — DoclingParser will fall back to unstructured"
            )

    def run(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if self._converter is None:
            return self._fallback_unstructured(file_path)
        return self._run_docling(file_path)

    # ------------------------------------------------------------------

    def _run_docling(self, file_path: Path) -> ParsedDocument:  # pragma: no cover - optional
        result = self._converter.convert(str(file_path))
        doc = result.document
        markdown = doc.export_to_markdown()

        images: list[ParsedImage] = []
        tables: list[str] = []

        # Tables → Markdown
        for t in getattr(doc, "tables", []) or []:
            try:
                tables.append(t.export_to_markdown())
            except Exception:
                continue

        # Image crops
        if self._export_images:
            for pic in getattr(doc, "pictures", []) or []:
                try:
                    pil_image = pic.get_image(doc)
                    if pil_image is None:
                        continue
                    import io
                    buf = io.BytesIO()
                    pil_image.save(buf, format="PNG")
                    bbox = getattr(pic, "bbox", None)
                    page = getattr(pic, "page_no", 0) or 0
                    images.append(
                        ParsedImage(
                            bytes_png=buf.getvalue(),
                            page=int(page),
                            bbox=tuple(bbox) if bbox else None,  # type: ignore[arg-type]
                            caption=(getattr(pic, "caption_text", "") or ""),
                            element_id=getattr(pic, "self_ref", "") or "",
                        )
                    )
                except Exception as exc:
                    log.debug("picture export failed", extra={"err": str(exc)})

        return ParsedDocument(
            markdown=markdown,
            images=images,
            tables=tables,
            metadata={"source": str(file_path), "backend": "docling"},
        )

    # ------------------------------------------------------------------

    def _fallback_unstructured(self, file_path: Path) -> ParsedDocument:
        try:
            from unstructured.partition.auto import partition
        except Exception as exc:
            log.error("both docling and unstructured missing", extra={"err": str(exc)})
            return ParsedDocument(
                markdown=file_path.read_text(encoding="utf-8", errors="ignore")
                if file_path.suffix.lower() in {".md", ".txt"}
                else "",
                metadata={"source": str(file_path), "backend": "plain"},
            )

        elements = partition(filename=str(file_path))
        md_parts: list[str] = []
        tables: list[str] = []
        for el in elements:
            category = getattr(el, "category", "") or type(el).__name__
            text = (getattr(el, "text", "") or "").strip()
            if not text:
                continue
            if "Table" in category:
                tables.append(text)
                md_parts.append(text)
            elif "Title" in category or "Header" in category:
                md_parts.append(f"## {text}")
            else:
                md_parts.append(text)

        return ParsedDocument(
            markdown="\n\n".join(md_parts),
            images=[],
            tables=tables,
            metadata={"source": str(file_path), "backend": "unstructured"},
        )
