"""
RAG3 Contextual Chunker
========================
Enriches each chunk with LLM-generated context that describes the chunk's
role within the broader document.

Inspired by Anthropic's "Contextual Retrieval" technique: prepending a
succinct situating sentence dramatically improves retrieval accuracy because
the embedding captures both the chunk content AND its document position.

Pipeline
--------
1. Receive a list of Haystack ``Document`` objects (output of any chunker).
2. Build a document-level summary (first N characters of the full document)
   as the backdrop.
3. For each chunk, call the LLM once to produce a 1-2 sentence context
   blurb: *"This chunk is from Section 3 of the user manual and describes
   the installation steps for Windows."*
4. Prepend the blurb to the chunk content, creating a new ``Document``
   whose ID is derived from the original.

Caching
-------
* Results are cached by a key of ``(model, doc_backdrop_hash, chunk_hash)``
  so re-running on the same corpus is free.

Haystack 2.x contract
----------------------
``run(documents: list[Document], full_text: str) -> {"documents": list[Document]}``

Usage::

    from src.ingestion.contextual_chunker import ContextualChunker

    enricher = ContextualChunker()
    result = enricher.run(documents=sem_docs, full_text=raw_text)
    enriched = result["documents"]
"""

from __future__ import annotations

import hashlib
import textwrap
from typing import Any

from haystack import Document, component
from haystack.dataclasses import ChatMessage

from src.config import get_settings
from src.monitoring.logger import get_logger, timed_operation
from src.monitoring.metrics import MetricsCollector
from src.utils.groq_client import RotatableGroqGenerator

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a precise document indexing assistant.

    Your task: given a document excerpt (backdrop) and a specific chunk from
    that document, write a single sentence (max 40 words) that situates the
    chunk within the document's overall structure.

    Rules:
    - State which section/topic the chunk belongs to.
    - Note what information the chunk contains.
    - Do NOT repeat content verbatim — be concise and descriptive.
    - Output ONLY the situating sentence, nothing else.
""")

_USER_TEMPLATE = """\
DOCUMENT BACKDROP (first {backdrop_chars} characters):
{backdrop}

CHUNK TO SITUATE:
{chunk}

Situating sentence:"""


# ---------------------------------------------------------------------------
# Haystack component
# ---------------------------------------------------------------------------


@component
class ContextualChunker:
    """
    LLM-powered chunk context enricher.

    Args:
        model:           Groq model for context generation (fast_model default).
        backdrop_chars:  How many characters of the source document to use
                         as context backdrop (sent to the LLM).
        max_tokens:      Max tokens in the LLM's context sentence.
        temperature:     LLM temperature (low → deterministic).
        concurrency:     Parallel LLM calls (controlled via threading pool).
        cache_enabled:   Toggle in-process caching of generated contexts.
    """

    def __init__(
        self,
        model: str | None = None,
        backdrop_chars: int = 3000,
        max_tokens: int = 80,
        temperature: float = 0.0,
        concurrency: int = 8,
        cache_enabled: bool = True,
    ) -> None:
        cfg = get_settings()
        self.model = model or cfg.groq.fast_model
        self.backdrop_chars = backdrop_chars
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        self._cache_enabled = cache_enabled
        self._cache: dict[str, str] = {}

        self._llm = RotatableGroqGenerator(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._metrics = MetricsCollector.get_instance()

        log.info(
            "ContextualChunker initialised",
            extra={"model": self.model, "backdrop_chars": backdrop_chars},
        )

    # ------------------------------------------------------------------
    # Haystack run()
    # ------------------------------------------------------------------

    @component.output_types(documents=list)
    def run(
        self,
        documents: list[Document],
        full_text: str,
        source_id: str = "",
    ) -> dict[str, list[Document]]:
        """
        Enrich each document with a situating context prefix.

        Args:
            documents:  Chunks to enrich (any chunker output).
            full_text:  Complete raw text of the source document used as
                        the LLM backdrop.
            source_id:  Optional identifier for the source document (used
                        in cache keys for isolation across documents).

        Returns:
            ``{"documents": list[Document]}`` — same length as input,
            each with an enriched ``content`` field and updated metadata.
        """
        if not documents:
            return {"documents": []}

        backdrop = full_text[: self.backdrop_chars]
        backdrop_hash = hashlib.sha256(
            f"{source_id}:{backdrop}".encode()
        ).hexdigest()[:16]

        enriched: list[Document] = []
        with timed_operation(
            "contextual_chunker.run", log,
            extra={"n_docs": len(documents)},
        ):
            for doc in documents:
                enriched_doc = self._enrich(doc, backdrop, backdrop_hash)
                enriched.append(enriched_doc)

        self._metrics.record_event("contextual_chunker.chunks_enriched", len(enriched))
        log.info(
            "ContextualChunker complete",
            extra={"enriched": len(enriched)},
        )
        return {"documents": enriched}

    # ------------------------------------------------------------------
    # Single-document enrichment
    # ------------------------------------------------------------------

    def _enrich(
        self,
        doc: Document,
        backdrop: str,
        backdrop_hash: str,
    ) -> Document:
        """Generate context for one chunk and prepend it to the content."""
        content = doc.content or ""
        chunk_hash = hashlib.sha256(content[:512].encode()).hexdigest()[:16]
        cache_key = f"{backdrop_hash}:{chunk_hash}"

        # Cache lookup
        context_sentence: str | None = None
        if self._cache_enabled:
            context_sentence = self._cache.get(cache_key)

        if context_sentence is None:
            context_sentence = self._generate_context(backdrop, content)
            if self._cache_enabled:
                if len(self._cache) > 8192:
                    # Simple eviction: clear oldest half
                    keys = list(self._cache.keys())
                    for k in keys[:4096]:
                        del self._cache[k]
                self._cache[cache_key] = context_sentence
            self._metrics.record_event("contextual_chunker.llm_calls")
        else:
            self._metrics.record_event("contextual_chunker.cache_hits")

        enriched_content = f"{context_sentence}\n\n{content}"

        return Document(
            id=f"ctx_{doc.id}",
            content=enriched_content,
            meta={
                **doc.meta,
                "context_sentence": context_sentence,
                "original_content": content,
                "original_doc_id": doc.id,
                "chunker": "contextual",
            },
            embedding=doc.embedding,
            score=doc.score,
        )

    def _generate_context(self, backdrop: str, chunk: str) -> str:
        """Call the LLM to produce a situating sentence for this chunk."""
        user_msg = _USER_TEMPLATE.format(
            backdrop_chars=self.backdrop_chars,
            backdrop=backdrop,
            chunk=chunk[:1500],  # cap chunk length sent to LLM
        )
        try:
            result = self._llm.run(
                messages=[
                    ChatMessage.from_system(_SYSTEM_PROMPT),
                    ChatMessage.from_user(user_msg),
                ],
                generation_kwargs={
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            return result["replies"][0].content.strip()
        except Exception as exc:
            log.warning(
                "ContextualChunker LLM call failed",
                extra={"error": str(exc)},
            )
            self._metrics.record_event("contextual_chunker.llm_errors")
            return ""  # Return empty — original content is still used

    def cache_stats(self) -> dict[str, Any]:
        return {
            "cache_size": len(self._cache),
            "llm_calls": self._metrics.get_counter("contextual_chunker.llm_calls"),
            "cache_hits": self._metrics.get_counter("contextual_chunker.cache_hits"),
        }
