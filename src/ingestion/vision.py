"""
RAG Vision Processor
======================
Converts base64-encoded images (extracted by ``DocumentParser``) into
rich natural-language descriptions suitable for embedding and retrieval.

Architecture
------------
* Uses Ollama with a vision-capable model (``llava:13b`` by default).
* Falls back to a simple caption template if Ollama is unreachable.
* Responses are cached (in-process dict) to avoid redundant API calls
  when the same image appears in multiple document chunks.
* Supports async batch processing for throughput.

Haystack 2.x contract
----------------------
``run(image_b64_list: list[str]) -> {"descriptions": list[str]}``

Usage::

    from src.ingestion.vision import VisionProcessor

    vp = VisionProcessor()
    result = vp.run(image_b64_list=[b64_string])
    print(result["descriptions"][0])
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import textwrap
from typing import Any

import httpx
from haystack import component

from src.config import get_settings
from src.monitoring.logger import get_logger
from src.monitoring.metrics import MetricsCollector

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = textwrap.dedent("""\
    You are a precise document analyst. Describe this image extracted from a
    technical or business document.

    Instructions:
    - If the image is a chart or graph: identify the type, axes labels,
      key data series, trends, and any notable values.
    - If the image is a diagram or flowchart: describe the structure,
      components, and relationships shown.
    - If the image is a photograph or illustration: describe what is depicted
      and its likely relevance to the surrounding document.
    - If the image contains text or formulas: transcribe the readable content.
    - Be specific and factual. Use 3-6 sentences.
    - Do NOT speculate beyond what is visible.
""")

_FALLBACK_CAPTION = "[Image: visual content could not be processed by vision model]"


# ---------------------------------------------------------------------------
# Haystack component
# ---------------------------------------------------------------------------


@component
class VisionProcessor:
    """
    Converts base64 images → natural-language descriptions via Ollama LLaVA.

    Args:
        model:         Ollama vision model tag.
        base_url:      Ollama server URL.
        prompt:        System prompt sent alongside each image.
        max_tokens:    Maximum tokens in the model's response.
        temperature:   Sampling temperature (lower = more deterministic).
        cache_size:    Maximum number of descriptions to cache in-process.
        timeout:       HTTP timeout in seconds.
        concurrency:   Maximum parallel Ollama requests (async mode).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        cache_size: int = 512,
        timeout: int | None = None,
        concurrency: int = 4,
    ) -> None:
        cfg = get_settings()
        self.model = model or cfg.ollama.vision_model
        self.base_url = (base_url or str(cfg.ollama.base_url)).rstrip("/")
        self.prompt = prompt or _DEFAULT_PROMPT
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._cache: dict[str, str] = {}
        self._cache_size = cache_size
        self.timeout = timeout or cfg.ollama.timeout
        self._semaphore = asyncio.Semaphore(concurrency)
        self._metrics = MetricsCollector.get_instance()

        log.info(
            "VisionProcessor initialised",
            extra={"model": self.model, "concurrency": concurrency},
        )

    # ------------------------------------------------------------------
    # Haystack run()  (sync wrapper around async impl)
    # ------------------------------------------------------------------

    @component.output_types(descriptions=list)
    def run(
        self,
        image_b64_list: list[str],
        extra_context: str | None = None,
    ) -> dict[str, list[str]]:
        """
        Generate natural-language descriptions for a batch of base64 images.

        Args:
            image_b64_list: List of base64-encoded image strings (PNG/JPEG).
            extra_context:  Optional surrounding text (e.g. the paragraph
                            adjacent to the image) to improve caption quality.

        Returns:
            ``{"descriptions": list[str]}`` — one description per image,
            in the same order as the input.
        """
        if not image_b64_list:
            return {"descriptions": []}

        descriptions = asyncio.run(
            self._process_batch(image_b64_list, extra_context)
        )
        return {"descriptions": descriptions}

    # ------------------------------------------------------------------
    # Async batch processing
    # ------------------------------------------------------------------

    async def _process_batch(
        self,
        image_b64_list: list[str],
        extra_context: str | None,
    ) -> list[str]:
        tasks = [
            self._process_one(b64, extra_context)
            for b64 in image_b64_list
        ]
        return list(await asyncio.gather(*tasks))

    async def _process_one(
        self,
        image_b64: str,
        extra_context: str | None,
    ) -> str:
        """Describe a single image, using cache if available."""
        cache_key = self._make_key(image_b64)
        if cache_key in self._cache:
            self._metrics.record_event("vision.cache_hit")
            return self._cache[cache_key]

        async with self._semaphore:
            with self._metrics.measure("vision.ollama_call"):
                description = await self._call_ollama(image_b64, extra_context)

        # Evict oldest entry if cache is full (simple FIFO)
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = description
        self._metrics.record_event("vision.images_processed")
        return description

    async def _call_ollama(self, image_b64: str, extra_context: str | None) -> str:
        """
        POST to the Ollama ``/api/chat`` endpoint with the image payload.

        Ollama expects images in the ``images`` field of a message.
        """
        prompt_text = self.prompt
        if extra_context:
            prompt_text += f"\n\nSurrounding document context:\n{extra_context[:500]}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [image_b64],
                }
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }

        url = f"{self.base_url}/api/chat"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["message"]["content"].strip()
        except httpx.HTTPStatusError as exc:
            log.error(
                "Ollama vision HTTP error",
                extra={"status": exc.response.status_code, "model": self.model},
            )
            self._metrics.record_event("vision.error")
            return _FALLBACK_CAPTION
        except Exception as exc:
            log.error("Ollama vision error", extra={"error": str(exc)})
            self._metrics.record_event("vision.error")
            return _FALLBACK_CAPTION

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(image_b64: str) -> str:
        """SHA-256 of the first 4096 bytes of the b64 string (fast enough)."""
        return hashlib.sha256(image_b64[:4096].encode("ascii")).hexdigest()

    def describe_bytes(
        self,
        images_png: list[bytes],
        extra_context: str | None = None,
    ) -> list[str]:
        """Convenience: accept raw PNG bytes instead of pre-encoded base64."""
        b64_list = [base64.b64encode(b).decode("ascii") for b in images_png]
        return self.run(image_b64_list=b64_list, extra_context=extra_context)["descriptions"]

    def warm_up(self) -> None:
        """Haystack lifecycle hook — validates Ollama vision connectivity."""
        log.info(
            "VisionProcessor warm_up: checking Ollama vision model",
            extra={"model": self.model},
        )
        try:
            # 1×1 transparent PNG in base64
            tiny_png = (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
                "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            )
            result = self.run(image_b64_list=[tiny_png])
            if result["descriptions"]:
                log.info("VisionProcessor warm_up: OK")
        except Exception as exc:
            log.warning(
                "VisionProcessor warm_up failed (non-fatal)",
                extra={"error": str(exc)},
            )
