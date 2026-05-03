"""
RAG3 MinIO Object Store
========================
Lightweight async wrapper around the ``minio`` SDK for storing raw image
crops produced during ingestion. Every stored object's URL is attached
to the corresponding embedding's metadata so the retrieval layer can
surface the image alongside the answer.

If the ``minio`` package is unavailable the adapter degrades gracefully
to a local-filesystem fallback that writes into ``data/object_store/``.
"""

from __future__ import annotations

import asyncio
import io
import uuid
from pathlib import Path
from typing import Any

from src.config import get_settings
from src.monitoring.logger import get_logger

log = get_logger(__name__)

try:  # pragma: no cover - optional
    from minio import Minio
    from minio.error import S3Error
    _MINIO_AVAILABLE = True
except Exception:
    Minio = None  # type: ignore
    S3Error = Exception  # type: ignore
    _MINIO_AVAILABLE = False


class MinioObjectStore:
    """
    Thin async facade over MinIO.

    Methods are async-friendly (``asyncio.to_thread``) even though the
    underlying SDK is synchronous — we do not want ingestion to block
    the event loop while uploading.
    """

    def __init__(self) -> None:
        cfg = get_settings().minio
        self._bucket = cfg.bucket
        self._public_base = (cfg.public_base_url or "").rstrip("/")
        self._secure = cfg.secure
        self._endpoint = cfg.endpoint
        self._local_root = Path("data/object_store") / cfg.bucket

        if _MINIO_AVAILABLE:
            self._client = Minio(
                cfg.endpoint,
                access_key=cfg.access_key.get_secret_value(),
                secret_key=cfg.secret_key.get_secret_value(),
                secure=cfg.secure,
                region=cfg.region,
            )
        else:
            self._client = None
            log.warning("minio package not installed — using local filesystem fallback")
            self._local_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    async def ensure_bucket(self) -> None:
        if self._client is None:
            self._local_root.mkdir(parents=True, exist_ok=True)
            return
        def _run() -> None:
            try:
                if not self._client.bucket_exists(self._bucket):
                    self._client.make_bucket(self._bucket)
            except S3Error as exc:
                log.warning("bucket ensure failed", extra={"err": str(exc)})
        await asyncio.to_thread(_run)

    async def put_bytes(
        self,
        data: bytes,
        *,
        key: str | None = None,
        content_type: str = "image/png",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload an object and return its public URL."""
        key = key or f"{uuid.uuid4().hex}.png"

        if self._client is None:
            dest = self._local_root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return f"file://{dest.resolve().as_posix()}"

        def _run() -> None:
            self._client.put_object(
                self._bucket,
                key,
                io.BytesIO(data),
                length=len(data),
                content_type=content_type,
                metadata=metadata or {},
            )

        await asyncio.to_thread(_run)
        return self._url_for(key)

    def _url_for(self, key: str) -> str:
        if self._public_base:
            return f"{self._public_base}/{self._bucket}/{key}"
        scheme = "https" if self._secure else "http"
        return f"{scheme}://{self._endpoint}/{self._bucket}/{key}"

    async def presigned_get(self, key: str, expires_seconds: int = 3600) -> str:
        """Generate a temporary signed GET URL for a stored object."""
        if self._client is None:
            return self._url_for(key)
        from datetime import timedelta

        def _run() -> str:
            return self._client.presigned_get_object(
                self._bucket, key, expires=timedelta(seconds=expires_seconds)
            )
        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        # Minio SDK has no explicit close — method kept for symmetry.
        return None
