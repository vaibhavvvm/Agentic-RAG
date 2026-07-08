"""
RAG System Configuration
=========================
Centralised Pydantic v2 BaseSettings for all environment variables.
Load order: defaults → .env file → OS environment (OS wins).

All secrets come from the environment; this module never hard-codes credentials.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AnyHttpUrl,
    Field,
    PostgresDsn,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Helper types
# ---------------------------------------------------------------------------

PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


# ---------------------------------------------------------------------------
# Sub-settings groups
# ---------------------------------------------------------------------------


class PostgresSettings(BaseSettings):
    """PostgreSQL / pgvector connection settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", extra="ignore")

    host: str = Field(default="localhost", description="PostgreSQL hostname")
    port: int = Field(default=5432, ge=1, le=65535)
    user: str = Field(default="postgres")
    password: SecretStr = Field(default=SecretStr("postgres"))
    db: str = Field(default="rag")

    # pgvector index settings
    vector_dim: PositiveInt = Field(default=768, description="Embedding dimension")
    hnsw_m: PositiveInt = Field(default=16, description="HNSW M parameter")
    hnsw_ef_construction: PositiveInt = Field(default=64)
    hnsw_ef_search: PositiveInt = Field(default=40)

    # Hybrid search weights (must sum ≤ 1.0 each component)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)

    # Connection pool
    pool_min_size: PositiveInt = Field(default=2)
    pool_max_size: PositiveInt = Field(default=10)
    pool_timeout: PositiveInt = Field(default=30, description="Seconds to wait for conn")

    @property
    def dsn(self) -> str:
        """Return asyncpg-compatible connection string."""
        pwd = self.password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def sync_dsn(self) -> str:
        """Return psycopg2-compatible (sync) connection string."""
        pwd = self.password.get_secret_value()
        return (
            f"postgresql://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class Neo4jSettings(BaseSettings):
    """Neo4j graph database connection settings."""

    model_config = SettingsConfigDict(env_prefix="NEO4J_", extra="ignore")

    uri: str = Field(default="bolt://localhost:7687")
    user: str = Field(default="neo4j")
    password: SecretStr = Field(default=SecretStr("neo4j"))
    database: str = Field(default="neo4j")

    # Graphiti-specific
    graphiti_episode_limit: PositiveInt = Field(
        default=10,
        description="Max episodes returned per Graphiti search",
    )
    max_hop_depth: int = Field(
        default=3,
        ge=1,
        le=6,
        description="Maximum graph traversal depth for multi-hop queries",
    )
    community_detection: bool = Field(
        default=True,
        description="Enable Graphiti community detection features",
    )


class GroqSettings(BaseSettings):
    """Groq API settings, including multi-key rotation support."""

    model_config = SettingsConfigDict(env_prefix="GROQ_", extra="ignore")

    # Primary single-key shortcut (optional if api_keys is set)
    api_key: SecretStr | None = Field(default=None)

    # Comma-separated list of keys for rotation: KEY1,KEY2,KEY3
    api_keys_raw: str | None = Field(
        default=None,
        alias="GROQ_API_KEYS",
        description="Comma-separated Groq API keys for rotation",
    )

    # Model selections
    primary_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default inference model",
    )
    fast_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Lightweight model for routing/grading tasks",
    )
    vision_model: str | None = Field(
        default=None,
        description="Optional vision-capable model",
    )

    # Rate-limit handling
    max_retries: PositiveInt = Field(default=5)
    retry_base_delay: NonNegativeFloat = Field(
        default=1.0,
        description="Base exponential back-off delay in seconds",
    )
    retry_max_delay: NonNegativeFloat = Field(
        default=60.0,
        description="Maximum back-off delay cap in seconds",
    )
    request_timeout: PositiveInt = Field(default=120, description="Seconds")

    # Generation defaults
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: PositiveInt = Field(default=2048)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def resolve_api_keys(self) -> "GroqSettings":
        """Ensure at least one API key is available."""
        has_single = self.api_key is not None
        has_multi = self.api_keys_raw is not None and self.api_keys_raw.strip()
        if not has_single and not has_multi:
            raise ValueError(
                "At least one Groq API key must be provided via GROQ_API_KEY "
                "or GROQ_API_KEYS (comma-separated)."
            )
        return self

    @property
    def api_keys(self) -> list[str]:
        """Return all available API keys as a plain list."""
        keys: list[str] = []
        if self.api_keys_raw:
            keys = [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]
        if self.api_key and self.api_key.get_secret_value() not in keys:
            keys.insert(0, self.api_key.get_secret_value())
        return keys


class MinioSettings(BaseSettings):
    """MinIO object storage for raw images cropped during ingestion."""

    model_config = SettingsConfigDict(env_prefix="MINIO_", extra="ignore")

    endpoint: str = Field(default="localhost:9000", description="host:port")
    access_key: SecretStr = Field(default=SecretStr("minioadmin"))
    secret_key: SecretStr = Field(default=SecretStr("minioadmin"))
    secure: bool = Field(default=False, description="Use HTTPS")
    bucket: str = Field(default="rag-assets")
    region: str = Field(default="us-east-1")
    public_base_url: str | None = Field(
        default=None,
        description="If set, object URLs use this prefix (e.g. CDN). Otherwise endpoint is used.",
    )


class FalkorSettings(BaseSettings):
    """FalkorDB (Redis-compatible graph DB) connection."""

    model_config = SettingsConfigDict(env_prefix="FALKOR_", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    password: SecretStr | None = Field(default=None)
    graph_name: str = Field(default="rag")


class OpenRouterSettings(BaseSettings):
    """OpenRouter settings — used as LLM fallback when Groq is unavailable."""

    model_config = SettingsConfigDict(env_prefix="OPENROUTER_", extra="ignore")

    api_key: SecretStr | None = Field(default=None)
    # Multi-key rotation: OPENROUTER_API_KEYS=or-k1,or-k2
    api_keys_raw: str | None = Field(
        default=None,
        alias="OPENROUTER_API_KEYS",
        description="Comma-separated OpenRouter keys for rotation",
    )
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    primary_model: str = Field(default="meta-llama/llama-3.3-70b-instruct")
    fast_model: str = Field(default="meta-llama/llama-3.1-8b-instruct")
    request_timeout: PositiveInt = Field(default=120)
    referer: str = Field(default="http://localhost", description="HTTP-Referer header")
    app_title: str = Field(default="RAG")
    max_retries_per_key: PositiveInt = Field(default=1)

    @property
    def api_keys(self) -> list[str]:
        """All configured keys (merges single + multi)."""
        keys: list[str] = []
        if self.api_keys_raw:
            keys = [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]
        if self.api_key:
            single = self.api_key.get_secret_value()
            if single and single not in keys:
                keys.insert(0, single)
        return keys


class ERExtractionSettings(BaseSettings):
    """Entity-relation extraction models for graph-episode ingestion."""

    model_config = SettingsConfigDict(env_prefix="ER_", extra="ignore")

    # Ollama primary: user-provided local model
    ollama_model: str = Field(default="gpt-oss:20b")
    # Groq fallback: Groq-hosted gpt-oss-20b
    groq_model: str = Field(default="llama-3.1-8b-instant")
    # OpenRouter secondary fallback
    openrouter_model: str = Field(default="llama-3.1-8b-instant")
    max_triples_per_episode: PositiveInt = Field(default=20)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: PositiveInt = Field(default=1024)
    # Try order — any subset of {"ollama","groq","openrouter"}
    fallback_chain: list[Literal["ollama", "groq", "openrouter"]] = Field(
        default=["ollama", "groq", "openrouter"],
    )


class RerankerSettings(BaseSettings):
    """Local HuggingFace reranker (bge-reranker-v2-m3)."""

    model_config = SettingsConfigDict(env_prefix="RERANKER_", extra="ignore")

    model_path: Path = Field(
        default=Path(r"D:\MODELS\bge-reranker-v2-m3"),
        description="Local path to HF bge-reranker-v2-m3 checkpoint",
    )
    device: Literal["cpu", "cuda", "auto"] = Field(default="auto")
    batch_size: PositiveInt = Field(default=16)
    max_length: PositiveInt = Field(default=512)
    backend: Literal["local_hf", "ollama"] = Field(
        default="local_hf",
        description="Which reranker backend to use",
    )


class OllamaSettings(BaseSettings):
    """Ollama local model server settings."""

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    base_url: AnyHttpUrl = Field(default="http://localhost:11434")  # type: ignore[assignment]
    timeout: PositiveInt = Field(default=300, description="Seconds; large models are slow")

    # Model names
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Text embedding model tag",
    )
    reranker_model: str = Field(
        default="bge-reranker-v2-m3",
        description="Cross-encoder reranking model tag",
    )
    vision_model: str = Field(
        default="llama3.2-vision:11b",
        description="Multi-modal model for image understanding (Ollama tag)",
    )
    chat_model: str = Field(
        default="llama3:8b",
        description="Local chat model (used as fallback)",
    )


class MemorySettings(BaseSettings):
    """Sliding-window, episodic and graph memory configuration."""

    model_config = SettingsConfigDict(env_prefix="MEMORY_", extra="ignore")

    # Sliding window
    window_size: PositiveInt = Field(
        default=10,
        description="Number of recent turns kept in RAM",
    )

    # FAISS episodic store
    faiss_index_path: Path = Field(
        default=Path("data/faiss_episodic"),
        description="Directory for persisted FAISS indices",
    )
    faiss_top_k: PositiveInt = Field(default=5)

    # Summarisation trigger
    summarise_after_turns: PositiveInt = Field(
        default=8,
        description="Summarise oldest messages when window reaches this size",
    )

    # Graph memory
    graph_memory_top_k: PositiveInt = Field(default=5)


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline knobs."""

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")

    top_k_vector: PositiveInt = Field(default=10)
    top_k_graph: PositiveInt = Field(default=5)
    top_k_rerank: PositiveInt = Field(default=5, description="After reranking")
    top_k_final: PositiveInt = Field(default=3, description="Passed to LLM context")

    # Self-reflection
    self_reflection_enabled: bool = Field(default=True)
    self_reflection_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Min quality score before re-retrieval",
    )
    max_reflection_rounds: int = Field(default=2, ge=1, le=5)

    # Query expansion
    query_expansion_enabled: bool = Field(default=True)
    num_expanded_queries: int = Field(
        default=3, ge=1, le=6, description="LLM-generated query variants"
    )

    # Cache
    cache_ttl_seconds: PositiveInt = Field(default=3600)
    cache_max_size: PositiveInt = Field(default=512)

    # RRF (Reciprocal Rank Fusion)
    rrf_k: int = Field(default=60, description="RRF smoothing constant")


class ChunkingSettings(BaseSettings):
    """Document chunking parameters."""

    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")

    # Semantic chunker
    semantic_breakpoint_percentile: float = Field(
        default=95.0,
        ge=50.0,
        le=99.9,
        description="Similarity drop percentile that triggers a new chunk",
    )
    semantic_min_chunk_size: PositiveInt = Field(default=100)
    semantic_max_chunk_size: PositiveInt = Field(default=1500)

    # Contextual chunker
    context_window: PositiveInt = Field(
        default=512,
        description="Tokens of surrounding context prepended to each chunk",
    )

    # Hierarchical chunker
    parent_chunk_size: PositiveInt = Field(default=2000)
    child_chunk_size: PositiveInt = Field(default=400)
    child_chunk_overlap: PositiveInt = Field(default=80)


class RouterSettings(BaseSettings):
    """3-tier intent router configuration."""

    model_config = SettingsConfigDict(env_prefix="ROUTER_", extra="ignore")

    # LLM arbiter model (uses fast_model from GroqSettings by default if empty)
    arbiter_model: str = Field(
        default="",
        description="Override model for LLM-based routing; empty = use GROQ fast_model",
    )
    keyword_confidence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Min keyword score to skip LLM arbiter",
    )


class LangSmithSettings(BaseSettings):
    """LangSmith observability / tracing for LangGraph runs."""

    model_config = SettingsConfigDict(env_prefix="LANGSMITH_", extra="ignore")

    api_key: SecretStr | None = Field(default=None, description="LangSmith API key")
    project: str = Field(default="rag", description="LangSmith project name")
    endpoint: str = Field(default="https://api.smith.langchain.com")
    tracing_enabled: bool = Field(
        default=False,
        description="Set true + provide api_key to enable LangSmith tracing",
    )


class MonitoringSettings(BaseSettings):
    """Observability configuration."""

    model_config = SettingsConfigDict(env_prefix="MONITORING_", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_format: Literal["json", "text"] = Field(default="json")
    log_file: Path | None = Field(
        default=None,
        description="Write logs to file in addition to stdout; None = stdout only",
    )

    # Metrics
    metrics_enabled: bool = Field(default=True)
    metrics_percentiles: list[float] = Field(
        default=[0.5, 0.9, 0.99],
        description="Latency percentiles to track (P50, P90, P99)",
    )
    metrics_export_path: Path | None = Field(
        default=None,
        description="Optional JSON file to export metric snapshots",
    )


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Root application settings for RAG.

    Priority (high → low):
        1. Environment variables
        2. .env file (``env_file`` below)
        3. Defaults declared here

    Usage::

        from src.config import get_settings
        cfg = get_settings()
        print(cfg.groq.primary_model)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",   # e.g. POSTGRES__HOST overrides postgres.host
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Nested groups ----
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    minio: MinioSettings = Field(default_factory=MinioSettings)
    falkor: FalkorSettings = Field(default_factory=FalkorSettings)
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    er: ERExtractionSettings = Field(default_factory=ERExtractionSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)

    # ---- Pluggable backends (can be overridden per CLI run) ----
    graph_backend: Literal["neo4j", "falkor", "pggraph", "none"] = Field(
        default="neo4j",
        description="Which graph store to use; 'none' disables graph features.",
    )
    vector_backend: Literal["pgvector", "memory", "auto"] = Field(
        default="auto",
        description="'auto' tries pgvector and falls back to in-memory on failure.",
    )
    llm_fallback_chain: list[Literal["groq", "openrouter", "ollama"]] = Field(
        default=["groq", "openrouter", "ollama"],
        description="LLM providers to try in order on failure.",
    )

    # ---- Top-level constants ----
    app_name: str = Field(default="RAG", description="Application name")
    app_version: str = Field(default="1.0.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=False)
    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for all persisted data",
    )

    @model_validator(mode="after")
    def create_data_dirs(self) -> "Settings":
        """Ensure critical local directories exist at startup."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory.faiss_index_path.mkdir(parents=True, exist_ok=True)
        if self.monitoring.log_file:
            self.monitoring.log_file.parent.mkdir(parents=True, exist_ok=True)
        return self

    @field_validator("environment", mode="before")
    @classmethod
    def normalise_env(cls, v: Any) -> str:
        return str(v).lower()


# ---------------------------------------------------------------------------
# Module-level singleton (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    The result is cached so Pydantic only reads .env once per process.
    In tests, call ``get_settings.cache_clear()`` after patching env vars.
    """
    return Settings()
