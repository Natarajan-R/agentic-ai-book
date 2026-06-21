"""
config/settings.py
==================
All configuration driven from environment variables.
No hardcoded values anywhere in the production codebase.

Usage:
    from config.settings import settings
    print(settings.model_primary)

Environment file (.env):
    MODEL_PRIMARY=qwen2.5:7b
    MODEL_SECONDARY=qwen2.5:7b
    OLLAMA_BASE_URL=http://localhost:11434
    CHROMA_PERSIST_DIR=./data/chroma
    MAX_ITERATIONS=15
    MAX_TOKENS_CONTEXT=8000
    LOG_LEVEL=INFO
    LOG_FORMAT=json
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    """
    Complete configuration for the production agent system.
    All values can be overridden via environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Model configuration ────────────────────────────────────────────────
    model_primary: str = Field(
        default="qwen2.5:7b",
        description="Primary LLM for complex reasoning steps",
    )
    model_secondary: str = Field(
        default="qwen2.5:7b",
        description="Secondary LLM for simple steps (summarise, classify, format)",
    )
    model_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="LLM temperature — 0.0 for deterministic production reasoning",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL",
    )

    # ── Loop configuration ─────────────────────────────────────────────────
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum loop iterations per task run",
    )
    max_tokens_context: int = Field(
        default=8000,
        ge=1000,
        le=128000,
        description="Maximum tokens in assembled context window",
    )
    context_compression_threshold: float = Field(
        default=0.75,
        ge=0.5,
        le=0.95,
        description="Compress context when it exceeds this fraction of max_tokens_context",
    )
    episodic_keep_recent: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Number of recent episodic turns to keep uncompressed",
    )

    # ── Tool configuration ─────────────────────────────────────────────────
    tool_rate_limits: dict[str, int] = Field(
        default={
            "web_search":   8,
            "rag_retrieve": 20,
            "code_execute": 5,
            "file_write":   3,
            "human_escalate": 10,
        },
        description="Maximum calls per tool per task run",
    )
    tool_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Maximum seconds to wait for a tool call to complete",
    )
    web_search_max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum results to return from web search",
    )

    # ── Memory configuration ───────────────────────────────────────────────
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="Persistent directory for ChromaDB vector store",
    )
    chroma_collection_name: str = Field(
        default="enterprise_knowledge",
        description="ChromaDB collection name for the knowledge base",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from vector store",
    )
    rag_max_chunk_tokens: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="Maximum tokens per retrieved chunk",
    )

    # ── Guardrail configuration ────────────────────────────────────────────
    agent_scope_keywords: list[str] = Field(
        default=[
            # Kept deliberately specific. Very generic words like "software" or
            # "technology" are omitted because they match unrelated requests
            # (e.g. "a software job") and weaken scope precision.
            "competitor", "pricing", "market", "product", "feature",
            "analysis", "compare", "research", "report", "agent",
            "enterprise", "vendor", "platform",
        ],
        description="Keywords that define the agent's authorised scope",
    )
    irreversible_tools: set[str] = Field(
        default={"send_email", "delete_record", "post_external", "execute_payment"},
        description="Tools that require human confirmation before execution",
    )
    max_input_length: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Maximum characters in a user request",
    )
    pii_redaction_enabled: bool = Field(
        default=True,
        description="Whether to scan and redact PII from tool results and outputs",
    )

    # ── Observability configuration ────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format — json for production, console for development",
    )
    audit_log_dir: Path = Field(
        default=Path("./data/audit"),
        description="Directory for immutable audit logs",
    )
    metrics_enabled: bool = Field(
        default=True,
        description="Whether to collect and expose metrics",
    )

    # ── Cost tracking ──────────────────────────────────────────────────────
    # These simulate cloud pricing for cost awareness reporting.
    # Ollama is free locally — these rates are for reporting equivalence.
    cost_per_1k_tokens_primary: float = Field(
        default=0.003,
        description="Simulated cost per 1K tokens for primary model (cloud equiv.)",
    )
    cost_per_1k_tokens_secondary: float = Field(
        default=0.0005,
        description="Simulated cost per 1K tokens for secondary model (cloud equiv.)",
    )
    max_cost_per_task_usd: float = Field(
        default=1.0,
        ge=0.01,
        description="Maximum simulated cost per task run before hard stop",
    )

    # ── History store (Ch14 self-refining) ────────────────────────────────
    history_store_path: Path = Field(
        default=Path("./data/history.json"),
        description="Path to persistent task history for self-refinement",
    )
    history_max_entries: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum task history entries to retain",
    )

    @field_validator("chroma_persist_dir", "audit_log_dir", mode="before")
    @classmethod
    def ensure_dir_exists(cls, v: str | Path) -> Path:
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("history_store_path", mode="before")
    @classmethod
    def ensure_parent_exists(cls, v: str | Path) -> Path:
        path = Path(v)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def is_irreversible(self, tool_name: str) -> bool:
        return tool_name in self.irreversible_tools

    def cost_estimate(self, tokens: int, model: str) -> float:
        rate = (
            self.cost_per_1k_tokens_primary
            if model == self.model_primary
            else self.cost_per_1k_tokens_secondary
        )
        return (tokens / 1000) * rate


# Single shared settings instance — import this everywhere
settings = AgentSettings()
