"""
core/logging.py
================
Production structured logging.

Every log entry is a machine-readable JSON object containing
all fields needed to reconstruct the agent's complete execution
trace without any additional context.

Compatible with OpenTelemetry, Datadog, Splunk, and ELK stack.
In development mode, outputs human-readable console format.

Usage:
    from core.logging import get_logger, AgentLogger

    log = get_logger(__name__)
    log.info("tool_called", tool="web_search", query="AI market size")

    # Or use the typed AgentLogger for structured agent events:
    agent_log = AgentLogger(task_id="abc123")
    agent_log.thought(iteration=3, content="I need to search for...")
    agent_log.tool_call(iteration=3, tool="web_search", args={"query": "..."})
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings


# ─── STRUCTURED LOG RECORD ───────────────────────────────────────────────────

@dataclass
class LogRecord:
    """
    Every field that matters for agent observability.
    Serialises cleanly to JSON for ingestion by any log aggregator.
    """
    # Identity
    trace_id:    str             # groups all events in one task run
    span_id:     str             # unique ID for this specific event
    service:     str = "prod_agent"
    version:     str = "1.0.0"

    # Timing
    timestamp:   str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: int = 0

    # Classification
    level:       str = "INFO"    # DEBUG | INFO | WARNING | ERROR | CRITICAL
    event:       str = ""        # machine-readable event name
    logger_name: str = ""

    # Agent context
    task_id:     str = ""
    iteration:   int = 0
    layer:       str = ""        # input | memory | reason | tool | reflect | output

    # Event payload — varies by event type
    payload:     dict[str, Any] = field(default_factory=dict)

    # Error context (populated only on errors)
    error_type:  str = ""
    error_msg:   str = ""
    recoverable: bool = True
    stack_trace: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    def to_console(self) -> str:
        level_colors = {
            "DEBUG":    "\033[36m",     # cyan
            "INFO":     "\033[32m",     # green
            "WARNING":  "\033[33m",     # yellow
            "ERROR":    "\033[31m",     # red
            "CRITICAL": "\033[35m",     # magenta
        }
        reset = "\033[0m"
        color = level_colors.get(self.level, "")

        parts = [
            f"{color}[{self.level:8}]{reset}",
            f"{self.timestamp[:19]}",
            f"task={self.task_id or '-':8}",
            f"iter={self.iteration:2}",
            f"layer={self.layer or '-':10}",
            f"{self.event}",
        ]
        if self.payload:
            parts.append(json.dumps(self.payload, default=str)[:120])
        if self.error_msg:
            parts.append(f"ERROR: {self.error_msg}")

        return "  ".join(parts)


# ─── JSON HANDLER ────────────────────────────────────────────────────────────

class StructuredJSONHandler(logging.Handler):
    """Writes one JSON record per line to stdout and optionally to audit file."""

    def __init__(self, audit_dir: Path | None = None):
        super().__init__()
        self._audit_dir = audit_dir
        self._audit_file = None
        if audit_dir:
            audit_dir.mkdir(parents=True, exist_ok=True)
            fname = audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            self._audit_file = open(fname, "a", buffering=1)   # line-buffered

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            print(line, file=sys.stdout, flush=True)
            if self._audit_file:
                print(line, file=self._audit_file, flush=True)
        except Exception:
            self.handleError(record)

    def close(self):
        if self._audit_file:
            self._audit_file.close()
        super().close()


class StructuredFormatter(logging.Formatter):
    """Converts a LogRecord to JSON or console format."""

    def __init__(self, fmt: str = "json"):
        super().__init__()
        self._fmt = fmt

    def format(self, record: logging.LogRecord) -> str:
        structured: LogRecord | None = getattr(record, "structured", None)
        if structured:
            if self._fmt == "json":
                return structured.to_json()
            return structured.to_console()

        # Fallback for non-structured log calls
        fallback = LogRecord(
            trace_id="-",
            span_id=str(uuid.uuid4())[:8],
            level=record.levelname,
            event=record.getMessage(),
            logger_name=record.name,
        )
        return fallback.to_json() if self._fmt == "json" else fallback.to_console()


# ─── LOGGER FACTORY ──────────────────────────────────────────────────────────

def _configure_root_logger():
    root = logging.getLogger("prod_agent")
    if root.handlers:
        return root

    root.setLevel(getattr(logging, settings.log_level))

    handler = StructuredJSONHandler(
        audit_dir=settings.audit_log_dir if settings.metrics_enabled else None
    )
    handler.setFormatter(StructuredFormatter(fmt=settings.log_format))
    root.addHandler(handler)
    root.propagate = False
    return root


_root = _configure_root_logger()


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger of the prod_agent root."""
    return _root.getChild(name.replace("prod_agent.", ""))


# ─── TYPED AGENT LOGGER ──────────────────────────────────────────────────────

class AgentLogger:
    """
    Typed logging interface for agent-specific events.
    Every method produces a fully structured log record.

    This is what all agent code should use — not raw log calls.
    """

    def __init__(self, task_id: str, trace_id: str | None = None):
        self.task_id  = task_id
        self.trace_id = trace_id or str(uuid.uuid4())
        self._logger  = get_logger("agent")

    def _emit(
        self,
        level:    str,
        event:    str,
        layer:    str,
        iteration: int,
        payload:  dict[str, Any] | None = None,
        duration_ms: int = 0,
        **kwargs,
    ):
        record = LogRecord(
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4())[:8],
            level=level,
            event=event,
            layer=layer,
            task_id=self.task_id,
            iteration=iteration,
            payload=payload or {},
            duration_ms=duration_ms,
            **{k: v for k, v in kwargs.items() if hasattr(LogRecord, k)},
        )
        log_level = getattr(logging, level, logging.INFO)
        log_record = self._logger.makeRecord(
            self._logger.name, log_level,
            "(structured)", 0, "", (), None,
        )
        log_record.structured = record
        self._logger.handle(log_record)
        return record

    # ── Agent lifecycle ───────────────────────────────────────────────────

    def task_start(self, goal: str):
        self._emit("INFO", "task_started", "lifecycle", 0,
                   {"goal": goal[:200]})

    def task_complete(self, iteration: int, exit_reason: str, answer_len: int):
        self._emit("INFO", "task_completed", "lifecycle", iteration,
                   {"exit_reason": exit_reason, "answer_length": answer_len})

    # ── Guardrail events ─────────────────────────────────────────────────

    def input_blocked(self, reason: str, violation: str):
        self._emit("WARNING", "input_blocked", "input", 0,
                   {"reason": reason[:200], "violation_type": violation})

    def input_passed(self, input_length: int):
        self._emit("INFO", "input_passed", "input", 0,
                   {"input_length": input_length})

    def output_validated(self, iteration: int, output_length: int, passed: bool):
        level = "INFO" if passed else "WARNING"
        self._emit(level, "output_validated", "output", iteration,
                   {"output_length": output_length, "passed": passed})

    def pii_redacted(self, layer: str, iteration: int, patterns_found: list[str]):
        self._emit("WARNING", "pii_redacted", layer, iteration,
                   {"patterns_found": patterns_found})

    # ── Memory events ─────────────────────────────────────────────────────

    def context_assembled(
        self,
        iteration: int,
        message_count: int,
        token_count: int,
        rag_chunks: int,
    ):
        self._emit("DEBUG", "context_assembled", "memory", iteration, {
            "message_count": message_count,
            "token_count":   token_count,
            "rag_chunks":    rag_chunks,
        })

    def context_compressed(
        self,
        iteration: int,
        before_tokens: int,
        after_tokens: int,
    ):
        self._emit("INFO", "context_compressed", "memory", iteration, {
            "before_tokens": before_tokens,
            "after_tokens":  after_tokens,
            "reduction_pct": round((1 - after_tokens / max(before_tokens, 1)) * 100, 1),
        })

    # ── Reasoning events ──────────────────────────────────────────────────

    def thought(self, iteration: int, content: str, tokens: int = 0, duration_ms: int = 0):
        self._emit("DEBUG", "thought", "reason", iteration,
                   {"content_preview": content[:150], "tokens": tokens},
                   duration_ms=duration_ms)

    # ── Tool events ───────────────────────────────────────────────────────

    def tool_call(
        self,
        iteration: int,
        tool: str,
        args: dict[str, Any],
        duration_ms: int = 0,
    ):
        self._emit("INFO", "tool_called", "tool", iteration,
                   {"tool": tool, "args": {k: str(v)[:100] for k, v in args.items()}},
                   duration_ms=duration_ms)

    def tool_result(
        self,
        iteration: int,
        tool: str,
        success: bool,
        result_preview: str,
        duration_ms: int = 0,
    ):
        level = "INFO" if success else "WARNING"
        self._emit(level, "tool_result", "tool", iteration,
                   {"tool": tool, "success": success, "preview": result_preview[:150]},
                   duration_ms=duration_ms)

    def tool_blocked(self, iteration: int, tool: str, reason: str):
        self._emit("WARNING", "tool_blocked", "tool", iteration,
                   {"tool": tool, "reason": reason[:200]})

    # ── Error events ──────────────────────────────────────────────────────

    def error(
        self,
        iteration: int,
        layer: str,
        error_type: str,
        message: str,
        recoverable: bool = True,
        stack_trace: str = "",
    ):
        self._emit("ERROR", "agent_error", layer, iteration,
                   {"error_type": error_type, "message": message[:300]},
                   recoverable=recoverable,
                   error_type=error_type,
                   error_msg=message[:300],
                   stack_trace=stack_trace[:500])

    # ── Cost / metrics ────────────────────────────────────────────────────

    def cost_update(
        self,
        iteration: int,
        tokens_this_call: int,
        total_tokens: int,
        estimated_cost_usd: float,
    ):
        self._emit("DEBUG", "cost_update", "metrics", iteration, {
            "tokens_this_call":   tokens_this_call,
            "total_tokens":       total_tokens,
            "estimated_cost_usd": round(estimated_cost_usd, 6),
        })

    def summary(
        self,
        total_iterations: int,
        total_tokens: int,
        tool_counts: dict[str, int],
        total_cost_usd: float,
        exit_reason: str,
    ):
        self._emit("INFO", "task_summary", "lifecycle", total_iterations, {
            "total_iterations": total_iterations,
            "total_tokens":     total_tokens,
            "tool_usage":       tool_counts,
            "cost_usd":         round(total_cost_usd, 6),
            "exit_reason":      exit_reason,
        })


# ─── TIMING CONTEXT MANAGER ──────────────────────────────────────────────────

class Timer:
    """Simple context manager that measures elapsed milliseconds."""

    def __init__(self):
        self.elapsed_ms: int = 0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)
