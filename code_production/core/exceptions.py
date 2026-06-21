"""
core/exceptions.py
===================
Structured exception hierarchy for the production agent.

Every failure has a specific type, a recoverable flag,
and enough context to make a sensible replanning decision.

Design principles:
  - Every exception carries task_id and iteration for correlation
  - recoverable=True  → agent should retry or substitute
  - recoverable=False → agent should escalate to human
  - All exceptions are loggable as structured dicts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ─── BASE ────────────────────────────────────────────────────────────────────

@dataclass
class AgentError(Exception):
    """Root exception for all agent errors."""

    message:     str
    task_id:     str          = "unknown"
    iteration:   int          = 0
    recoverable: bool         = True
    context:     dict[str, Any] = field(default_factory=dict)
    timestamp:   str          = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"message={self.message!r}, "
            f"task_id={self.task_id}, "
            f"iter={self.iteration}, "
            f"recoverable={self.recoverable})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type":  type(self).__name__,
            "message":     self.message,
            "task_id":     self.task_id,
            "iteration":   self.iteration,
            "recoverable": self.recoverable,
            "context":     self.context,
            "timestamp":   self.timestamp,
        }


# ─── GUARDRAIL ERRORS ────────────────────────────────────────────────────────

@dataclass
class GuardrailError(AgentError):
    """Raised when a guardrail blocks an input, tool call, or output."""
    guardrail_layer: str = "unknown"    # input | tool_pre | tool_post | output
    violation_type:  str = "unknown"    # injection | scope | rate_limit | pii | incomplete

    def __post_init__(self):
        self.recoverable = False        # guardrail blocks are not retried
        self.context.update({
            "guardrail_layer": self.guardrail_layer,
            "violation_type":  self.violation_type,
        })


class InjectionDetectedError(GuardrailError):
    """Prompt injection attempt detected."""
    def __post_init__(self):
        super().__post_init__()
        self.guardrail_layer = "input"
        self.violation_type  = "injection"


class OutOfScopeError(GuardrailError):
    """Request falls outside the agent's authorised scope."""
    def __post_init__(self):
        super().__post_init__()
        self.guardrail_layer = "input"
        self.violation_type  = "scope"


@dataclass
class RateLimitError(GuardrailError):
    """Tool call rate limit exceeded."""
    tool_name: str = "unknown"

    def __post_init__(self):
        super().__post_init__()
        self.guardrail_layer = "tool_pre"
        self.violation_type  = "rate_limit"
        self.recoverable     = False


class OutputValidationError(GuardrailError):
    """Final output failed validation."""
    def __post_init__(self):
        super().__post_init__()
        self.guardrail_layer = "output"
        self.violation_type  = "incomplete"
        self.recoverable     = True     # can trigger one remediation loop


# ─── TOOL ERRORS ─────────────────────────────────────────────────────────────

@dataclass
class ToolError(AgentError):
    """Base class for tool execution failures."""
    tool_name:  str = "unknown"
    tool_args:  dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.context.update({
            "tool_name": self.tool_name,
            "tool_args": {k: str(v)[:100] for k, v in self.tool_args.items()},
        })


@dataclass
class ToolTimeoutError(ToolError):
    """Tool call exceeded the configured timeout."""
    timeout_seconds: int = 30

    def __post_init__(self):
        super().__post_init__()
        self.recoverable = True     # retry with backoff
        self.context["timeout_seconds"] = self.timeout_seconds


@dataclass
class ToolExecutionError(ToolError):
    """Tool raised an exception during execution."""
    original_error: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.recoverable = True
        self.context["original_error"] = self.original_error


class ToolNotFoundError(ToolError):
    """Agent requested a tool that is not in the registry."""
    def __post_init__(self):
        super().__post_init__()
        self.recoverable = False    # cannot retry with a non-existent tool


@dataclass
class ToolResultParseError(ToolError):
    """Tool returned output that could not be parsed."""
    raw_output: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.recoverable = True
        self.context["raw_output"] = self.raw_output[:200]


# ─── MEMORY ERRORS ───────────────────────────────────────────────────────────

@dataclass
class MemoryError(AgentError):
    """Base class for memory layer failures."""
    memory_type: str = "unknown"   # in_context | episodic | semantic | structured

    def __post_init__(self):
        self.context["memory_type"] = self.memory_type


@dataclass
class ContextWindowExceededError(MemoryError):
    """Assembled context exceeds the configured token limit."""
    token_count: int = 0
    token_limit: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.memory_type = "in_context"
        self.recoverable = True     # trigger compression
        self.context.update({
            "token_count": self.token_count,
            "token_limit": self.token_limit,
        })


class VectorStoreError(MemoryError):
    """ChromaDB or other vector store operation failed."""
    def __post_init__(self):
        super().__post_init__()
        self.memory_type = "semantic"
        self.recoverable = True     # degrade gracefully without RAG


# ─── PLANNING ERRORS ─────────────────────────────────────────────────────────

@dataclass
class PlanningError(AgentError):
    """Base class for planning layer failures."""
    plan_step: int = 0


class PlanGenerationError(PlanningError):
    """LLM failed to produce a valid structured plan."""
    def __post_init__(self):
        self.recoverable = True     # retry with simplified prompt


@dataclass
class PlanInvalidatedError(PlanningError):
    """New observation invalidates one or more remaining plan steps."""
    invalidated_steps: list[int] = field(default_factory=list)
    reason:            str = ""

    def __post_init__(self):
        self.recoverable = True     # triggers dynamic replanning
        self.context.update({
            "invalidated_steps": self.invalidated_steps,
            "reason":            self.reason,
        })


@dataclass
class DependencyNotMetError(PlanningError):
    """A plan step's dependency has not been completed."""
    depends_on: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.recoverable = False    # dependency failures require resequencing
        self.context["depends_on"] = self.depends_on


# ─── LOOP ERRORS ─────────────────────────────────────────────────────────────

@dataclass
class LoopError(AgentError):
    """Base class for execution loop failures."""


@dataclass
class MaxIterationsError(LoopError):
    """Loop reached the configured maximum iteration count."""
    max_iterations: int = 15

    def __post_init__(self):
        self.recoverable = False    # hard cap — escalate to human
        self.context["max_iterations"] = self.max_iterations


@dataclass
class OscillationDetectedError(LoopError):
    """Agent is repeating the same actions without progress."""
    repeated_action: str = ""

    def __post_init__(self):
        self.recoverable = False    # oscillation requires human review
        self.context["repeated_action"] = self.repeated_action


@dataclass
class CostBudgetExceededError(LoopError):
    """Task run has exceeded the configured cost budget."""
    current_cost:  float = 0.0
    budget_limit:  float = 0.0

    def __post_init__(self):
        self.recoverable = False
        self.context.update({
            "current_cost": self.current_cost,
            "budget_limit": self.budget_limit,
        })


# ─── LLM ERRORS ──────────────────────────────────────────────────────────────

@dataclass
class LLMError(AgentError):
    """Base class for LLM inference failures."""
    model_name: str = "unknown"

    def __post_init__(self):
        self.context["model_name"] = self.model_name


class LLMConnectionError(LLMError):
    """Cannot reach the Ollama server."""
    def __post_init__(self):
        super().__post_init__()
        self.recoverable = True     # retry after backoff


@dataclass
class LLMOutputParseError(LLMError):
    """LLM output could not be parsed as expected structured format."""
    raw_output: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.recoverable = True
        self.context["raw_output"] = self.raw_output[:300]
