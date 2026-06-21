"""
core/state.py
==============
LangGraph state definition — the single source of truth
for everything that flows through the agent graph.

Every node reads from and writes to this state.
LangGraph checkpoints this automatically, giving us
state persistence across process restarts for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """
    Complete state for the production agent.

    Design principles:
    - Every field has a clear owner (which node writes it)
    - No field is written by more than one node
    - Exit conditions are explicit, never implicit
    - Cost and iteration tracking are first-class citizens
    """

    # ── Identity ────────────────────────────────────────────────────────────
    task_id:    str          # unique per task run
    trace_id:   str          # for distributed tracing correlation

    # ── Goal ────────────────────────────────────────────────────────────────
    goal:       str          # original, immutable user request
    goal_validated: bool     # True once input guardrail has passed it

    # ── Messages (LangGraph manages this list) ───────────────────────────────
    messages:   list[BaseMessage]   # complete conversation including tool calls

    # ── Planning ────────────────────────────────────────────────────────────
    plan:       list[dict[str, Any]]  # [{step_number, action, tool, expected_outcome, depends_on}]
    plan_index: int                   # current step being executed
    completed_steps: dict[int, str]   # step_number → result

    # ── Loop control ─────────────────────────────────────────────────────────
    iteration:      int
    exit_reason:    str | None   # goal_achieved | max_iterations | blocked |
                                 # cost_exceeded | unrecoverable_error | human_stopped
    final_answer:   str | None

    # ── Tool tracking ────────────────────────────────────────────────────────
    tool_counts:    dict[str, int]    # tool_name → call count
    tool_errors:    list[dict]        # [{iteration, tool, error, recoverable}]
    last_actions:   list[str]         # rolling window of last 3 actions (oscillation detection)

    # ── Cost and token tracking ───────────────────────────────────────────────
    total_tokens:   int
    total_cost_usd: float
    per_iter_tokens: list[int]        # token count per iteration

    # ── Memory ──────────────────────────────────────────────────────────────
    episodic:   list[dict[str, str]]  # [{role, content}] — persists across sessions
    scratchpad: dict[str, str]        # temporary task-scoped working notes

    # ── Audit ────────────────────────────────────────────────────────────────
    audit_events: list[dict]          # structured audit trail for compliance


def initial_state(goal: str, task_id: str, trace_id: str,
                  episodic: list | None = None) -> AgentState:
    """
    Create a clean initial state for a new task run.
    Pass episodic history to carry forward memory from previous sessions.
    """
    return AgentState(
        task_id=task_id,
        trace_id=trace_id,
        goal=goal,
        goal_validated=False,
        messages=[],
        plan=[],
        plan_index=0,
        completed_steps={},
        iteration=0,
        exit_reason=None,
        final_answer=None,
        tool_counts={},
        tool_errors=[],
        last_actions=[],
        total_tokens=0,
        total_cost_usd=0.0,
        per_iter_tokens=[],
        episodic=episodic or [],
        scratchpad={},
        audit_events=[],
    )
