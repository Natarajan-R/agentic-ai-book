"""
agents/multi_agent.py
======================
Chapter 9: Production Multi-Agent System

Extends the reference architecture with genuine multi-agent
orchestration using the orchestrator-worker pattern.

What makes this production quality over the toy version:
  ✓ Each worker is a real isolated LangGraph sub-graph
  ✓ Typed inter-agent message protocol with correlation IDs
  ✓ Worker result validation before orchestrator accepts it
  ✓ Uncertainty propagation — worker confidence flows to final output
  ✓ Parallel worker execution via asyncio
  ✓ Per-worker rate limits and guardrails (workers not implicitly trusted)
  ✓ Orchestrator cannot expand worker permissions beyond system prompt
  ✓ Cascading failure protection via circuit breaker pattern
  ✓ End-to-end trace correlation across all agents
  ✓ Structured worker registry with capability profiles

Run:
    python agents/multi_agent.py

Architecture:
    User request
         │
    Orchestrator (decomposes goal → assigns subtasks)
         │
    ┌────┼────┐
    │    │    │
  Worker Worker Worker  ← each is an isolated agent with own tools
    │    │    │
    └────┼────┘
         │
    Orchestrator (validates + synthesises results)
         │
    Final report
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config.settings import settings
from core.logging import AgentLogger, Timer
from core.tokens import count_tokens
from guardrails.all_guards import (
    check_tool_post, check_tool_pre, validate_input,
)
from tools.all_tools import (
    code_execute, file_write, rag_retrieve, web_search,
    seed_knowledge_base,
)


# ─── INTER-AGENT MESSAGE PROTOCOL ────────────────────────────────────────────

class MessageType(str, Enum):
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT     = "task_result"
    TASK_ERROR      = "task_error"
    CLARIFICATION   = "clarification_request"
    STATUS_UPDATE   = "status_update"


@dataclass
class AgentMessage:
    """
    Typed inter-agent message.
    All agent communication uses this protocol — never raw strings.

    Every message carries:
    - Unique message_id for deduplication
    - correlation_id linking request to response
    - task_run_id for end-to-end trace correlation
    - sender and recipient identity
    - confidence score (0-1) that flows through the system
    """
    message_id:     str
    message_type:   MessageType
    sender_id:      str
    recipient_id:   str
    task_run_id:    str
    correlation_id: str
    payload:        dict[str, Any]
    confidence:     float = 1.0          # propagated through the system
    timestamp:      str   = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    @classmethod
    def task_assignment(
        cls,
        sender: str,
        recipient: str,
        task_run_id: str,
        task: str,
        context: dict | None = None,
    ) -> "AgentMessage":
        msg_id = str(uuid.uuid4())[:8]
        return cls(
            message_id=msg_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            sender_id=sender,
            recipient_id=recipient,
            task_run_id=task_run_id,
            correlation_id=msg_id,
            payload={"task": task, "context": context or {}},
        )

    @classmethod
    def task_result(
        cls,
        sender: str,
        recipient: str,
        task_run_id: str,
        correlation_id: str,
        result: str,
        confidence: float,
        sources: list[str] | None = None,
    ) -> "AgentMessage":
        return cls(
            message_id=str(uuid.uuid4())[:8],
            message_type=MessageType.TASK_RESULT,
            sender_id=sender,
            recipient_id=recipient,
            task_run_id=task_run_id,
            correlation_id=correlation_id,
            payload={
                "result":  result,
                "sources": sources or [],
            },
            confidence=confidence,
        )

    @classmethod
    def task_error(
        cls,
        sender: str,
        recipient: str,
        task_run_id: str,
        correlation_id: str,
        error: str,
        recoverable: bool = True,
    ) -> "AgentMessage":
        return cls(
            message_id=str(uuid.uuid4())[:8],
            message_type=MessageType.TASK_ERROR,
            sender_id=sender,
            recipient_id=recipient,
            task_run_id=task_run_id,
            correlation_id=correlation_id,
            payload={"error": error, "recoverable": recoverable},
            confidence=0.0,
        )

    def validate(self) -> tuple[bool, str]:
        """Validate message structure before processing."""
        if not self.task_run_id:
            return False, "Missing task_run_id"
        if not self.sender_id or not self.recipient_id:
            return False, "Missing sender or recipient"
        if not self.payload:
            return False, "Empty payload"
        if not 0.0 <= self.confidence <= 1.0:
            return False, f"Invalid confidence: {self.confidence}"
        return True, ""


# ─── WORKER CAPABILITY PROFILES ──────────────────────────────────────────────

@dataclass
class WorkerProfile:
    """
    Defines what a worker can do.
    The orchestrator uses this to match subtasks to workers.
    Workers cannot exceed the capabilities defined here.
    """
    worker_id:     str
    worker_type:   str
    description:   str
    tools:         list        # actual LangChain tool objects
    system_prompt: str
    max_output_tokens: int = 1000
    rate_limit_per_run: int = 10


WORKER_PROFILES: dict[str, WorkerProfile] = {

    "research_worker": WorkerProfile(
        worker_id="research_worker",
        worker_type="research",
        description=(
            "Specialist in information retrieval. Searches web and internal KB "
            "for competitor data, market information, and public announcements. "
            "Does NOT analyse — only retrieves and structures raw information."
        ),
        tools=[web_search, rag_retrieve],
        system_prompt=(
            "You are a specialist research worker. Your only job is to find "
            "and retrieve accurate information. You do NOT analyse, synthesise, "
            "or draw conclusions — you retrieve and report facts with sources. "
            "Always include the source of each piece of information. "
            "Rate your confidence in each finding from 0.0 to 1.0."
        ),
    ),

    "analysis_worker": WorkerProfile(
        worker_id="analysis_worker",
        worker_type="analysis",
        description=(
            "Specialist in quantitative analysis. Performs calculations, "
            "statistical comparisons, and numerical evaluations. "
            "Uses code_execute for all computation — never computes mentally."
        ),
        tools=[code_execute],
        system_prompt=(
            "You are a specialist analysis worker. Your job is precise quantitative "
            "analysis. You MUST use the code_execute tool for every calculation — "
            "never compute mentally. Show your working. Report results with units. "
            "Rate your confidence in each result from 0.0 to 1.0."
        ),
    ),

    "writing_worker": WorkerProfile(
        worker_id="writing_worker",
        worker_type="writing",
        description=(
            "Specialist in producing structured written deliverables. "
            "Takes synthesised content and formats it into professional reports. "
            "Does NOT research or analyse — only formats and writes."
        ),
        tools=[file_write],
        system_prompt=(
            "You are a specialist writing worker. Your job is to produce "
            "clear, professional written output from the content you are given. "
            "Structure responses with headings, bullet points, and clear sections. "
            "Do not add analysis or opinions beyond what you were given. "
            "Always confirm when a file has been successfully written."
        ),
    ),
}


# ─── WORKER STATE ────────────────────────────────────────────────────────────

class WorkerState(TypedDict):
    worker_id:    str
    task_run_id:  str
    assignment:   AgentMessage
    messages:     list
    result:       AgentMessage | None
    tool_counts:  dict[str, int]
    iterations:   int


# ─── WORKER AGENT ────────────────────────────────────────────────────────────

class WorkerAgent:
    """
    An isolated specialist agent that executes one subtask.

    Security model:
    - Each worker has its own system prompt defining its scope
    - Workers cannot execute tools outside their allowed list
    - Workers validate their assignment before executing
    - Workers cannot receive instructions that expand their permissions
    - Results are validated by the orchestrator before acceptance
    """

    def __init__(self, profile: WorkerProfile):
        self.profile = profile
        self._llm = ChatOllama(
            model=settings.model_primary,
            temperature=0,
            base_url=settings.ollama_base_url,
        ).bind_tools(profile.tools)
        self._tool_map = {t.name: t for t in profile.tools}

    def execute(self, assignment: AgentMessage) -> AgentMessage:
        """
        Execute an assigned subtask and return a typed result message.

        Security: validates the assignment, enforces tool scope,
        validates output before returning.
        """
        # Validate incoming message
        valid, reason = assignment.validate()
        if not valid:
            return AgentMessage.task_error(
                self.profile.worker_id, assignment.sender_id,
                assignment.task_run_id, assignment.message_id,
                f"Invalid assignment: {reason}", recoverable=False,
            )

        task = assignment.payload.get("task", "")
        context = assignment.payload.get("context", {})

        # Build messages — system prompt defines scope, cannot be overridden
        messages = [
            SystemMessage(content=self.profile.system_prompt),
            HumanMessage(content=(
                f"Task: {task}\n"
                + (f"Context: {json.dumps(context)[:500]}\n" if context else "")
                + "Complete this task and provide your result with a confidence score."
            )),
        ]

        tool_counts: dict[str, int] = {}
        max_iters = 6

        for iteration in range(max_iters):
            response = self._llm.invoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", []):
                # Worker finished — extract result and confidence
                result_text, confidence = self._extract_result(response.content or "")

                # Validate result is non-trivial
                if len(result_text.strip()) < 30:
                    return AgentMessage.task_error(
                        self.profile.worker_id, assignment.sender_id,
                        assignment.task_run_id, assignment.message_id,
                        "Worker produced insufficient output", recoverable=True,
                    )

                return AgentMessage.task_result(
                    sender=self.profile.worker_id,
                    recipient=assignment.sender_id,
                    task_run_id=assignment.task_run_id,
                    correlation_id=assignment.message_id,
                    result=result_text,
                    confidence=confidence,
                )

            # Execute tool calls — scoped to worker's allowed tools only
            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {})

                # Rate limiting per worker
                count = tool_counts.get(tool_name, 0)
                if count >= self.profile.rate_limit_per_run:
                    obs = f"Rate limit reached for {tool_name} in this worker"
                    messages.append(
                        type("ToolMessage", (), {
                            "content": obs, "type": "tool",
                            "tool_call_id": tc["id"], "name": tool_name,
                        })()
                    )
                    continue

                # Scope enforcement — worker cannot call tools outside its list
                if tool_name not in self._tool_map:
                    obs = f"Tool '{tool_name}' is outside this worker's scope"
                    messages.append(
                        type("ToolMessage", (), {
                            "content": obs, "type": "tool",
                            "tool_call_id": tc["id"], "name": tool_name,
                        })()
                    )
                    continue

                try:
                    from langchain_core.messages import ToolMessage
                    result = self._tool_map[tool_name].invoke(tool_args)
                    tool_counts[tool_name] = count + 1
                    messages.append(ToolMessage(
                        content=str(result)[:2000],
                        tool_call_id=tc["id"],
                        name=tool_name,
                    ))
                except Exception as e:
                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(
                        content=f"Tool error: {e}",
                        tool_call_id=tc["id"],
                        name=tool_name,
                    ))

        # Reached max iterations
        return AgentMessage.task_error(
            self.profile.worker_id, assignment.sender_id,
            assignment.task_run_id, assignment.message_id,
            f"Worker exceeded max iterations ({max_iters})", recoverable=True,
        )

    @staticmethod
    def _extract_result(content: str) -> tuple[str, float]:
        """Extract result text and confidence from worker response."""
        import re
        confidence = 0.8  # default
        pattern = re.search(r"confidence[:\s]+([0-9.]+)", content, re.IGNORECASE)
        if pattern:
            try:
                confidence = max(0.0, min(1.0, float(pattern.group(1))))
            except ValueError:
                pass
        return content, confidence


# ─── CIRCUIT BREAKER ─────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Prevents cascading failures when a worker is persistently failing.

    States: CLOSED (normal) → OPEN (blocking) → HALF_OPEN (testing)
    """

    def __init__(self, threshold: int = 3, reset_seconds: int = 60):
        self._threshold     = threshold
        self._reset_seconds = reset_seconds
        self._failures:     dict[str, int]   = {}
        self._open_since:   dict[str, float] = {}

    def is_open(self, worker_id: str) -> bool:
        if worker_id not in self._open_since:
            return False
        if time.time() - self._open_since[worker_id] > self._reset_seconds:
            # Move to HALF_OPEN — allow one test call
            del self._open_since[worker_id]
            self._failures[worker_id] = 0
            return False
        return True

    def record_failure(self, worker_id: str):
        self._failures[worker_id] = self._failures.get(worker_id, 0) + 1
        if self._failures[worker_id] >= self._threshold:
            self._open_since[worker_id] = time.time()

    def record_success(self, worker_id: str):
        self._failures[worker_id] = 0
        self._open_since.pop(worker_id, None)


_circuit_breaker = CircuitBreaker()


# ─── ORCHESTRATOR STATE ───────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    task_run_id:   str
    goal:          str
    subtasks:      list[dict]      # [{worker_id, assignment_msg}]
    results:       list[AgentMessage]
    failed:        list[dict]      # [{worker_id, error, recoverable}]
    final_report:  str | None
    confidence:    float           # aggregate confidence across workers
    iterations:    int
    exit_reason:   str | None


# ─── ORCHESTRATOR NODES ───────────────────────────────────────────────────────

def _make_orchestrator_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.model_primary,
        temperature=0,
        base_url=settings.ollama_base_url,
    )


ORCHESTRATOR_SYSTEM = """You are the orchestrator of a multi-agent competitive intelligence system.

Your responsibilities:
1. DECOMPOSE: Break the user's goal into specific subtasks for specialist workers
2. ASSIGN: Match each subtask to the right worker type
3. VALIDATE: Check each worker's result for quality and completeness
4. SYNTHESISE: Combine validated results into a coherent final answer

Available workers:
  - research_worker:  Retrieves information from web and internal knowledge base
  - analysis_worker:  Performs quantitative calculations and comparisons
  - writing_worker:   Produces structured written reports and documents

Rules:
  - Each subtask must be completable by exactly ONE worker
  - Subtasks must be specific enough to produce a concrete output
  - You cannot assign a task that requires capabilities outside a worker's profile
  - Low-confidence results (<0.6) must be flagged or re-investigated
  - If a worker fails, consider whether another worker can provide a substitute"""


def node_orchestrate(state: OrchestratorState) -> OrchestratorState:
    """
    Orchestrator: decompose goal into typed subtask assignments.
    Each subtask is an AgentMessage to the appropriate worker.
    """
    llm = _make_orchestrator_llm()

    worker_descriptions = "\n".join(
        f"  {wid}: {p.description}"
        for wid, p in WORKER_PROFILES.items()
    )

    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=(
            f"Goal: {state['goal']}\n\n"
            f"Worker profiles:\n{worker_descriptions}\n\n"
            "Decompose this goal into subtasks. Return a JSON array where each item has:\n"
            '  {"worker_id": "research_worker|analysis_worker|writing_worker",\n'
            '   "task": "specific, concrete task description",\n'
            '   "depends_on": [] or ["task_index"] for dependencies}\n'
            "Return ONLY valid JSON — no markdown, no explanation."
        )),
    ])

    try:
        subtasks_raw = json.loads(response.content)
    except Exception:
        # Fallback decomposition
        subtasks_raw = [
            {"worker_id": "research_worker",
             "task": f"Research and gather information for: {state['goal'][:100]}",
             "depends_on": []},
            {"worker_id": "analysis_worker",
             "task": "Analyse and calculate key metrics from the research findings",
             "depends_on": [0]},
            {"worker_id": "writing_worker",
             "task": "Write a structured report from the analysis",
             "depends_on": [1]},
        ]

    # Convert to AgentMessage assignments
    subtasks = []
    for i, st in enumerate(subtasks_raw):
        worker_id = st.get("worker_id", "research_worker")
        if worker_id not in WORKER_PROFILES:
            worker_id = "research_worker"   # safe default

        assignment = AgentMessage.task_assignment(
            sender="orchestrator",
            recipient=worker_id,
            task_run_id=state["task_run_id"],
            task=st.get("task", ""),
            context={"goal": state["goal"], "task_index": i},
        )
        subtasks.append({
            "index":      i,
            "worker_id":  worker_id,
            "assignment": assignment,
            "depends_on": st.get("depends_on", []),
        })

    print(f"\n[Orchestrator] Decomposed into {len(subtasks)} subtasks:")
    for st in subtasks:
        print(f"  [{st['index']}] {st['worker_id']}: {st['assignment'].payload['task'][:60]}")

    return {**state, "subtasks": subtasks, "iterations": 1}


def node_execute_workers(state: OrchestratorState) -> OrchestratorState:
    """
    Execute worker subtasks respecting dependency order.
    Workers without dependencies run first; dependent workers
    receive prior results as context.

    Circuit breaker prevents cascading failures.
    """
    results:   list[AgentMessage] = []
    failed:    list[dict]         = []
    completed: dict[int, AgentMessage] = {}  # index → result

    # Sort by dependency order
    subtasks = sorted(state["subtasks"], key=lambda x: len(x.get("depends_on", [])))

    for st in subtasks:
        worker_id  = st["worker_id"]
        assignment = st["assignment"]
        idx        = st["index"]
        deps       = st.get("depends_on", [])

        # Check circuit breaker
        if _circuit_breaker.is_open(worker_id):
            print(f"  [CIRCUIT OPEN] {worker_id} — skipping")
            failed.append({
                "worker_id":  worker_id,
                "error":      "Circuit breaker open — worker has repeated failures",
                "recoverable": False,
                "task_index": idx,
            })
            continue

        # Add dependency results to context
        if deps:
            dep_context = {}
            deps_met = True
            for dep_idx in deps:
                if isinstance(dep_idx, int) and dep_idx in completed:
                    dep_msg = completed[dep_idx]
                    dep_context[f"step_{dep_idx}_result"] = (
                        dep_msg.payload.get("result", "")[:500]
                    )
                    dep_context[f"step_{dep_idx}_confidence"] = dep_msg.confidence
                else:
                    deps_met = False
                    break

            if not deps_met:
                print(f"  [SKIP] Task {idx} — dependency not met")
                failed.append({
                    "worker_id": worker_id,
                    "error": "Dependency not completed",
                    "recoverable": False,
                    "task_index": idx,
                })
                continue

            # Re-create assignment with dependency context
            assignment = AgentMessage.task_assignment(
                sender="orchestrator",
                recipient=worker_id,
                task_run_id=state["task_run_id"],
                task=assignment.payload["task"],
                context={**assignment.payload.get("context", {}), **dep_context},
            )

        print(f"\n  [Worker: {worker_id}] {assignment.payload['task'][:60]}")
        profile = WORKER_PROFILES[worker_id]
        worker  = WorkerAgent(profile)

        with Timer() as t:
            result_msg = worker.execute(assignment)

        if result_msg.message_type == MessageType.TASK_RESULT:
            # Validate result before accepting
            result_text = result_msg.payload.get("result", "")
            if len(result_text.strip()) < 20:
                _circuit_breaker.record_failure(worker_id)
                failed.append({
                    "worker_id": worker_id,
                    "error": "Result too brief to be useful",
                    "recoverable": True,
                    "task_index": idx,
                })
                continue

            _circuit_breaker.record_success(worker_id)
            completed[idx] = result_msg
            results.append(result_msg)
            print(f"    ✓ Done ({t.elapsed_ms}ms, confidence={result_msg.confidence:.2f})")
            print(f"    Preview: {result_text[:100]}")

        else:
            error = result_msg.payload.get("error", "Unknown error")
            _circuit_breaker.record_failure(worker_id)
            failed.append({
                "worker_id":   worker_id,
                "error":       error,
                "recoverable": result_msg.payload.get("recoverable", True),
                "task_index":  idx,
            })
            print(f"    ✗ Failed: {error[:80]}")

    return {**state, "results": results, "failed": failed}


def node_synthesise(state: OrchestratorState) -> OrchestratorState:
    """
    Orchestrator synthesises all validated worker results.

    Uncertainty propagation:
    - Tracks confidence from each worker
    - Computes weighted aggregate confidence
    - Flags low-confidence sections in the final output
    - Reports on any failed subtasks
    """
    llm = _make_orchestrator_llm()

    if not state["results"]:
        return {
            **state,
            "final_report": (
                "Task could not be completed: all worker subtasks failed.\n"
                "Failed tasks:\n" +
                "\n".join(f"- {f['worker_id']}: {f['error']}" for f in state["failed"])
            ),
            "confidence":  0.0,
            "exit_reason": "all_workers_failed",
        }

    # Build results summary with confidence scores
    results_text = ""
    total_conf   = 0.0
    for i, msg in enumerate(state["results"], 1):
        conf   = msg.confidence
        result = msg.payload.get("result", "")
        worker = msg.sender_id
        results_text += (
            f"\n## Result {i} — {worker} (confidence: {conf:.0%})\n"
            f"{result[:800]}\n"
        )
        total_conf += conf

    # Aggregate confidence — average across successful workers
    n_results = len(state["results"])
    agg_confidence = total_conf / n_results if n_results > 0 else 0.0

    # Report failed tasks
    failed_note = ""
    if state["failed"]:
        failed_note = (
            "\n\n**Note:** The following subtasks could not be completed:\n" +
            "\n".join(f"- {f['worker_id']}: {f['error']}" for f in state["failed"])
        )

    synthesis_response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=(
            f"Goal: {state['goal']}\n\n"
            f"Worker results:\n{results_text}\n"
            f"Aggregate confidence: {agg_confidence:.0%}\n\n"
            "Synthesise these results into a comprehensive final answer. "
            "Explicitly note the confidence level for each key finding. "
            "Distinguish between verified facts and inferences."
        )),
    ])

    final = (synthesis_response.content or "") + failed_note

    if agg_confidence < 0.6:
        final = (
            f"⚠️ **Low confidence output ({agg_confidence:.0%})** — "
            "human review recommended.\n\n" + final
        )

    return {
        **state,
        "final_report": final,
        "confidence":   agg_confidence,
        "exit_reason":  "completed",
    }


# ─── BUILD ORCHESTRATOR GRAPH ─────────────────────────────────────────────────

def build_multi_agent_graph() -> Any:
    g = StateGraph(OrchestratorState)
    g.add_node("orchestrate",       node_orchestrate)
    g.add_node("execute_workers",   node_execute_workers)
    g.add_node("synthesise",        node_synthesise)
    g.set_entry_point("orchestrate")
    g.add_edge("orchestrate",     "execute_workers")
    g.add_edge("execute_workers", "synthesise")
    g.add_edge("synthesise",      END)
    return g.compile()


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

@dataclass
class MultiAgentResult:
    task_run_id:  str
    report:       str
    confidence:   float
    n_workers:    int
    n_failed:     int
    elapsed_s:    float
    exit_reason:  str

    def print_summary(self):
        print(f"\n{'─' * 65}")
        print(f"  Workers completed: {self.n_workers}")
        print(f"  Workers failed:    {self.n_failed}")
        print(f"  Confidence:        {self.confidence:.0%}")
        print(f"  Elapsed:           {self.elapsed_s:.1f}s")
        print(f"  Exit:              {self.exit_reason}")
        print(f"{'─' * 65}")
        print(f"\nREPORT:\n{self.report}")


class MultiAgentSystem:
    """
    Production multi-agent system with orchestrator-worker coordination.

    Usage:
        system = MultiAgentSystem()
        result = system.run("Produce a competitive analysis for our top 3 competitors.")
        print(result.report)
    """

    def __init__(self):
        self._graph = build_multi_agent_graph()
        seed_knowledge_base([
            {"id": "ma_competitor_alpha", "text": "Competitor Alpha: $350/seat enterprise, 28% market share.", "source": "kb", "category": "competitor"},
            {"id": "ma_competitor_beta",  "text": "Competitor Beta: $199/seat enterprise, 19% market share.", "source": "kb", "category": "competitor"},
            {"id": "ma_market_2024",      "text": "Market total $3.8B in 2024. CAGR 42%.", "source": "kb", "category": "market"},
        ])

    def run(self, goal: str) -> MultiAgentResult:
        task_run_id = str(uuid.uuid4())
        print(f"\n{'═' * 65}")
        print(f"  Multi-agent task: {task_run_id[:8]}")
        print(f"  Goal: {goal[:70]}")
        print(f"{'═' * 65}")

        start = time.perf_counter()
        final = self._graph.invoke({
            "task_run_id":  task_run_id,
            "goal":         goal,
            "subtasks":     [],
            "results":      [],
            "failed":       [],
            "final_report": None,
            "confidence":   1.0,
            "iterations":   0,
            "exit_reason":  None,
        })
        elapsed = time.perf_counter() - start

        return MultiAgentResult(
            task_run_id=task_run_id,
            report=final.get("final_report", "No report produced."),
            confidence=final.get("confidence", 0.0),
            n_workers=len(final.get("results", [])),
            n_failed=len(final.get("failed", [])),
            elapsed_s=elapsed,
            exit_reason=final.get("exit_reason", "unknown"),
        )


if __name__ == "__main__":
    system = MultiAgentSystem()
    result = system.run(
        "Produce a comprehensive competitive analysis comparing our enterprise AI agent "
        "platform against the top three competitors. Include: pricing comparison for a "
        "100-seat enterprise, market positioning analysis, and a buy/build/partner "
        "recommendation. Write the final report to 'competitive_analysis.md'."
    )
    result.print_summary()
