"""
Chapter 12: Observability and Cost Management — LangGraph
===========================================================
Token tracking, latency measurement, cost estimation,
context compression, and model-tier routing.
"""

import time
import uuid
from dataclasses import dataclass, field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Standardised on qwen2.5:7b for this book. In production, model_router below
# would point the "frontier" tier at a larger model (e.g. a 70B or a hosted
# model) and the "cheap" tier at a small local model — the routing logic and
# the cost/latency tracking are identical regardless of the exact models.
FRONTIER_MODEL = "qwen2.5:7b"
CHEAP_MODEL    = "qwen2.5:7b"

@dataclass
class Metrics:
    task_id:         str
    iterations:      int = 0
    total_tokens:    int = 0
    tool_calls:      int = 0
    latencies_ms:    list[float] = field(default_factory=list)
    tool_latencies:  dict[str, list[float]] = field(default_factory=dict)
    cost_usd:        float = 0.0
    events:          list[dict] = field(default_factory=list)

    # Cost: Qwen2.5 via Ollama = free locally
    # These rates simulate cloud pricing for cost awareness
    COST_PER_1K_TOKENS = 0.002

    def record_llm_call(self, prompt_tokens: int, completion_tokens: int, latency_ms: float):
        total = prompt_tokens + completion_tokens
        self.total_tokens   += total
        self.cost_usd       += (total / 1000) * self.COST_PER_1K_TOKENS
        self.latencies_ms.append(latency_ms)
        self.iterations     += 1
        self.events.append({
            "type": "llm", "tokens": total,
            "latency_ms": latency_ms, "iter": self.iterations
        })

    def record_tool_call(self, tool_name: str, latency_ms: float):
        self.tool_calls += 1
        if tool_name not in self.tool_latencies:
            self.tool_latencies[tool_name] = []
        self.tool_latencies[tool_name].append(latency_ms)

    def p95_latency(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sl = sorted(self.latencies_ms)
        return sl[int(len(sl) * 0.95)]

    def report(self):
        print(f"\n{'─' * 50}")
        print(f"OBSERVABILITY REPORT — Task {self.task_id}")
        print(f"{'─' * 50}")
        print(f"  Iterations:     {self.iterations}")
        print(f"  Total tokens:   {self.total_tokens:,}")
        print(f"  Est. cost:      ${self.cost_usd:.4f} (cloud equiv.)")
        print(f"  Tool calls:     {self.tool_calls}")
        if self.latencies_ms:
            avg = sum(self.latencies_ms) / len(self.latencies_ms)
            print(f"  Avg latency:    {avg:.0f}ms")
            print(f"  P95 latency:    {self.p95_latency():.0f}ms")
        if self.tool_latencies:
            print("  Tool latencies:")
            for t, lats in self.tool_latencies.items():
                print(f"    {t}: avg {sum(lats)/len(lats):.0f}ms")


def context_compress(messages: list, keep_recent: int = 4) -> list:
    """
    Episodic compression: summarise old turns, keep recent ones in full.
    Reduces context window token consumption.
    """
    if len(messages) <= keep_recent + 1:   # +1 for system message
        return messages

    system_msg   = messages[0]
    recent_msgs  = messages[-(keep_recent):]
    older_msgs   = messages[1:-(keep_recent)]

    if not older_msgs:
        return messages

    # Summarise older history
    llm_summarise = ChatOllama(model=CHEAP_MODEL, temperature=0)
    older_text = "\n".join(
        f"{m.type}: {getattr(m, 'content', '')[:200]}"
        for m in older_msgs
    )
    summary_resp = llm_summarise.invoke([
        HumanMessage(content=(
            "Summarise this conversation history in 2-3 sentences, "
            "preserving key facts and decisions:\n\n" + older_text
        ))
    ])
    summary_msg = SystemMessage(
        content=f"[Compressed history]: {summary_resp.content}"
    )

    compressed = [system_msg, summary_msg] + list(recent_msgs)
    print(f"  [Context compressed: {len(messages)} → {len(compressed)} messages]")
    return compressed


def model_router(step_type: str) -> ChatOllama:
    """
    Route to cheaper model for simple steps,
    frontier model for complex reasoning.
    Cost saving: ~60% on simple steps.
    """
    if step_type in {"summarise", "format", "classify"}:
        return ChatOllama(model=CHEAP_MODEL, temperature=0)
    return ChatOllama(model=FRONTIER_MODEL, temperature=0)


def run_observable_agent(goal: str):
    """
    Run a short multi-turn research session so the observability metrics —
    token counts, per-iteration latency, p95, cost, and context compression —
    are actually exercised across several iterations.
    """
    metrics = Metrics(task_id=str(uuid.uuid4())[:8])
    print(f"\nTask {metrics.task_id}: {goal[:60]}")

    messages = [
        SystemMessage(content="You are a research agent. Be concise."),
        HumanMessage(content=goal),
    ]

    # Follow-up turns simulate a multi-step session (in a real agent these come
    # from the loop's own reasoning / tool observations).
    followups = [
        "List the three biggest adoption drivers.",
        "What are the main risks or blockers?",
        "Which industries are moving fastest?",
        "Give one concrete business implication.",
        "Summarise the whole discussion in three bullets.",
    ]

    response = None
    for iteration, question in enumerate(followups):
        # Compress context once history grows past the recent window.
        if iteration > 0 and iteration % 3 == 0:
            messages = context_compress(messages)

        llm_step = model_router("reason")

        t0 = time.time()
        response = llm_step.invoke(messages)
        elapsed  = (time.time() - t0) * 1000

        # Estimate tokens (rough: 1 token ≈ 0.75 words).
        prompt_tokens     = sum(len(str(getattr(m, "content", "")).split()) * 4 // 3
                                for m in messages)
        completion_tokens = len((response.content or "").split()) * 4 // 3
        metrics.record_llm_call(prompt_tokens, completion_tokens, elapsed)
        print(f"  [iter {iteration+1}] {elapsed:.0f}ms, "
              f"{prompt_tokens + completion_tokens} tokens")

        messages.append(response)
        messages.append(HumanMessage(content=question))   # next turn

    metrics.report()
    return response.content if response is not None else ""


if __name__ == "__main__":
    run_observable_agent(
        "Summarise the key trends in enterprise AI adoption and their business implications."
    )