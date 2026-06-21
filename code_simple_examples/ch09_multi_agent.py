"""
Chapter 9: Multi-Agent System — LangGraph
==========================================
Orchestrator-worker pattern.
Orchestrator decomposes goal → spawns specialist workers → merges results.

Run:  python ch09_multi_agent.py
Need: ollama pull qwen2.5:14b
      pip install langgraph langchain-ollama
"""

import json
import re
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

MODEL = "qwen2.5:7b"
llm   = ChatOllama(model=MODEL, temperature=0)


def extract_json_array(text: str):
    """Pull the first JSON array out of a model reply; None if not parseable."""
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        candidate = m.group(0) if m else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ─── WORKER TOOLS ────────────────────────────────────────────────────────────

@tool
def search_competitor_pricing(competitor: str) -> str:
    """Search for a competitor's current pricing information."""
    data = {
        "Alpha": "Enterprise: $350/seat/month. Startup: $99/month flat.",
        "Beta":  "Freemium (5 users). Enterprise: $199/seat/month.",
        "Gamma": "Usage-based: $0.02/API call. Min $99/month.",
    }
    return data.get(competitor, f"No data found for {competitor}.")

@tool
def analyse_market_position(competitor: str) -> str:
    """Retrieve market share and positioning data for a competitor."""
    data = {
        "Alpha": "28% market share. Positioned as premium enterprise.",
        "Beta":  "19% market share. Positioned as SMB-friendly.",
        "Gamma": "14% market share. Positioned as developer-first.",
    }
    return data.get(competitor, f"No positioning data for {competitor}.")

@tool
def write_section(title: str, content: str) -> str:
    """Write a report section to the output file."""
    line = f"\n## {title}\n{content}\n"
    with open("competitive_report.md", "a") as f:
        f.write(line)
    return f"Section '{title}' written ({len(content)} chars)."


# ─── STATE ───────────────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    goal:         str
    subtasks:     list[dict]       # [{worker, task, result}]
    results:      dict[str, str]   # worker_id → result
    final_report: str | None
    iteration:    int


# ─── ORCHESTRATOR NODE ───────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are an orchestrator agent. Your job is to:
1. Decompose the goal into subtasks for specialist workers.
2. Assign each subtask to the right worker type.
3. Synthesise worker results into a final report.

Workers available:
  - pricing_worker:     retrieves competitor pricing
  - positioning_worker: retrieves market positioning
  - writing_worker:     writes sections of the report

Respond with JSON only when decomposing. Respond with prose when synthesising."""

def node_orchestrator(state: MultiAgentState) -> MultiAgentState:
    """Orchestrator: decompose goal into worker assignments."""
    print(f"\n[Orchestrator] Decomposing goal: {state['goal'][:60]}")

    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=(
            f"Goal: {state['goal']}\n\n"
            "Decompose into a JSON array of subtasks. Create ONE subtask per "
            "competitor per aspect — never combine two competitors in a single "
            "subtask (the worker tools accept only one competitor at a time). "
            "Each subtask: {\"worker\": "
            '"pricing_worker|positioning_worker|writing_worker", '
            '"task": "specific instruction", "competitor": "Alpha|Beta|Gamma"}.\n'
            "Example for Alpha and Beta:\n"
            '[{"worker": "pricing_worker", "task": "Get Alpha pricing", '
            '"competitor": "Alpha"}, {"worker": "pricing_worker", "task": '
            '"Get Beta pricing", "competitor": "Beta"}, {"worker": '
            '"positioning_worker", "task": "Get Alpha positioning", '
            '"competitor": "Alpha"}, {"worker": "positioning_worker", "task": '
            '"Get Beta positioning", "competitor": "Beta"}]'
        )),
    ])

    subtasks = extract_json_array(response.content)
    # Validate the decomposition; fall back to a known-good plan if the small
    # model returned prose or malformed JSON.
    if (not isinstance(subtasks, list) or not subtasks or
            not all(isinstance(s, dict) and "worker" in s for s in subtasks)):
        subtasks = [
            {"worker": "pricing_worker",     "task": "Get Alpha pricing",     "competitor": "Alpha"},
            {"worker": "pricing_worker",     "task": "Get Beta pricing",      "competitor": "Beta"},
            {"worker": "positioning_worker", "task": "Get Alpha positioning",  "competitor": "Alpha"},
            {"worker": "positioning_worker", "task": "Get Beta positioning",   "competitor": "Beta"},
        ]

    print(f"  Decomposed into {len(subtasks)} subtasks")
    for st in subtasks:
        print(f"    → {st['worker']}: {st['task'][:50]}")

    return {**state, "subtasks": subtasks, "results": {}, "iteration": 1}


# ─── WORKER NODES ────────────────────────────────────────────────────────────

WORKER_TOOLS = {
    "pricing_worker":     llm.bind_tools([search_competitor_pricing]),
    "positioning_worker": llm.bind_tools([analyse_market_position]),
    "writing_worker":     llm.bind_tools([write_section]),
}

def _run_worker(worker_type: str, task: str, competitor: str) -> str:
    """Run a single worker agent on its assigned subtask."""
    bound_llm = WORKER_TOOLS.get(worker_type, llm)
    print(f"\n  [Worker: {worker_type}] {task[:50]}")

    response = bound_llm.invoke([
        SystemMessage(content=f"You are a specialist {worker_type}. Complete your task precisely."),
        HumanMessage(content=f"Task: {task}\nCompetitor: {competitor}"),
    ])

    # Execute tool calls if any. A small model sometimes emits arguments that
    # don't match the tool schema, so invoke defensively and recover with the
    # subtask's competitor rather than letting one bad call crash the graph.
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc.get("args", {}) or {}
            for t in [search_competitor_pricing, analyse_market_position, write_section]:
                if t.name == name:
                    try:
                        result = t.invoke(args)
                    except Exception:
                        # Retry with a clean single-competitor argument.
                        try:
                            result = t.invoke({"competitor": competitor})
                        except Exception as e:
                            result = f"(worker could not run {name}: {e})"
                    print(f"    Tool {name} → {str(result)[:80]}")
                    return str(result)

    return response.content or "(no result)"


def node_run_workers(state: MultiAgentState) -> MultiAgentState:
    """Execute all worker subtasks (sequentially for simplicity; parallelise in production)."""
    results = {}
    for i, subtask in enumerate(state["subtasks"]):
        key = f"{subtask['worker']}_{i}"
        result = _run_worker(
            subtask["worker"],
            subtask["task"],
            subtask.get("competitor", "N/A"),
        )
        results[key] = result
        # Trust hierarchy: validate result before accepting
        if len(result) < 5:
            results[key] = f"[Worker {key} produced insufficient output]"

    return {**state, "results": results}


# ─── SYNTHESIS NODE ───────────────────────────────────────────────────────────

def node_synthesise(state: MultiAgentState) -> MultiAgentState:
    """Orchestrator synthesises all worker results into final report."""
    print(f"\n[Orchestrator] Synthesising {len(state['results'])} results")

    results_text = "\n".join(
        f"{k}: {v}" for k, v in state["results"].items()
    )

    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=(
            f"Goal: {state['goal']}\n\n"
            f"Worker results:\n{results_text}\n\n"
            "Synthesise these results into a coherent executive summary written "
            "in plain prose paragraphs. Do NOT return JSON."
        )),
    ])

    report = response.content
    print(f"\n[Final Report]\n{report[:400]}")
    return {**state, "final_report": report}


# ─── GRAPH ───────────────────────────────────────────────────────────────────

def build_multi_agent_graph() -> Any:
    g = StateGraph(MultiAgentState)
    g.add_node("orchestrator", node_orchestrator)
    g.add_node("workers",      node_run_workers)
    g.add_node("synthesise",   node_synthesise)
    g.set_entry_point("orchestrator")
    g.add_edge("orchestrator", "workers")
    g.add_edge("workers",      "synthesise")
    g.add_edge("synthesise",   END)
    return g.compile()


if __name__ == "__main__":
    import os
    if os.path.exists("competitive_report.md"):
        os.remove("competitive_report.md")

    graph = build_multi_agent_graph()
    result = graph.invoke({
        "goal": "Produce a competitive analysis covering pricing and market positioning "
                "for competitors Alpha and Beta.",
        "subtasks": [], "results": {}, "final_report": None, "iteration": 0,
    })
    print(f"\nFinal report:\n{result['final_report']}")
