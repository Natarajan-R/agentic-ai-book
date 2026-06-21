"""
Chapter 5: Sequential Planning
================================
Upfront plan generation -> step-by-step execution -> outcome validation
-> replanning (step retry) on failure -> synthesis.

Run:  python ch05_planning.py
Need: ollama pull qwen2.5:7b

What this example demonstrates (mapping to the chapter)
------------------------------------------------------
  * 5.5  A well-formed plan step has four parts: action, tool,
         expected_outcome, depends_on.
  * 5.6  Replanning as STEP RETRY: when a step's outcome is not met, the
         agent refines the step and tries again rather than giving up.
  * 5.7  Outcome validation: deciding whether a step is "done".
  * 5.4  A final REVIEW step closes the plan.

Reliability choices
-------------------
The original temptation is to validate each step by asking the LLM
"did this meet the goal? yes/no". A small local model answers that
inconsistently, which (in the naive version of this code) made EVERY
step fail and the whole plan collapse. So here outcome validation is a
deterministic RULE, and we stage exactly one realistic failure (the
first web search returns nothing) so you can watch the retry mechanism
work the same way on every run.
"""

import json
import re
from dataclasses import dataclass, field
import ollama

MODEL = "qwen2.5:7b"


# ─── PLAN DATA STRUCTURE (the four components from section 5.5) ───────────────

@dataclass
class PlanStep:
    step_number:      int
    action:           str
    tool:             str          # web_search | retrieve | calculate | write
    expected_outcome: str
    depends_on:       list[int] = field(default_factory=list)


def extract_json_array(text: str):
    """Pull the first JSON array out of a model response; None if none found."""
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


# ─── PLANNING LAYER ──────────────────────────────────────────────────────────

FALLBACK_PLAN = [
    PlanStep(1, "Search the web for the top three cloud AI platforms and their "
                "recent pricing", "web_search",
             "Pricing facts for three platforms"),
    PlanStep(2, "Retrieve the internal competitive-analysis notes", "retrieve",
             "Internal notes on competitor positioning", depends_on=[1]),
    PlanStep(3, "Write a short comparison with a recommendation", "write",
             "A concise comparison naming a recommended platform",
             depends_on=[1, 2]),
    PlanStep(4, "Review the comparison against the goal", "write",
             "Confirmation the goal's requirements are all covered",
             depends_on=[3]),
]


# Tools the planner may use in this demo. (A precise `calculate` tool is the
# subject of Chapter 6; this competitive-analysis task does not need it, and
# leaving it out keeps every generated plan executable by the simulated tools.)
VALID_TOOLS = {"web_search", "retrieve", "write"}


def _coerce_step(raw: dict, index: int) -> PlanStep | None:
    """
    Build a PlanStep from one raw dict, tolerating extra or misspelled keys
    (small models are sloppy). Returns None if the step lacks the essentials.
    """
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action", "")).strip()
    tool = raw.get("tool", "")
    if tool not in VALID_TOOLS or len(action) < 8 or action == tool:
        return None                       # not specific enough to be a real step
    deps = [int(d) for d in raw.get("depends_on", []) if str(d).isdigit()]
    return PlanStep(
        step_number=int(raw.get("step_number", index)),
        action=action,
        tool=tool,
        expected_outcome=str(raw.get("expected_outcome", "Step completed")),
        depends_on=deps,
    )


def generate_plan(goal: str) -> list[PlanStep]:
    """
    Ask the LLM for a sequential plan as JSON. We tolerate sloppy JSON, but if
    the model gives us nothing well-formed we fall back to a known-good plan so
    execution always has a valid plan to run.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": ("Produce a sequential plan as a JSON array. Each item has: "
                         '"step_number" (int), "action" (a specific instruction, '
                         'not just a tool name), "tool" (one of web_search, '
                         'retrieve, write), "expected_outcome" (str), '
                         '"depends_on" (list of step numbers). Return ONLY JSON.\n'
                         "Example item:\n"
                         '{"step_number": 1, "action": "Search the web for recent '
                         'competitor pricing", "tool": "web_search", '
                         '"expected_outcome": "Pricing facts for each competitor", '
                         '"depends_on": []}')},
            {"role": "user", "content": f"Goal: {goal}"},
        ],
        options={"temperature": 0},
    )
    raw = extract_json_array(response.message.content) or []
    plan = [s for i, s in enumerate((_coerce_step(r, i) for i, r in enumerate(raw, 1)), 1)
            if s is not None]
    if len(plan) >= 2:
        return plan
    print("  (model plan unusable — using built-in fallback plan)")
    return FALLBACK_PLAN


# ─── EXECUTION LAYER ─────────────────────────────────────────────────────────

# Counts web_search calls so the FIRST one can realistically "miss" — this is
# what lets us demonstrate the retry mechanism deterministically.
_web_search_calls = 0


def execute_step(step: PlanStep, context: dict[int, str]) -> str:
    """Execute a single step with simulated-but-deterministic tools."""
    global _web_search_calls

    if step.tool == "web_search":
        _web_search_calls += 1
        if _web_search_calls == 1:
            # Realistic first miss: a too-broad query returns nothing useful.
            return "No relevant results found for that query."
        return ("Web results — cloud AI platform pricing:\n"
                "- Platform A: $350/seat/month, flat enterprise tier\n"
                "- Platform B: usage-based, ~$280/seat equivalent\n"
                "- Platform C: $297/seat/month after a recent 15% cut")

    if step.tool == "retrieve":
        return ("Internal notes — 'Cloud AI Platform Review':\n"
                "- Platform A: strongest security/compliance story\n"
                "- Platform B: best for small, variable teams\n"
                "- Platform C: aggressive pricing, thinner support")

    if step.tool == "calculate":
        try:
            return str(eval(step.action, {"__builtins__": {}}))  # noqa: S307
        except Exception as e:
            return f"Calculation error: {e}"

    # tool == "write": synthesise from the results gathered so far.
    prior = "\n".join(f"Step {k}: {v}" for k, v in context.items()) or "(none)"
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": ("Complete this step using ONLY the provided "
                         "information. Be concise.")},
            {"role": "user",
             "content": f"Step: {step.action}\n\nGathered so far:\n{prior}"},
        ],
        options={"temperature": 0},
    )
    return response.message.content.strip()


# ─── EVALUATION LAYER (section 5.7 — outcome validation) ──────────────────────

def evaluate_step(result: str) -> bool:
    """
    Deterministic outcome check: a step passes if it produced a substantial
    result that is not an explicit miss. (Production systems can use an LLM
    here for richer 'outcome validation'; we keep it a rule so the demo is
    reproducible — see the module docstring.)
    """
    if "No relevant results" in result:
        return False
    return len(result.strip()) >= 40


# ─── REPLANNING (section 5.6 — step retry with a refined action) ──────────────

def refine_step(step: PlanStep) -> PlanStep:
    """Retry strategy: make the action more specific and try the same tool."""
    step.action = (step.action +
                   " — refine: focus on enterprise pricing in the last quarter")
    return step


# ─── MAIN PLANNER ────────────────────────────────────────────────────────────

def run_planner(goal: str, max_retries: int = 1):
    print(f"\n{'=' * 60}\nGOAL: {goal}\n{'=' * 60}")

    print("\n[Phase 1] Generating plan ...")
    plan = generate_plan(goal)
    print(f"Plan has {len(plan)} steps:")
    for s in plan:
        deps = f" (depends on {s.depends_on})" if s.depends_on else ""
        print(f"  Step {s.step_number}: [{s.tool}] {s.action[:60]}{deps}")
        print(f"           expect: {s.expected_outcome}")

    print("\n[Phase 2] Executing plan ...")
    completed: dict[int, str] = {}
    failed:    list[int] = []

    for step in sorted(plan, key=lambda x: x.step_number):
        sn = step.step_number
        print(f"\n  Step {sn}: {step.action[:60]}")

        # Enforce dependencies before executing (section 5.5 / 5.11).
        missing = [d for d in step.depends_on if d not in completed]
        if missing:
            print(f"    [SKIP] Dependencies not met: {missing}")
            failed.append(sn)
            continue

        # Execute, validate, and retry on failure (section 5.6 + 5.7).
        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f"    [REPLAN] Outcome not met — refining and retrying ...")
                step = refine_step(step)

            result = execute_step(step, completed)
            if evaluate_step(result):
                print(f"    [PASS] {result.splitlines()[0][:70]}")
                completed[sn] = result
                break
            print(f"    [FAIL] {result.splitlines()[0][:70]}")
        else:
            print(f"    [ESCALATE] Step {sn} could not be completed")
            failed.append(sn)

    print(f"\n[Phase 3] Synthesis")
    print(f"  Completed: {sorted(completed.keys())}")
    print(f"  Failed:    {failed}")

    if completed:
        synthesis_input = "\n".join(f"Step {k}: {v}" for k, v in completed.items())
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system",
                 "content": "Synthesise these step results into a short, coherent answer."},
                {"role": "user", "content": f"Goal: {goal}\n\nResults:\n{synthesis_input}"},
            ],
            options={"temperature": 0},
        )
        print(f"\n[Final Answer]\n{response.message.content.strip()}")


if __name__ == "__main__":
    run_planner(
        "Produce a brief competitive analysis of cloud AI platforms: compare "
        "their pricing, and recommend which to evaluate for enterprise use."
    )
