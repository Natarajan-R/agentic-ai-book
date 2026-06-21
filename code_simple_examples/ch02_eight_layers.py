"""
Chapter 2: The Eight Layers
============================
One request flows top-to-bottom through eight layers; the reflection
layer decides whether the goal is met or the loop should run again.

This mirrors the running example from the chapter — a competitive-
intelligence agent answering a pricing-summary request — so you can see
each layer described in the text doing real work in code.

Run:  python ch02_eight_layers.py
Need: ollama pull qwen2.5:7b

A note on reliability (important for beginners)
-----------------------------------------------
A 7B model running locally is NOT perfectly reliable at returning
structured JSON or picking tools. So every layer here is written
*defensively*:

  * The planning layer tries to use the model's JSON plan, but falls
    back to a known-good plan if that JSON can't be parsed.
  * The tools return deterministic, realistic data, so execution always
    produces something the next layer can use.
  * The reflection layer uses a simple, reproducible RULE (did every
    step run and did we produce a real answer?) instead of asking the
    model "is this good? yes/no" — a small model answers that
    inconsistently, which would make the demo behave differently every
    run.

This defensive style is exactly how you make agents dependable in
production: never bet the whole system on a single fragile model output.
"""

import json
import re
import ollama

MODEL = "qwen2.5:7b"

# The tools this agent is allowed to use (its "tool registry" — Layer 5).
TOOL_REGISTRY = {"web_search", "rag_retrieve", "write"}


# ─── Small helper: pull a JSON array out of a model response ──────────────────

def extract_json_array(text: str):
    """
    Small models often wrap JSON in ```json fences or add a sentence of
    explanation. This pulls out the first [...] block and parses it.
    Returns None if nothing parseable is found — the caller then uses a
    fallback. This single helper is why the planning layer never crashes.
    """
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        candidate = match.group(0) if match else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# ─── LAYER 1: PERCEPTION ─────────────────────────────────────────────────────

def layer_1_perception(raw_input: str) -> dict:
    """
    Receive raw input and normalise it into a structured form the rest of
    the system can work with consistently — regardless of where it came
    from (chat box, email, webhook, voice transcript ...).
    """
    return {
        "query":  raw_input.strip(),
        "length": len(raw_input.split()),
        "type":   "natural_language",
        "source": "user_interface",
    }


# ─── LAYER 2: MEMORY ─────────────────────────────────────────────────────────

def layer_2_memory(perceived: dict, history: list[dict]) -> list[dict]:
    """
    Assemble the context window. The stateless LLM is handed:
      - a system role (who it is),
      - relevant SEMANTIC memory (a retrieved knowledge-base note),
      - recent EPISODIC memory (the last few turns),
      - the current query.
    In a real system the semantic note comes from a vector search; here it
    is a fixed snippet so the demo is reproducible.
    """
    semantic_note = (
        "[Knowledge base] The company sells an Enterprise tier at "
        "$299/seat/month. Sales has flagged competitor pricing as a "
        "recurring deal blocker."
    )
    messages = [
        {"role": "system",
         "content": ("You are a competitive-intelligence assistant for an "
                     "enterprise software vendor. Be concise and factual.")},
        {"role": "system", "content": semantic_note},
    ]
    messages.extend(history[-4:])                       # episodic memory
    messages.append({"role": "user", "content": perceived["query"]})
    return messages


# ─── LAYER 3: REASONING ──────────────────────────────────────────────────────

def layer_3_reasoning(messages: list[dict]) -> str:
    """
    The LLM interprets the assembled context and states, in one or two
    sentences, what the goal is and how it intends to approach it. This is
    the 'understanding' step — the only place the model decides meaning.
    """
    prompt = messages + [
        {"role": "system",
         "content": ("In ONE or TWO sentences, restate the user's goal and "
                     "the approach you will take. Do not answer it yet.")},
    ]
    response = ollama.chat(model=MODEL, messages=prompt,
                           options={"temperature": 0})
    return response.message.content.strip()


# ─── LAYER 4: PLANNING ───────────────────────────────────────────────────────

FALLBACK_PLAN = [
    {"action": "Search the web for the latest competitor pricing announcements",
     "tool": "web_search"},
    {"action": "Retrieve the internal competitive-analysis document",
     "tool": "rag_retrieve"},
    {"action": "Write a short pricing summary from the gathered information",
     "tool": "write"},
]


def layer_4_planning(goal: str) -> list[dict]:
    """
    Turn the goal into an ordered list of steps, each naming a tool from the
    registry. We ASK the model for a JSON plan, but if it returns prose or
    malformed JSON (common on small models) we drop to FALLBACK_PLAN so the
    pipeline always has a valid plan to execute.
    """
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": ("Break the goal into 2-4 ordered steps. Return ONLY a "
                         "JSON array of objects, each with keys \"action\" "
                         "(string) and \"tool\" (one of web_search, "
                         "rag_retrieve, write). No prose.\n"
                         "Example for a different goal:\n"
                         '[{"action": "Find recent market figures", "tool": '
                         '"web_search"}, {"action": "Draft the summary", '
                         '"tool": "write"}]')},
            {"role": "user", "content": goal},
        ],
        options={"temperature": 0},
    )
    plan = extract_json_array(response.message.content)

    # Validate: must be a non-empty list of dicts with a known tool.
    if (not isinstance(plan, list) or not plan or
            not all(isinstance(s, dict) and s.get("tool") in TOOL_REGISTRY
                    for s in plan)):
        print("  (model plan unusable — using built-in fallback plan)")
        return FALLBACK_PLAN
    return plan


# ─── LAYER 5: TOOL USE / ACTION SELECTION ────────────────────────────────────

def layer_5_tool_selection(step: dict) -> str:
    """
    Validate the tool the plan asked for against the registry. If the plan
    named something we don't have, fall back to a direct written answer.
    (In production this is also where you'd check the tool is *authorised*.)
    """
    tool = step.get("tool", "write")
    return tool if tool in TOOL_REGISTRY else "write"


# ─── LAYER 6: EXECUTION ──────────────────────────────────────────────────────

def layer_6_execution(tool: str, step: dict, gathered: list[str]) -> str:
    """
    Carry out the tool call and return a structured result. web_search and
    rag_retrieve return deterministic simulated data (swap in real APIs in
    production). 'write' is the synthesis step: the LLM turns everything
    gathered so far into the answer — the one place the model truly shines.
    """
    if tool == "web_search":
        return ("Web results — competitor pricing, last quarter:\n"
                "- Competitor Alpha: Enterprise tier $350/seat/month (unchanged)\n"
                "- Competitor Beta: moved to usage-based, ~$280/seat equivalent\n"
                "- Competitor Gamma: cut list price 15%, now ~$297/seat/month")

    if tool == "rag_retrieve":
        return ("Internal KB — 'Q2 Competitive Pricing Analysis':\n"
                "- Our Enterprise tier: $299/seat/month\n"
                "- Win/loss notes: price raised in 22% of lost deals\n"
                "- Beta's usage model appeals to small teams; weak above 50 seats")

    # tool == "write": synthesise the final answer from what we gathered.
    context = "\n\n".join(gathered) if gathered else "(no prior data)"
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": ("Write a concise competitive pricing summary (4-6 "
                         "sentences) using ONLY the information provided. End "
                         "with a one-line recommendation.")},
            {"role": "user",
             "content": f"Task: {step['action']}\n\nInformation:\n{context}"},
        ],
        options={"temperature": 0},
    )
    return response.message.content.strip()


# ─── LAYER 7: FEEDBACK / REFLECTION ──────────────────────────────────────────

def layer_7_reflection(plan: list[dict], results: list[str],
                       final_answer: str) -> tuple[bool, str]:
    """
    Decide whether the goal is met. We use a reproducible rule rather than a
    flaky 'yes/no' model call:
      - every planned step produced a result, AND
      - the final written answer is substantial (not a stub).
    Returns (done, reason). This is the layer that lets the loop terminate
    cleanly — a loop with no exit condition is a runaway agent.
    """
    all_steps_ran = len(results) == len(plan)
    answer_is_real = len(final_answer.strip()) >= 60
    done = all_steps_ran and answer_is_real
    reason = ("all steps ran and a substantial answer was produced"
              if done else "incomplete — would loop again to fill the gap")
    return done, reason


# ─── LAYER 8: OUTPUT ─────────────────────────────────────────────────────────

def layer_8_output(final_answer: str, goal: str) -> str:
    """Format the result for delivery to the user."""
    return (f"\n{'─' * 60}\n"
            f"GOAL:   {goal}\n"
            f"{'─' * 60}\n"
            f"{final_answer}\n"
            f"{'─' * 60}")


# ─── FULL PIPELINE ────────────────────────────────────────────────────────────

def run_pipeline(user_input: str, history: list[dict] | None = None):
    history = history or []
    print(f"\n{'=' * 60}\nINPUT: {user_input}\n{'=' * 60}")

    print("\n[Layer 1] Perception")
    perceived = layer_1_perception(user_input)
    print(f"  {perceived}")

    print("\n[Layer 2] Memory assembly")
    messages = layer_2_memory(perceived, history)
    print(f"  Context window: {len(messages)} messages "
          f"(system + semantic note + {len(history[-4:])} history + query)")

    print("\n[Layer 3] Reasoning")
    understanding = layer_3_reasoning(messages)
    print(f"  Understood goal: {understanding}")

    print("\n[Layer 4] Planning")
    plan = layer_4_planning(perceived["query"])
    for i, step in enumerate(plan, 1):
        print(f"  Step {i}: [{step['tool']}] {step['action']}")

    gathered: list[str] = []
    for i, step in enumerate(plan, 1):
        print(f"\n[Layer 5] Tool selection — step {i}")
        tool = layer_5_tool_selection(step)
        print(f"  Selected: {tool}")

        print(f"[Layer 6] Execution — step {i}")
        result = layer_6_execution(tool, step, gathered)
        gathered.append(result)
        preview = result.replace("\n", " ")[:90]
        print(f"  Result: {preview}...")

    final_answer = gathered[-1] if gathered else "(no result)"

    print("\n[Layer 7] Reflection")
    done, reason = layer_7_reflection(plan, gathered, final_answer)
    print(f"  Goal achieved: {done} — {reason}")

    print("\n[Layer 8] Output")
    print(layer_8_output(final_answer, user_input))

    # Update episodic memory so a follow-up question has continuity.
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final_answer})
    return final_answer, history


def main():
    history: list[dict] = []
    goal = ("Give me a short summary of how our top three competitors have "
            "positioned their pricing in the last quarter, and what it means "
            "for us.")
    run_pipeline(goal, history)


if __name__ == "__main__":
    main()
