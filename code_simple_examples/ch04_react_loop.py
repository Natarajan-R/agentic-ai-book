"""
Chapter 4: The ReAct Loop
==========================
Thought -> Action -> Observation, repeating until one of the loop's
exit conditions fires.

Run:  python ch04_react_loop.py
Need: ollama pull qwen2.5:7b

This implements the four exit conditions from the chapter:
  1. Goal achieved        - the agent produces a final answer
  2. Max iterations       - hard safety cap
  3. Human intervention   - an irreversible tool needs approval (stub)
  4. Unrecoverable error  - an exception is caught

...plus the safety practice from section 4.8: OSCILLATION DETECTION.
If the agent repeats the same action it took in a recent iteration, it
is stuck in a loop and we exit instead of spinning forever.

Why this matters (and why the naive version fails)
---------------------------------------------------
A small local model is unreliable at calling a special "task_complete"
tool. After it gathers information, it usually just WRITES the answer as
plain text instead of emitting a structured call. A loop that only looks
for a tool call will therefore never notice the task is done and will
spin until it hits the iteration cap. So here we treat "the model wrote a
substantive answer and asked for no tool" as the natural goal-achieved
signal — which is exactly how a 7B model behaves in practice.
"""

import json
import ollama

MODEL = "qwen2.5:7b"

# Tools the agent can call. Note there is NO "task_complete" tool: the
# agent signals completion simply by writing its final answer (see above).
TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string",
                                                "description": "Search query"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": ("Evaluate a single arithmetic expression precisely, "
                        "e.g. '3.8e9 * (1.42 ** 3)'. Use this for ANY math "
                        "instead of computing in your head."),
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string",
                                                     "description": "Python math expression"}},
                       "required": ["expression"]}}},
]

# Tools that may NOT run without a human (exit condition 3).
IRREVERSIBLE = {"send_email", "delete_record", "post_public"}


def execute_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return ("Results for '" + args.get("query", "") + "':\n"
                "- AI agent market reached $3.8B in 2024 (Gartner)\n"
                "- Enterprise adoption: 61% of Fortune 500 piloting agents\n"
                "- Projected CAGR: 42% through 2028\n"
                "- Top use cases: customer service, research, code generation")
    if name == "calculate":
        try:
            value = eval(args.get("expression", "0"), {"__builtins__": {}})  # noqa: S307
            return str(value)
        except Exception as e:                      # tool fails gracefully
            return f"Calculation error: {e}"
    return f"Unknown tool: {name}"


def action_signature(msg) -> str:
    """
    A compact fingerprint of what the agent did this turn, used for
    oscillation detection. Two iterations with the same signature mean the
    agent is repeating itself.
    """
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        return f"{tc.function.name}:{json.dumps(tc.function.arguments, sort_keys=True, default=str)}"
    return f"text:{(msg.content or '').strip()[:120]}"


def react_loop(goal: str, max_iterations: int = 8) -> str:
    messages = [
        {"role": "system",
         "content": ("You are a research agent that works in a Thought -> "
                     "Action -> Observation loop. Think briefly, then either "
                     "call a tool to gather what you need, or — once you have "
                     "enough — reply with your FINAL ANSWER as plain text and "
                     "no tool call. Always use the calculate tool for math.")},
        {"role": "user", "content": goal},
    ]

    print(f"\n{'=' * 55}\nGOAL: {goal}\n{'=' * 55}")

    recent_signatures: list[str] = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n-- Iteration {iteration}/{max_iterations} --")

        # Re-ground on the original goal each turn (section 4.8: prevents
        # goal drift over long runs). The reminder is sent only for this
        # call; it is not added to the permanent history.
        call_messages = messages + [
            {"role": "system", "content": f"Reminder - the goal is: {goal}"}]

        # Exit condition 4: catch unrecoverable model/transport errors.
        try:
            response = ollama.chat(model=MODEL, messages=call_messages,
                                   tools=TOOLS, options={"temperature": 0})
        except Exception as e:
            print(f"[EXIT 4] Unrecoverable error: {e}")
            return f"Task failed due to error: {e}"

        msg = response.message

        # Oscillation detection (section 4.8): same action as a recent turn?
        sig = action_signature(msg)
        if sig in recent_signatures:
            print("[EXIT - oscillation] Repeated a recent action; agent is "
                  "stuck. Escalating to a human.")
            return "Task stopped: the agent began repeating itself."
        recent_signatures.append(sig)
        recent_signatures = recent_signatures[-2:]   # keep last two states

        # THOUGHT phase
        if msg.content:
            print(f"[Thought] {msg.content[:200]}")

        # ACTION phase
        if not msg.tool_calls:
            # Exit condition 1: no tool requested + a real answer => done.
            if msg.content and len(msg.content.strip()) >= 40:
                print(f"\n[EXIT 1] Goal achieved at iteration {iteration}")
                print(f"\n{'-' * 55}\nFINAL ANSWER:\n{msg.content.strip()}\n{'-' * 55}")
                return msg.content.strip()
            # Empty/too-short reply: ask it to continue once.
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user",
                             "content": "Continue. Use a tool, or give your final answer."})
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            # Exit condition 3: irreversible action needs a human.
            if name in IRREVERSIBLE:
                print(f"[EXIT 3] Human confirmation required for: {name}")
                return f"Paused - awaiting human approval for: {name}"

            print(f"[Action] {name}({args})")
            observation = execute_tool(name, args)
            print(f"[Observation] {observation[:150]}")

            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc]})
            messages.append({"role": "tool", "name": name, "content": observation})

    # Exit condition 2: hard iteration cap.
    print(f"\n[EXIT 2] Max iterations ({max_iterations}) reached")
    return "Task incomplete - maximum iterations reached. Human review recommended."


if __name__ == "__main__":
    react_loop(
        "Find the current AI agent market size, then calculate its projected "
        "size in 3 years assuming it keeps growing at the reported annual rate. "
        "Give the market size, the growth rate, and the projected figure."
    )
