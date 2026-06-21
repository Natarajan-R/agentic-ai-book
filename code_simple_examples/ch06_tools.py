"""
Chapter 6: Tool Registry and Tool Use
=======================================
A small registry of well-designed tools, each following the five
principles from section 6.3:

  1. Single responsibility - each tool does exactly one thing.
  2. Unambiguous description - the model can tell tools apart and knows
     exactly what input each expects.
  3. Predictable output - every call returns a structured ToolResult.
  4. Graceful failure - errors come back as data, not crashes.
  5. Observable execution - every call is logged with timing.

Run:  python ch06_tools.py
Need: ollama pull qwen2.5:7b

Why the tool SCHEMAS matter (the bug this fixes)
------------------------------------------------
A common mistake is to give every tool one generic "input" string. The
model then has to guess the format, and it guesses wrong: it sends
multi-line code to a calculator that only accepts a single expression,
or it has nowhere to put a filename for a file writer. Here each tool
declares its OWN parameters with a precise description, so the model
fills them in correctly. This is section 6.3's "unambiguous
description" principle doing real work.
"""

import json
import math
import time
from dataclasses import dataclass
import ollama

MODEL = "qwen2.5:7b"


# ─── TOOL REGISTRY (rate limits + human-readable catalogue) ───────────────────

TOOL_REGISTRY = {
    "web_search":   {"rate_limit": 10},
    "rag_retrieve": {"rate_limit": 20},
    "code_execute": {"rate_limit": 5},
    "file_write":   {"rate_limit": 3},
}

# Each tool declares its OWN parameters with a precise description — this is
# what lets a small model call them correctly (section 6.3).
OLLAMA_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": ("Search the public web for current information "
                        "(news, market data, competitor info)."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to search for"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "rag_retrieve",
        "description": ("Retrieve from the internal enterprise knowledge base "
                        "(policies, past reports, product docs)."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look up internally"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "code_execute",
        "description": ("Evaluate ONE Python arithmetic expression precisely and "
                        "return the number. Keep the SAME units as the problem "
                        "(if figures are in billions, compute in billions: "
                        "'3.8 * (1.42 ** 3)'). Send a single expression only — "
                        "no statements or multiple lines. Use this for ALL math."),
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string",
                           "description": "A single Python math expression"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "file_write",
        "description": "Write text content to a named deliverable file.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Target file name"},
            "content":  {"type": "string", "description": "Text to write"}},
            "required": ["filename", "content"]}}},
]


# ─── TOOL EXECUTION ──────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    success:   bool
    output:    str
    tool_name: str
    duration:  float


_call_counts: dict[str, int] = {}

# A safe namespace for code_execute: maths only, no builtins, no file/network.
_SAFE_ENV = {"__builtins__": {}, "math": math,
             "round": round, "abs": abs, "min": min, "max": max,
             "sum": sum, "pow": pow, "len": len, "str": str, "format": format}


def execute_tool(name: str, args: dict) -> ToolResult:
    """Execute a tool with rate limiting, timing, and graceful failure."""
    t0 = time.time()

    count = _call_counts.get(name, 0)
    limit = TOOL_REGISTRY.get(name, {}).get("rate_limit", 99)
    if count >= limit:
        return ToolResult(False, f"Rate limit reached for {name} ({limit}/session)",
                          name, 0.0)
    _call_counts[name] = count + 1

    try:
        if name == "web_search":
            output = (f"Web results for '{args.get('query', '')}':\n"
                      "- AI agent market: $3.8B in 2024, CAGR 42% to 2028 (Gartner)\n"
                      "- 61% of Fortune 500 piloting agents")

        elif name == "rag_retrieve":
            output = (f"Internal KB for '{args.get('query', '')}':\n"
                      "- Q2 Competitive Analysis (June 2024)\n"
                      "- Customer ROI survey: avg 3.2x return")

        elif name == "code_execute":
            expr = args.get("expression", "")
            value = eval(expr, _SAFE_ENV)              # noqa: S307 - sandboxed
            # Return a clean, copyable value (avoid float noise like
            # 10.880494399999996 that the model then mis-transcribes).
            output = str(round(value, 2) if isinstance(value, float) else value)

        elif name == "file_write":
            filename = args.get("filename", "output.txt")
            content = args.get("content", "")
            with open(filename, "w") as f:
                f.write(content)
            output = f"Wrote {len(content)} chars to {filename}"

        else:
            return ToolResult(False, f"Unknown tool: {name}", name, 0.0)

        return ToolResult(True, output, name, time.time() - t0)

    except Exception as e:                              # graceful failure
        return ToolResult(False, f"{type(e).__name__}: {e}", name, time.time() - t0)


# ─── TOOL-USING AGENT ────────────────────────────────────────────────────────

def run_tool_agent(task: str, max_iterations: int = 8):
    print(f"\n{'=' * 55}\nTASK: {task}\n{'=' * 55}")

    messages = [
        {"role": "system",
         "content": ("You are an enterprise research agent with tools. Use "
                     "code_execute (a single Python expression, in the same "
                     "units as the problem) for EVERY calculation — never do "
                     "math in your head. Use file_write(filename, content) to "
                     "save deliverables. When the task is complete, reply with "
                     "a short final summary that states the computed figure, "
                     "and make no tool call.")},
        {"role": "user", "content": task},
    ]

    for iteration in range(1, max_iterations + 1):
        response = ollama.chat(model=MODEL, messages=messages,
                               tools=OLLAMA_TOOLS, options={"temperature": 0})
        msg = response.message

        if not msg.tool_calls:
            # No tool requested: treat a real answer as completion.
            if msg.content and len(msg.content.strip()) >= 30:
                print(f"\n[Final answer]\n{msg.content.strip()}")
                return msg.content.strip()
            messages.append({"role": "assistant", "content": msg.content or ""})
            messages.append({"role": "user",
                             "content": "Continue, or give your final summary."})
            continue

        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            print(f"\n[Tool] {name}({args})")
            result = execute_tool(name, args)

            # Observable execution: log outcome + timing (section 6.3).
            if result.success:
                print(f"  [ok] {result.output[:100]} ({result.duration * 1000:.0f}ms)")
                obs = result.output
            else:
                print(f"  [fail] {result.output}")
                obs = f"Tool failed: {result.output}. Try a different approach."

            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc]})
            messages.append({"role": "tool", "name": name, "content": str(obs)})

    print("\n[Max iterations reached]")
    return "Incomplete"


if __name__ == "__main__":
    run_tool_agent(
        "The AI agent market is $3.8B and grows 42% per year. Calculate its "
        "projected size in 3 years, then save a short qualitative note (no "
        "numbers) about why this growth trend matters to 'brief.txt'. Finally, "
        "report the projected figure."
    )
    print(f"\nTool usage counts: {_call_counts}")
