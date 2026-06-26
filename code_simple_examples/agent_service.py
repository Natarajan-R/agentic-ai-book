"""
Chapter 1 (going further): the same minimal agent as ONE Python service
=======================================================================
brain.py + agent_tools.py show the agent as two scripts joined by a pipe,
which makes the parts obvious. This file shows the *same* agent as a single
program — closer to how a real backend service is structured: one process that
reasons, routes, and acts.

It is deliberately written in this book's style and tuned for a local 7B model:
  - system prompt is a fixed constant (the "software contract")
  - temperature 0 for deterministic extraction
  - fence-tolerant JSON parsing (small models sometimes add ```json)
  - the model decides act / ask / decline; our code stays the deterministic anchor

Run:  python agent_service.py
Need: ollama pull qwen2.5:7b

Under the hood, ollama.chat() below is just an HTTP POST to
http://localhost:11434/api/chat with a {system, user} messages array — the same
call a framework like LangChain makes for you. Nothing magic.
"""

import json
import re
import ollama

MODEL = "qwen2.5:7b"

# ─── FIXED CONTRACT ──────────────────────────────────────────────────────────
# For a domain agent the system prompt and schema are fixed in code, not user
# input. That fixed contract is exactly what lets the code below trust the shape
# of the JSON it gets back.
SYSTEM_PROMPT = """You are a parameter-extraction layer for a real-estate assistant.
Read the user's message and return a single JSON object with exactly these keys:

{"is_property_intent": true or false,
 "city": string or null,
 "property_type": string or null,
 "price_in_lakhs": number or null}

Rules:
- is_property_intent is true ONLY if the user wants to buy, sell, list, value, or
  make a checklist for a property. For anything else it is false. Always a
  boolean, never null.
- city, property_type and price_in_lakhs: fill only from the user's message; if a
  value is not stated, set it to null. Never invent or guess.
- Output only the JSON. No markdown, no backticks, no explanation.

Examples:
User: What is the capital of France?
JSON: {"is_property_intent": false, "city": null, "property_type": null, "price_in_lakhs": null}
User: I need a checklist for my place in Chennai
JSON: {"is_property_intent": true, "city": "Chennai", "property_type": null, "price_in_lakhs": null}
User: list a sea-facing villa in Kochi for 150 lakhs
JSON: {"is_property_intent": true, "city": "Kochi", "property_type": "villa", "price_in_lakhs": 150}
"""

REQUIRED = ["city", "property_type"]


def _extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else text.strip()


def reason(user_input: str) -> dict:
    """The BRAIN: natural language in, a structured decision out."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        options={"temperature": 0},
    )
    raw = response["message"]["content"]
    try:
        return json.loads(_extract_json(raw))
    except json.JSONDecodeError:
        # Even with temperature 0, a small model can occasionally return junk.
        # Fail closed: an unparseable decision is not an excuse to act.
        return {"is_property_intent": False, "_parse_error": raw}


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", value).strip().replace(" ", "_")
    return cleaned or "unknown"


def act(city: str, property_type: str, price_in_lakhs) -> str:
    """The TOOL: deterministic, no intelligence."""
    filename = f"{_safe_slug(city)}_{_safe_slug(property_type)}_checklist.md"
    budget = f"{price_in_lakhs} Lakhs" if price_in_lakhs else "Market Valuation"
    with open(filename, "w") as f:
        f.write(f"# Property Checklist — {city.upper()}\n"
                f"* Asset class: {property_type}\n"
                f"* Budget cap: {budget}\n")
    return f"🎉 [ACTED] generated '{filename}' (budget: {budget})"


def handle(user_input: str) -> str:
    """The ROUTER: reason, then act / ask / decline. Our code owns this logic."""
    decision = reason(user_input)

    if not decision.get("is_property_intent"):     # not truthy => decline
        return "ℹ️  [DECLINED] Not a property request — route to general chat."

    missing = [k for k in REQUIRED if decision.get(k) in (None, "")]
    if missing:
        return ("🤔 [ASK] Property request, but missing " + ", ".join(missing)
                + " — ask the user, do not guess.")

    return act(decision["city"], decision["property_type"],
               decision.get("price_in_lakhs"))


if __name__ == "__main__":
    for request in [
        "Can you write a poem about rain?",            # off-topic  -> DECLINE
        "I need to list my flat in Chennai tomorrow.",  # partial    -> ... ACT/ASK
        "Build a layout for an 85 Lakh villa in Bangalore.",  # full -> ACT
    ]:
        print(f"\nUSER: {request}\n  -> {handle(request)}")
