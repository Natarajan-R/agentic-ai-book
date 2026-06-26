"""
Chapter 1: The BRAIN half of a minimal agent
=============================================
Reads a plain-English request on stdin, asks the LLM to turn it into a
structured tool call (JSON), and prints that JSON to stdout.

This is the *reasoning* half. The *acting* half is agent_tools.py.
Pipe them together and you have the smallest possible agent:

    echo "list a luxury flat in Bangalore for 92 lakhs" | python brain.py | python agent_tools.py

The model also decides whether the request is its job at all
(is_property_intent), so the agent can act, ask, or politely decline.

Need: ollama pull qwen2.5:7b
"""

import sys
import json
import ollama

MODEL = "qwen2.5:7b"

# ─── THE SYSTEM PROMPT (the agent's fixed "software contract") ────────────────
# Fixed developer instructions. Identical on every run. Kept completely separate
# from whatever the user types. The schema and rules below are the contract our
# Python code relies on — so we pin them here, in code, not in user input.
SYSTEM_PROMPT = """You are a parameter-extraction layer for a real-estate assistant.
Read the user's message and return a single JSON object with exactly these keys:

{"is_property_intent": true or false,
 "city": string or null,
 "property_type": string or null,
 "price_in_lakhs": number or null}

Rules:
- is_property_intent is true ONLY if the user wants to buy, sell, list, value, or
  make a checklist for a property. For anything else (weather, trivia, jokes,
  general questions) it is false. This field is ALWAYS a boolean, never null.
- city, property_type and price_in_lakhs: fill a field ONLY from the user's
  message. If a value is not stated, set it to null. Never invent or guess.
- Output only the JSON. No markdown, no backticks, no explanation.

Examples:
User: What is the capital of France?
JSON: {"is_property_intent": false, "city": null, "property_type": null, "price_in_lakhs": null}
User: I need a checklist for my place in Chennai
JSON: {"is_property_intent": true, "city": "Chennai", "property_type": null, "price_in_lakhs": null}
User: list a sea-facing villa in Kochi for 150 lakhs
JSON: {"is_property_intent": true, "city": "Kochi", "property_type": "villa", "price_in_lakhs": 150}
"""


def extract_json(text: str) -> str:
    """Small models sometimes wrap JSON in prose or ``` fences. Defensively
    take the substring from the first { to the last }."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else text.strip()


def main():
    # ─── THE USER MESSAGE ────────────────────────────────────────────────────
    # The only part that changes from one run to the next: the request itself.
    user_message = sys.stdin.read().strip()

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},   # fixed contract
            {"role": "user", "content": user_message},      # variable request
        ],
        options={"temperature": 0},                         # deterministic
    )
    print(extract_json(response["message"]["content"]))


if __name__ == "__main__":
    main()
