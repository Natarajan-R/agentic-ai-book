"""
Chapter 1: The TOOL half of a minimal agent
============================================
A tool is just a script. It reads a structured tool call (JSON) on stdin,
checks it, and decides one of three things:

    DECLINE  - the request is not this agent's job  (is_property_intent false)
    ASK      - it is our job, but required info is missing  (city/type null)
    ACT      - everything is present, so do the real work  (write the file)

It does NOT reason. brain.py (the LLM) decided the parameters; this script only
validates them and acts. Reasoning and acting are deliberately separate.

    echo '{"is_property_intent": true, "city": "Chennai", "property_type": "flat", "price_in_lakhs": 78}' | python agent_tools.py
"""

import json
import re
import sys

REQUIRED = ["city", "property_type"]   # price is optional (falls back below)


def safe_slug(value: str) -> str:
    """Never build a filename from raw model output. Keep letters, numbers,
    spaces, underscores and dashes; drop anything that could redirect the path
    (a city of '../../etc/x' must not write outside this folder)."""
    cleaned = re.sub(r"[^A-Za-z0-9 _-]", "", value).strip().replace(" ", "_")
    return cleaned or "unknown"


def create_property_checklist(city, property_type, price_in_lakhs):
    """The actual tool: deterministic logic, no intelligence."""
    filename = f"{safe_slug(city)}_{safe_slug(property_type)}_checklist.md"

    stamp_duty = "7%" if city.lower() == "chennai" else "5%"
    budget = f"{price_in_lakhs} Lakhs" if price_in_lakhs else "Market Valuation"
    verification_step = (
        "Verify legal vault for multi-story regulations."
        if (price_in_lakhs or 0) > 60 else "Standard registration check."
    )

    content = f"""# SRE Property Agent Checklist
*   **Target Location:** {city.upper()}
*   **Asset Class:** {property_type.capitalize()}
*   **Assigned Stamp Duty Framework:** {stamp_duty}

## Required Action Nodes:
1. [ ] {verification_step}
2. [ ] Validate encumbrance certificate (EC) via local registry portal.
3. [ ] Confirm budget envelope fits below target limit of {budget}.
"""
    with open(filename, "w") as f:
        f.write(content)
    print(f"\n🎉 [TOOL EXECUTED]: generated '{filename}' (budget: {budget})")


def main():
    raw = sys.stdin.read()

    # Is it even JSON? A small model can emit broken output.
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(f"❌ [REJECTED] Not valid JSON. Got: {raw!r}")
        sys.exit(1)

    # DECLINE — not our job. Note we test "not truthy", not "is False": a small
    # model often returns null instead of false, and null must decline too.
    if not data.get("is_property_intent"):
        print("ℹ️  [DECLINED] This request is not about property. "
              "A real agent would hand it to a general chat assistant.")
        return

    # ASK — our job, but the user did not give us everything the tool needs.
    missing = [k for k in REQUIRED if data.get(k) in (None, "")]
    if missing:
        print("🤔 [CANNOT ACT] Property request, but missing: "
              + ", ".join(missing)
              + ".\n   A real agent would ask the user for these, not guess.")
        return

    # Guard the optional price's type before the tool relies on it.
    price = data.get("price_in_lakhs")
    if price is not None and not isinstance(price, (int, float)):
        print(f"❌ [REJECTED] price_in_lakhs must be a number or null, got {price!r}")
        sys.exit(1)

    # ACT.
    create_property_checklist(data["city"], data["property_type"], price)


if __name__ == "__main__":
    main()
