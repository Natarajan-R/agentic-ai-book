"""
Chapter 1: Script vs AI Agent
==============================
The same task solved two ways.
Script: hardcoded logic, always same path.
Agent:  reasons its way to an answer, adapts to context.

Run:  python ch01_script_vs_agent.py
Need: ollama pull qwen2.5:7b
"""

import ollama

MODEL = "qwen2.5:7b"


# ─── THE SCRIPT ──────────────────────────────────────────────────────────────
# Fixed logic. Cannot adapt. Handles only the case its author foresaw.

def script_summarise(text: str) -> str:
    """Always returns first 20 words. No judgment. No adaptation."""
    words = text.split()
    return " ".join(words[:20]) + " ..."


def script_classify_sentiment(text: str) -> str:
    """Keyword-based. Breaks on anything outside the list."""
    positive = ["good", "great", "excellent", "love", "happy"]
    negative = ["bad",  "poor",  "terrible",  "hate", "sad"]
    text_lower = text.lower()
    pos = sum(1 for w in positive if w in text_lower)
    neg = sum(1 for w in negative if w in text_lower)
    if pos > neg:
        return "POSITIVE"
    if neg > pos:
        return "NEGATIVE"
    return "NEUTRAL"


# ─── THE AGENT ───────────────────────────────────────────────────────────────
# Reasons over the goal. Adapts to style, context, and nuance.

def agent_summarise(text: str, style: str) -> str:
    """Understands the goal. Adapts output to the requested style."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You are a professional summarisation assistant."},
            {"role": "user",
             "content": f"Summarise the following in a {style} style:\n\n{text}"},
        ],
        options={"temperature": 0},
    )
    return response.message.content


def agent_classify_sentiment(text: str) -> str:
    """Understands nuance, sarcasm, context — not just keywords."""
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": (
                 "Classify the sentiment of the text. "
                 "Return exactly one word: POSITIVE, NEGATIVE, or NEUTRAL. "
                 "Consider nuance and sarcasm."
             )},
            {"role": "user", "content": text},
        ],
        options={"temperature": 0},
    )
    return response.message.content.strip()


# ─── DEMO ────────────────────────────────────────────────────────────────────

def main():
    sample = (
        "Artificial intelligence is reshaping enterprise software. "
        "Agents can now plan, reason, and execute multi-step tasks "
        "autonomously using large language models as their reasoning core. "
        "The implications for knowledge work are profound and far-reaching."
    )

    print("=" * 60)
    print("COMPARISON 1: Summarisation")
    print("=" * 60)
    print(f"\nInput: {sample}\n")

    print("--- Script output (always first 20 words) ---")
    print(script_summarise(sample))

    print("\n--- Agent output (adapts to style) ---")
    for style in ["formal one sentence", "casual tweet", "bullet points"]:
        print(f"\nStyle: {style}")
        print(agent_summarise(sample, style))

    print("\n" + "=" * 60)
    print("COMPARISON 2: Sentiment — handling sarcasm")
    print("=" * 60)

    tricky = "Oh great, another AI tool that will solve all our problems."
    print(f"\nInput: {tricky}\n")
    print(f"Script result:  {script_classify_sentiment(tricky)}")
    print(f"Agent result:   {agent_classify_sentiment(tricky)}")

    print("\n" + "=" * 60)
    print("KEY LESSON")
    print("=" * 60)
    print("Script: executes what its author foresaw.")
    print("Agent:  reasons through what it was not explicitly told.")


if __name__ == "__main__":
    main()
