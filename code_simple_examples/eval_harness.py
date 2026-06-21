"""
A minimal agent evaluation harness (Chapter 12, section 12.11).

Demonstrates the four parts of agent evaluation on a small, runnable example:
  1. Success criteria      — defined per task in the golden set
  2. A golden set          — representative tasks with known-good answers
  3. Scoring               — programmatic checks + LLM-as-a-judge
  4. Regression evaluation — compare this run against a saved baseline

Task under test: a support-ticket assistant that (a) classifies a ticket's
intent and (b) writes a one-line acknowledgement to the customer. Intent is
checked programmatically (exact match); the reply is scored by an LLM judge.

Run:  python eval_harness.py                 # evaluate and compare to baseline
      python eval_harness.py --save-baseline # record the current scores
Need: ollama pull qwen2.5:7b
"""

import json
import re
import sys
from pathlib import Path
import ollama

MODEL = "qwen2.5:7b"
BASELINE = Path("eval_baseline.json")
INTENTS = ["billing", "technical", "sales", "other"]


# ─── 1 + 2. The golden set: representative tasks with success criteria ────────

GOLDEN_SET = [
    {"id": "t1", "ticket": "I was charged twice for my subscription this month.",
     "expected_intent": "billing"},
    {"id": "t2", "ticket": "The app crashes every time I open the reports page.",
     "expected_intent": "technical"},
    {"id": "t3", "ticket": "Do you offer a discount for a 50-seat annual plan?",
     "expected_intent": "sales"},
    {"id": "t4", "ticket": "Just wanted to say your support team is wonderful!",
     "expected_intent": "other"},
    {"id": "t5", "ticket": "My invoice shows the wrong VAT number.",
     "expected_intent": "billing"},
    {"id": "t6", "ticket": "How do I export my report data to a CSV file?",
     "expected_intent": "technical"},
]


# ─── the agent under test ────────────────────────────────────────────────────

def classify_intent(ticket: str) -> str:
    resp = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": (
            "Classify the support ticket into exactly one of: "
            "billing, technical, sales, other. Reply with the single word only.")},
        {"role": "user", "content": ticket},
    ], options={"temperature": 0})
    word = resp.message.content.strip().lower()
    for intent in INTENTS:            # normalise to a known label
        if intent in word:
            return intent
    return "other"


def write_reply(ticket: str) -> str:
    resp = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": (
            "Write a single polite, relevant one-sentence acknowledgement to the "
            "customer. Do not try to solve the issue; just acknowledge it.")},
        {"role": "user", "content": ticket},
    ], options={"temperature": 0})
    return resp.message.content.strip()


# ─── 3. Scoring: programmatic + LLM-as-a-judge ───────────────────────────────

def programmatic_score(predicted: str, expected: str) -> float:
    """Exact-match check — fast, free, perfectly repeatable."""
    return 1.0 if predicted == expected else 0.0


def judge_score(ticket: str, reply: str) -> float:
    """
    LLM-as-a-judge: rate the reply 1-5 against an explicit rubric and normalise
    to 0-1. Run at temperature 0; in production, calibrate the judge against
    human scores on a sample before trusting it (see section 12.11).
    """
    resp = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": (
            "You are an evaluator. Rate the customer reply from 1 to 5 on whether "
            "it is polite AND clearly relevant to the ticket. 5 = polite and "
            "directly relevant; 1 = rude or irrelevant. Reply with the digit only.")},
        {"role": "user", "content": f"Ticket: {ticket}\nReply: {reply}"},
    ], options={"temperature": 0})
    m = re.search(r"[1-5]", resp.message.content)
    return (int(m.group()) / 5.0) if m else 0.0


# ─── run + aggregate ─────────────────────────────────────────────────────────

def run_eval() -> dict:
    results = []
    for case in GOLDEN_SET:
        intent = classify_intent(case["ticket"])
        reply = write_reply(case["ticket"])
        p = programmatic_score(intent, case["expected_intent"])
        j = judge_score(case["ticket"], reply)
        composite = round((p + j) / 2, 3)
        results.append({"id": case["id"], "intent_ok": p, "reply_quality": j,
                        "score": composite})
        status = "ok" if p else f"WRONG (got {intent}, want {case['expected_intent']})"
        print(f"  {case['id']}: intent {status} | reply quality {j:.1f} | score {composite}")

    n = len(results)
    return {
        "intent_accuracy":   round(sum(r["intent_ok"] for r in results) / n, 3),
        "mean_reply_quality": round(sum(r["reply_quality"] for r in results) / n, 3),
        "mean_score":        round(sum(r["score"] for r in results) / n, 3),
        "per_task":          {r["id"]: r["score"] for r in results},
    }


# ─── 4. Regression: compare against a saved baseline ─────────────────────────

def compare_to_baseline(current: dict) -> None:
    if not BASELINE.exists():
        print("\nNo baseline recorded yet. Run with --save-baseline to create one,")
        print("then re-run after any change (prompt, tools, or model) to catch regressions.")
        return
    base = json.loads(BASELINE.read_text())
    print("\n-- Regression vs baseline --")
    for metric in ("intent_accuracy", "mean_reply_quality", "mean_score"):
        delta = round(current[metric] - base.get(metric, 0), 3)
        flag = "REGRESSION" if delta < -0.001 else ("improved" if delta > 0.001 else "same")
        print(f"  {metric}: {base.get(metric)} -> {current[metric]}  ({delta:+}) {flag}")
    regressed = [tid for tid, s in current["per_task"].items()
                 if s < base.get("per_task", {}).get(tid, 0) - 0.001]
    if regressed:
        print(f"  tasks that regressed: {regressed}")


if __name__ == "__main__":
    print(f"Evaluating the agent over {len(GOLDEN_SET)} golden tasks ...\n")
    current = run_eval()
    print(f"\nintent accuracy: {current['intent_accuracy']:.0%} | "
          f"mean reply quality: {current['mean_reply_quality']:.0%} | "
          f"mean score: {current['mean_score']:.0%}")
    if "--save-baseline" in sys.argv:
        BASELINE.write_text(json.dumps(current, indent=2))
        print(f"\nBaseline saved to {BASELINE}.")
    else:
        compare_to_baseline(current)
