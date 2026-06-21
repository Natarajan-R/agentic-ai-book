"""
Chapter 14: Self-Refining Agent — LangGraph
=============================================
Agent stores execution history and uses it
to improve planning quality across task runs.
Experience-guided planning without weight updates.
"""

import json, os, re
from pathlib import Path
from datetime import datetime, timezone
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

MODEL = "qwen2.5:7b"
HISTORY_FILE = Path("agent_history.json")

def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []

def save_history(history: list[dict]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def retrieve_similar_tasks(goal: str, history: list[dict],
                           n: int = 3) -> list[dict]:
    """
    Find the most similar past task runs.
    Simple keyword overlap — replace with embedding similarity in production.
    """
    goal_words = set(goal.lower().split())
    scored = []
    for entry in history:
        past_words = set(entry.get("goal", "").lower().split())
        overlap = len(goal_words & past_words) / max(len(goal_words), 1)
        scored.append((overlap, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:n] if _ > 0.1]


def run_self_refining_agent(goal: str):
    llm      = ChatOllama(model=MODEL, temperature=0)
    history  = load_history()
    similar  = retrieve_similar_tasks(goal, history)

    print(f"\n{'═' * 55}")
    print(f"GOAL: {goal}")
    print(f"Similar past tasks found: {len(similar)}")

    # Build experience context
    experience = ""
    if similar:
        experience = "\n\nLearning from similar past tasks:\n"
        for s in similar:
            quality = s.get("quality_score", "unknown")
            notes   = s.get("lessons_learned", "none")
            experience += (
                f"- Task: {s['goal'][:60]}\n"
                f"  Quality: {quality} | Lessons: {notes}\n"
            )
        print(experience)

    # Generate experience-informed plan
    plan_resp = llm.invoke([
        SystemMessage(content=(
            "You are a self-refining planning agent. "
            "Use lessons from past tasks to generate a better plan."
            + experience
        )),
        HumanMessage(content=(
            f"Goal: {goal}\n\n"
            "Generate a brief sequential plan (3-5 steps) that avoids "
            "the failure patterns seen in similar past tasks."
        )),
    ])
    plan = plan_resp.content
    print(f"\n[Experience-informed plan]\n{plan}")

    # Execute (simplified)
    exec_resp = llm.invoke([
        SystemMessage(content="Execute the plan and produce the result."),
        HumanMessage(content=f"Plan:\n{plan}\n\nGoal: {goal}"),
    ])
    result = exec_resp.content
    print(f"\n[Result]\n{result[:300]}")

    # Self-evaluate quality
    quality_resp = llm.invoke([
        SystemMessage(content="Score the result quality from 1-10. Return only the number."),
        HumanMessage(content=f"Goal: {goal}\nResult: {result[:300]}"),
    ])
    try:
        quality_score = int(re.search(r"\d+", quality_resp.content).group())
        quality_score = max(1, min(10, quality_score))
    except Exception:
        quality_score = 5

    # Extract lessons
    lessons_resp = llm.invoke([
        SystemMessage(content=(
            "In one sentence, what is the key lesson from this task run "
            "that would help plan similar tasks better in the future?"
        )),
        HumanMessage(content=f"Goal: {goal}\nResult: {result[:300]}"),
    ])
    lessons = lessons_resp.content.strip()

    # Persist to history
    entry = {
        "goal":            goal,
        "plan":            plan[:300],
        "result_summary":  result[:200],
        "quality_score":   quality_score,
        "lessons_learned": lessons,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
    }
    history.append(entry)
    save_history(history)

    print(f"\n[Self-evaluation] Quality: {quality_score}/10")
    print(f"[Lesson learned] {lessons}")
    print(f"[History] {len(history)} task runs stored in {HISTORY_FILE}")

    return result


if __name__ == "__main__":
    # Run twice to demonstrate learning across runs
    run_self_refining_agent(
        "Compare pricing strategies of enterprise AI platforms and "
        "identify the one with the best value for a 100-seat team."
    )
    print("\n\n── Second run (learns from first) ──")
    run_self_refining_agent(
        "Analyse pricing models of cloud AI agent providers and "
        "recommend the most cost-effective option for enterprise use."
    )