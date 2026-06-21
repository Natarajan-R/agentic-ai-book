"""
Chapter 11: Tree of Thought — LangGraph
=========================================
Branch generation → scoring → pruning → synthesis.
Explores multiple reasoning paths before committing.
"""

import json
import re
import random
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

MODEL = "qwen2.5:7b"


def extract_json_array(text: str):
    """Pull the first JSON array out of a model reply; None if not parseable."""
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

def run_tree_of_thought(goal: str,
                        branching_factor: int = 3,
                        exploration_depth: int = 2,
                        prune_threshold: float = 0.4):
    """
    Tree of Thought planning.

    Parameters:
      branching_factor:  candidate approaches to generate
      exploration_depth: steps to explore each branch
      prune_threshold:   score below which a branch is abandoned
    """
    llm = ChatOllama(model=MODEL, temperature=0.3)   # slight temp for diversity

    print(f"\n{'═' * 55}")
    print(f"GOAL: {goal}")
    print(f"Branching={branching_factor}, Depth={exploration_depth}, "
          f"Prune threshold={prune_threshold}")
    print(f"{'═' * 55}")

    # ── Step 1: Generate candidate approaches ─────────────────────────────────
    print(f"\n[Phase 1] Generating {branching_factor} candidate approaches ...")
    response = llm.invoke([
        SystemMessage(content=(
            f"Generate exactly {branching_factor} meaningfully different approaches "
            "to solve the goal. Each approach should explore different information "
            "sources or analytical methods. "
            f"Return a JSON array of {branching_factor} objects: "
            '[{"branch_id": 1, "approach": "description", "first_step": "action"}]'
        )),
        HumanMessage(content=f"Goal: {goal}"),
    ])
    branches = extract_json_array(response.content)
    if (not isinstance(branches, list) or not branches or
            not all(isinstance(b, dict) and "approach" in b for b in branches)):
        branches = [
            {"branch_id": i+1,
             "approach": f"Approach {i+1}",
             "first_step": f"Step {i+1} of goal"}
            for i in range(branching_factor)
        ]
    # Ensure every branch has a branch_id for downstream printing.
    for i, b in enumerate(branches, 1):
        b.setdefault("branch_id", i)

    for b in branches:
        print(f"  Branch {b['branch_id']}: {b['approach'][:60]}")

    # ── Step 2: Explore each branch ───────────────────────────────────────────
    print(f"\n[Phase 2] Exploring branches to depth {exploration_depth} ...")
    branch_results = []

    for branch in branches:
        print(f"\n  Branch {branch['branch_id']}: {branch['approach'][:40]}")
        evidence = []

        for depth in range(exploration_depth):
            step_resp = llm.invoke([
                SystemMessage(content=(
                    "You are exploring a research branch. "
                    "Execute the next step and report findings. Be concise."
                )),
                HumanMessage(content=(
                    f"Goal: {goal}\n"
                    f"Approach: {branch['approach']}\n"
                    f"Depth {depth+1}/{exploration_depth}\n"
                    f"Previous findings: {' | '.join(evidence)}\n"
                    "What do you find at this depth?"
                )),
            ])
            finding = step_resp.content[:200]
            evidence.append(finding)
            print(f"    Depth {depth+1}: {finding[:80]}")

        # ── Step 3: Score each branch ─────────────────────────────────────────
        score_resp = llm.invoke([
            SystemMessage(content=(
                "Score the quality and relevance of these research findings "
                "for the given goal. Return ONLY a decimal between 0.0 and 1.0."
            )),
            HumanMessage(content=(
                f"Goal: {goal}\n"
                f"Findings: {' | '.join(evidence)}"
            )),
        ])
        try:
            score = float(re.search(r"0?\.\d+|1\.0|0|1", score_resp.content).group())
        except Exception:
            score = 0.5

        branch_results.append({
            "branch":   branch,
            "evidence": evidence,
            "score":    score,
        })
        print(f"    Score: {score:.2f} {'✓ KEEP' if score >= prune_threshold else '✗ PRUNE'}")

    # ── Step 4: Prune low-scoring branches ────────────────────────────────────
    kept   = [b for b in branch_results if b["score"] >= prune_threshold]
    pruned = [b for b in branch_results if b["score"] <  prune_threshold]
    print(f"\n[Phase 3] Pruned {len(pruned)} branches. Keeping {len(kept)}.")

    if not kept:
        kept = sorted(branch_results, key=lambda x: x["score"], reverse=True)[:1]
        print("  (All below threshold — keeping best branch anyway)")

    # ── Step 5: Synthesise from best branches ─────────────────────────────────
    print(f"\n[Phase 4] Synthesising from top {len(kept)} branches ...")
    evidence_summary = "\n".join(
        f"Branch {b['branch']['branch_id']} (score {b['score']:.2f}): "
        + " | ".join(b["evidence"])
        for b in sorted(kept, key=lambda x: x["score"], reverse=True)
    )

    synth = llm.invoke([
        SystemMessage(content="Synthesise the research findings into a final answer."),
        HumanMessage(content=(
            f"Goal: {goal}\n\n"
            f"Research findings from best branches:\n{evidence_summary}\n\n"
            "Produce a comprehensive final answer."
        )),
    ])

    print(f"\n[Final Answer]\n{synth.content}")
    return synth.content


if __name__ == "__main__":
    run_tree_of_thought(
        "Assess whether Competitor Alpha is preparing to enter the SMB market segment.",
        branching_factor=3,
        exploration_depth=2,
        prune_threshold=0.35,
    )