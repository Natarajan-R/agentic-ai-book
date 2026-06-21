"""
agents/tot_agent.py
====================
Chapter 11: Production Tree of Thought Agent

Extends the reference agent with Tree of Thought planning —
exploring multiple reasoning paths before committing,
then synthesising from the most productive branches.

What makes this production quality:
  ✓ Hard parameter controls (branching factor, depth, prune threshold,
    max active branches, convergence criterion)
  ✓ Branch scoring with multi-criteria evaluation (relevance, coverage,
    source quality, internal consistency)
  ✓ Explicit convergence — stops when evidence threshold met, not just
    when budget exhausted
  ✓ Contradiction detection across branches
  ✓ Token budget tracking per branch — prune expensive low-yield branches
  ✓ Structured branch registry with full audit trail
  ✓ Falls back to sequential planning when ToT is unnecessary

Run:
    python agents/tot_agent.py
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from config.settings import settings
from core.logging import get_logger
from core.tokens import count_tokens
from tools.all_tools import rag_retrieve, web_search

log = get_logger("tot_agent")


# ─── TREE OF THOUGHT PARAMETERS ──────────────────────────────────────────────

@dataclass
class ToTConfig:
    """
    All ToT parameters in one place.
    Never scattered as magic numbers through the code.
    """
    branching_factor:   int   = 3      # candidate approaches per branching point
    exploration_depth:  int   = 3      # steps per branch before scoring
    prune_threshold:    float = 0.40   # branches below this score are abandoned
    max_active_branches: int  = 4      # hard cap on simultaneous branches
    convergence_threshold: float = 0.85 # stop exploring when best branch reaches this
    max_total_tokens:   int   = 12000  # token budget for entire ToT process
    evidence_min_sources: int = 2      # minimum distinct sources per branch to score well


DEFAULT_TOT_CONFIG = ToTConfig()


# ─── BRANCH DATA STRUCTURES ──────────────────────────────────────────────────

class BranchStatus(str, Enum):
    ACTIVE   = "active"
    PRUNED   = "pruned"
    CONVERGED = "converged"
    EXHAUSTED = "exhausted"


@dataclass
class BranchScore:
    """
    Multi-criteria branch evaluation.
    Each criterion is scored 0-1 and weighted to produce a composite.
    """
    relevance:    float = 0.0   # how relevant to the goal
    coverage:     float = 0.0   # how much of the goal is addressed
    source_quality: float = 0.0 # quality and diversity of sources
    consistency:  float = 0.0   # internal consistency of findings
    token_efficiency: float = 0.0  # yield per token invested

    WEIGHTS = {
        "relevance":       0.30,
        "coverage":        0.25,
        "source_quality":  0.20,
        "consistency":     0.15,
        "token_efficiency": 0.10,
    }

    @property
    def composite(self) -> float:
        return sum(
            getattr(self, k) * w
            for k, w in self.WEIGHTS.items()
        )

    def to_dict(self) -> dict:
        return {
            "relevance":       self.relevance,
            "coverage":        self.coverage,
            "source_quality":  self.source_quality,
            "consistency":     self.consistency,
            "token_efficiency": self.token_efficiency,
            "composite":       round(self.composite, 3),
        }


@dataclass
class Branch:
    """A single reasoning path in the Tree of Thought."""
    branch_id:    str
    approach:     str             # description of this reasoning approach
    depth:        int = 0
    status:       BranchStatus = BranchStatus.ACTIVE
    evidence:     list[str] = field(default_factory=list)      # findings at each step
    sources:      list[str] = field(default_factory=list)      # sources cited
    tokens_used:  int = 0
    score:        BranchScore | None = None
    prune_reason: str = ""

    def add_evidence(self, finding: str, source: str = "", tokens: int = 0):
        self.evidence.append(finding)
        if source:
            self.sources.append(source)
        self.tokens_used += tokens
        self.depth += 1

    def evidence_summary(self, max_chars: int = 600) -> str:
        return " | ".join(self.evidence)[:max_chars]


# ─── TOT STATE ───────────────────────────────────────────────────────────────

class ToTState:
    """Manages the complete Tree of Thought execution state."""

    def __init__(self, goal: str, config: ToTConfig):
        self.goal          = goal
        self.config        = config
        self.branches:     dict[str, Branch] = {}
        self.total_tokens  = 0
        self.phase         = "generate"   # generate | explore | prune | synthesise
        self.run_id        = str(uuid.uuid4())[:8]
        self.contradictions: list[str] = []

    @property
    def active_branches(self) -> list[Branch]:
        return [b for b in self.branches.values() if b.status == BranchStatus.ACTIVE]

    @property
    def scored_branches(self) -> list[Branch]:
        return sorted(
            [b for b in self.branches.values() if b.score is not None],
            key=lambda b: b.score.composite,
            reverse=True,
        )

    def add_branch(self, approach: str) -> Branch:
        branch_id = f"B{len(self.branches)+1:02d}"
        branch = Branch(branch_id=branch_id, approach=approach)
        self.branches[branch_id] = branch
        return branch

    def prune_branch(self, branch_id: str, reason: str):
        if branch_id in self.branches:
            self.branches[branch_id].status = BranchStatus.PRUNED
            self.branches[branch_id].prune_reason = reason
            log.info(f"Pruned branch {branch_id}: {reason}")

    def budget_remaining(self) -> int:
        return self.config.max_total_tokens - self.total_tokens

    def should_converge(self) -> bool:
        if not self.scored_branches:
            return False
        best = self.scored_branches[0]
        return best.score.composite >= self.config.convergence_threshold

    def summary(self) -> dict:
        return {
            "run_id":          self.run_id,
            "total_branches":  len(self.branches),
            "active":          len(self.active_branches),
            "pruned":          sum(1 for b in self.branches.values() if b.status == BranchStatus.PRUNED),
            "total_tokens":    self.total_tokens,
            "phase":           self.phase,
            "contradictions":  len(self.contradictions),
        }


# ─── TOT ENGINE ──────────────────────────────────────────────────────────────

class TreeOfThoughtEngine:
    """
    Production Tree of Thought reasoning engine.

    Phases:
      1. Generate  — produce branching_factor candidate approaches
      2. Explore   — gather evidence for each branch (exploration_depth steps)
      3. Score     — evaluate each branch on 5 criteria
      4. Prune     — abandon branches below prune_threshold
      5. Converge  — check if best branch meets convergence_threshold
      6. Synthesise — combine evidence from best branches into final answer
    """

    def __init__(self, config: ToTConfig = DEFAULT_TOT_CONFIG):
        self.config = config
        self._llm_primary   = ChatOllama(model=settings.model_primary,   temperature=0.2)
        self._llm_secondary = ChatOllama(model=settings.model_secondary,  temperature=0)
        self._llm_scorer    = ChatOllama(model=settings.model_secondary,  temperature=0)
        self._tools = [web_search, rag_retrieve]

    def run(self, goal: str) -> dict[str, Any]:
        state = ToTState(goal, self.config)

        print(f"\n{'═' * 65}")
        print(f"Tree of Thought: {goal[:60]}")
        print(f"Config: branching={self.config.branching_factor}, "
              f"depth={self.config.exploration_depth}, "
              f"prune={self.config.prune_threshold}")
        print(f"{'═' * 65}")

        # Phase 1: Generate candidate approaches
        print(f"\n[Phase 1] Generating {self.config.branching_factor} candidate approaches ...")
        self._phase_generate(state)

        # Main exploration loop
        for cycle in range(self.config.exploration_depth):
            if not state.active_branches:
                break
            if state.budget_remaining() < 500:
                print(f"  Token budget exhausted at cycle {cycle+1}")
                break

            # Phase 2: Explore all active branches one step
            print(f"\n[Phase 2.{cycle+1}] Exploring {len(state.active_branches)} active branches ...")
            self._phase_explore(state)

            # Phase 3: Score all branches
            print(f"\n[Phase 3.{cycle+1}] Scoring branches ...")
            self._phase_score(state)

            # Phase 4: Prune low-scoring branches
            print(f"\n[Phase 4.{cycle+1}] Pruning ...")
            self._phase_prune(state)

            # Phase 5: Check convergence
            if state.should_converge():
                best = state.scored_branches[0]
                print(f"\n[CONVERGED] Branch {best.branch_id} reached "
                      f"{best.score.composite:.0%} ≥ threshold "
                      f"{self.config.convergence_threshold:.0%}")
                break

        # Detect contradictions across branches
        self._detect_contradictions(state)

        # Phase 6: Synthesise
        print(f"\n[Phase 6] Synthesising from best branches ...")
        final_answer = self._phase_synthesise(state)

        print(f"\n{'─' * 65}")
        print(f"Summary: {state.summary()}")

        return {
            "answer":        final_answer,
            "state_summary": state.summary(),
            "best_branches": [
                {"id": b.branch_id, "approach": b.approach[:60], "score": b.score.composite}
                for b in state.scored_branches[:3]
            ],
            "contradictions": state.contradictions,
            "total_tokens":  state.total_tokens,
        }

    # ── Phases ────────────────────────────────────────────────────────────

    def _phase_generate(self, state: ToTState):
        """Generate branching_factor meaningfully different approaches."""
        response = self._llm_primary.invoke([
            SystemMessage(content=(
                "You are a strategic research planner. Generate candidate "
                "approaches for investigating the given goal. Each approach "
                "should use different information sources or analytical methods. "
                f"Generate exactly {self.config.branching_factor} approaches. "
                'Return JSON: [{"approach": "description", "first_step": "action"}]'
            )),
            HumanMessage(content=f"Goal: {state.goal}"),
        ])

        tokens = count_tokens(response.content or "")
        state.total_tokens += tokens

        try:
            approaches = json.loads(response.content or "[]")
        except Exception:
            approaches = [
                {"approach": f"Approach {i+1}: standard investigation path {i+1}",
                 "first_step": f"Step {i+1}"}
                for i in range(self.config.branching_factor)
            ]

        # Enforce branching factor limit
        approaches = approaches[:self.config.branching_factor]

        for a in approaches:
            branch = state.add_branch(a.get("approach", "unnamed"))
            print(f"  {branch.branch_id}: {branch.approach[:60]}")

    def _phase_explore(self, state: ToTState):
        """Execute one exploration step for each active branch."""
        # Enforce max active branches cap
        active = state.active_branches[:self.config.max_active_branches]

        for branch in active:
            if state.budget_remaining() < 200:
                break

            prompt = (
                f"Goal: {state.goal}\n"
                f"Approach: {branch.approach}\n"
                f"Depth: {branch.depth + 1}/{self.config.exploration_depth}\n"
                f"Previous findings: {branch.evidence_summary(300)}\n\n"
                "Execute the next exploration step. Search for specific evidence. "
                "Report: (1) what you found, (2) your source, (3) relevance to goal. "
                "Be concise — max 150 words."
            )

            # Use tools for real evidence gathering
            tool_result = ""
            if branch.depth == 0:
                # First step: search relevant sources
                try:
                    tool_result = web_search.invoke(
                        {"query": f"{state.goal[:50]} {branch.approach[:30]}"}
                    )
                    tool_result = tool_result[:400]
                except Exception:
                    pass

            response = self._llm_primary.invoke([
                HumanMessage(content=prompt + (
                    f"\n\nSearch results available:\n{tool_result}" if tool_result else ""
                )),
            ])

            finding  = (response.content or "")[:300]
            tokens   = count_tokens(prompt + finding)
            source   = "web_search" if tool_result else "llm_reasoning"

            branch.add_evidence(finding, source, tokens)
            state.total_tokens += tokens

            print(f"  {branch.branch_id} (depth {branch.depth}): "
                  f"{finding[:80]}...")

    def _phase_score(self, state: ToTState):
        """Score each active branch on 5 criteria."""
        for branch in state.active_branches:
            if not branch.evidence:
                branch.score = BranchScore()
                continue

            score_prompt = (
                f"Goal: {state.goal}\n\n"
                f"Branch approach: {branch.approach}\n"
                f"Evidence gathered:\n{branch.evidence_summary(400)}\n"
                f"Sources used: {', '.join(set(branch.sources)) or 'none'}\n"
                f"Tokens invested: {branch.tokens_used}\n\n"
                "Score this research branch on each criterion (0.0 to 1.0):\n"
                "- relevance:     How directly relevant is the evidence to the goal?\n"
                "- coverage:      What fraction of the goal is addressed?\n"
                "- source_quality: How credible and diverse are the sources?\n"
                "- consistency:   Is the evidence internally consistent?\n"
                "- token_efficiency: Evidence quality per token invested?\n\n"
                'Return JSON only: {"relevance": 0.0, "coverage": 0.0, '
                '"source_quality": 0.0, "consistency": 0.0, "token_efficiency": 0.0}'
            )

            response = self._llm_scorer.invoke([HumanMessage(content=score_prompt)])
            state.total_tokens += count_tokens(score_prompt)

            try:
                raw = json.loads(response.content or "{}")
                branch.score = BranchScore(
                    relevance=float(raw.get("relevance", 0.5)),
                    coverage=float(raw.get("coverage", 0.5)),
                    source_quality=float(raw.get("source_quality", 0.5)),
                    consistency=float(raw.get("consistency", 0.5)),
                    token_efficiency=float(raw.get("token_efficiency", 0.5)),
                )
            except Exception:
                branch.score = BranchScore(
                    relevance=0.5, coverage=0.5,
                    source_quality=0.5, consistency=0.5, token_efficiency=0.5,
                )

            composite = branch.score.composite
            symbol = "✓" if composite >= self.config.prune_threshold else "✗"
            print(f"  {symbol} {branch.branch_id}: composite={composite:.2f} "
                  f"(rel={branch.score.relevance:.2f}, "
                  f"cov={branch.score.coverage:.2f})")

    def _phase_prune(self, state: ToTState):
        """Prune branches below the threshold."""
        pruned = 0
        for branch in state.active_branches:
            if branch.score and branch.score.composite < self.config.prune_threshold:
                state.prune_branch(
                    branch.branch_id,
                    f"Score {branch.score.composite:.2f} below threshold {self.config.prune_threshold}",
                )
                pruned += 1

        # Also enforce max active branches
        active = state.active_branches
        if len(active) > self.config.max_active_branches:
            to_prune = sorted(active, key=lambda b: b.score.composite if b.score else 0)
            for b in to_prune[:len(active) - self.config.max_active_branches]:
                state.prune_branch(b.branch_id, "Max active branches cap")
                pruned += 1

        print(f"  Pruned {pruned} branches. Active: {len(state.active_branches)}")

    def _detect_contradictions(self, state: ToTState):
        """Detect contradictions across branches."""
        branches_with_evidence = [b for b in state.branches.values() if b.evidence]
        if len(branches_with_evidence) < 2:
            return

        all_evidence = "\n".join(
            f"{b.branch_id}: {b.evidence_summary(200)}"
            for b in branches_with_evidence
        )

        response = self._llm_secondary.invoke([
            SystemMessage(content="Identify any direct contradictions between research findings."),
            HumanMessage(content=(
                f"Goal: {state.goal}\n\nFindings from different research branches:\n{all_evidence}\n\n"
                "List any direct factual contradictions. "
                'Return JSON: {"contradictions": ["description of each contradiction"]}'
            )),
        ])

        try:
            raw = json.loads(response.content or "{}")
            state.contradictions = raw.get("contradictions", [])
            if state.contradictions:
                print(f"  ⚠ {len(state.contradictions)} contradictions detected")
        except Exception:
            pass

    def _phase_synthesise(self, state: ToTState) -> str:
        """Synthesise the final answer from the best branches."""
        # Use top branches (up to 3)
        best = state.scored_branches[:3]
        if not best:
            # Fallback to all branches if none scored
            best = list(state.branches.values())[:3]

        evidence_text = "\n\n".join(
            f"**Branch {b.branch_id}** (score: {b.score.composite:.0%} if b.score else 'unscored')\n"
            f"Approach: {b.approach}\n"
            f"Evidence: {b.evidence_summary(500)}"
            for b in best
        )

        contradiction_note = ""
        if state.contradictions:
            contradiction_note = (
                "\n\nNOTE — The following contradictions were detected between research branches:\n"
                + "\n".join(f"- {c}" for c in state.contradictions)
                + "\nThese should be flagged in the final answer."
            )

        response = self._llm_primary.invoke([
            SystemMessage(content=(
                "You are synthesising research from multiple investigation branches. "
                "Produce a comprehensive, well-structured final answer. "
                "Attribute claims to their source branches. "
                "Flag any contradictions or uncertainties explicitly."
            )),
            HumanMessage(content=(
                f"Goal: {state.goal}\n\n"
                f"Research from best branches:\n{evidence_text}"
                f"{contradiction_note}\n\n"
                "Synthesise a final comprehensive answer."
            )),
        ])

        return response.content or "Synthesis failed to produce output."


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

class ToTAgent:
    """
    Production Tree of Thought agent.

    Automatically decides whether ToT is warranted based on
    goal complexity — falls back to sequential reasoning for simple tasks.

    Usage:
        agent = ToTAgent()
        result = agent.run("Assess whether Competitor Alpha is preparing to enter the SMB market.")
        print(result["answer"])
    """

    COMPLEXITY_KEYWORDS = {
        "assess", "evaluate", "investigate", "analyse", "determine",
        "explore", "examine", "consider", "whether", "why", "how might",
        "what factors", "comprehensive", "thorough", "in-depth",
    }

    def __init__(self, config: ToTConfig = DEFAULT_TOT_CONFIG):
        self.engine    = TreeOfThoughtEngine(config)
        self._fallback = ChatOllama(model=settings.model_primary, temperature=0)

    def _warrants_tot(self, goal: str) -> bool:
        """
        Determine whether the goal warrants Tree of Thought exploration.
        Simple factual queries do not need it — sequential planning is faster.
        """
        goal_lower = goal.lower()
        return any(kw in goal_lower for kw in self.COMPLEXITY_KEYWORDS)

    def run(self, goal: str) -> dict[str, Any]:
        if self._warrants_tot(goal):
            print(f"[ToT] Goal complexity warrants Tree of Thought exploration")
            return self.engine.run(goal)
        else:
            print(f"[Sequential] Simple goal — using direct reasoning")
            response = self._fallback.invoke([HumanMessage(content=goal)])
            return {
                "answer":     response.content,
                "method":     "sequential",
                "total_tokens": count_tokens(response.content or ""),
            }


if __name__ == "__main__":
    agent = ToTAgent()

    # Complex ambiguous goal — warrants ToT
    result1 = agent.run(
        "Assess whether Competitor Alpha is preparing to enter the SMB market segment. "
        "Consider multiple evidence types: hiring patterns, product announcements, "
        "pricing changes, and partnership signals."
    )
    print(f"\nAnswer:\n{result1['answer'][:600]}")
    if "best_branches" in result1:
        print(f"\nBest branches: {result1['best_branches']}")

    # Simple factual — direct reasoning
    result2 = agent.run("What is Competitor Beta's enterprise pricing?")
    print(f"\nDirect answer:\n{result2['answer'][:200]}")
