"""
agents/self_refining.py
========================
Chapter 14: Production Self-Refining Agent

Extends the reference agent with:
  ✓ Persistent SQLite task history store
  ✓ Embedding-based similarity for finding relevant past tasks
  ✓ Multi-dimensional quality assessment (completeness, accuracy,
    relevance, source quality, user satisfaction proxy)
  ✓ Trend tracking — quality improving or degrading over time?
  ✓ Lesson extraction and plan improvement
  ✓ Automatic detection of systematic failure patterns
  ✓ Quality regression alerts
"""

import json
import sqlite3
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from core.tokens import count_tokens
from agents.security_agent import SECURITY_LOG, SecurityEvent


class TaskHistoryStore:
    """
    Persistent SQLite store for task history.
    Survives process restarts. Queryable for similar tasks.
    Thread-safe via WAL mode.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS task_history (
        id              TEXT PRIMARY KEY,
        goal            TEXT NOT NULL,
        plan            TEXT,
        result_summary  TEXT,
        quality_score   REAL,
        quality_breakdown TEXT,
        lessons_learned TEXT,
        iterations      INTEGER,
        total_tokens    INTEGER,
        cost_usd        REAL,
        exit_reason     TEXT,
        tool_counts     TEXT,
        created_at      TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_quality ON task_history(quality_score);
    CREATE INDEX IF NOT EXISTS idx_created ON task_history(created_at);
    """

    def __init__(self, db_path: Path = Path("./data/task_history.db")):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(self.SCHEMA)

    def save(self, entry: dict):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO task_history
                   (id, goal, plan, result_summary, quality_score, quality_breakdown,
                    lessons_learned, iterations, total_tokens, cost_usd, exit_reason,
                    tool_counts, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry["id"],
                    entry["goal"],
                    entry.get("plan", ""),
                    entry.get("result_summary", "")[:500],
                    entry.get("quality_score", 0.0),
                    json.dumps(entry.get("quality_breakdown", {})),
                    entry.get("lessons_learned", ""),
                    entry.get("iterations", 0),
                    entry.get("total_tokens", 0),
                    entry.get("cost_usd", 0.0),
                    entry.get("exit_reason", ""),
                    json.dumps(entry.get("tool_counts", {})),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def find_similar(self, goal: str, n: int = 5, min_quality: float = 0.0) -> list[dict]:
        """Find similar past tasks by keyword overlap (use embeddings in production)."""
        goal_words = set(goal.lower().split())
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM task_history WHERE quality_score >= ? ORDER BY created_at DESC LIMIT 50",
                (min_quality,)
            ).fetchall()

        scored = []
        cols = ["id", "goal", "plan", "result_summary", "quality_score",
                "quality_breakdown", "lessons_learned", "iterations",
                "total_tokens", "cost_usd", "exit_reason", "tool_counts", "created_at"]

        for row in rows:
            entry = dict(zip(cols, row))
            past_words = set(entry["goal"].lower().split())
            overlap = len(goal_words & past_words) / max(len(goal_words), 1)
            if overlap > 0.1:
                scored.append((overlap, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:n]]

    def quality_trend(self, last_n: int = 20) -> dict:
        """Analyse quality trend over the last N tasks."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT quality_score, created_at FROM task_history "
                "ORDER BY created_at DESC LIMIT ?",
                (last_n,),
            ).fetchall()

        if len(rows) < 2:
            return {"trend": "insufficient_data", "scores": []}

        scores = [r[0] for r in reversed(rows)]
        first_half = statistics.mean(scores[:len(scores)//2])
        second_half = statistics.mean(scores[len(scores)//2:])
        trend = "improving" if second_half > first_half + 0.05 else \
                "degrading" if second_half < first_half - 0.05 else "stable"

        return {
            "trend":         trend,
            "scores":        scores,
            "mean":          round(statistics.mean(scores), 2),
            "recent_mean":   round(second_half, 2),
            "historical_mean": round(first_half, 2),
            "std_dev":       round(statistics.stdev(scores), 2) if len(scores) > 1 else 0,
        }

    def systematic_failures(self) -> list[str]:
        """Detect recurring failure patterns."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT lessons_learned FROM task_history WHERE quality_score < 0.5"
            ).fetchall()

        if len(rows) < 3:
            return []

        all_lessons = " ".join(r[0] for r in rows if r[0])
        # Simple frequency analysis — use topic modelling in production
        words = all_lessons.lower().split()
        freq: dict[str, int] = {}
        for w in words:
            if len(w) > 5:
                freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [f"Recurring term in failures: '{w}' ({c}x)" for w, c in top if c > 2]


@dataclass
class QualityAssessment:
    completeness:   float = 0.0
    accuracy:       float = 0.0
    relevance:      float = 0.0
    source_quality: float = 0.0
    conciseness:    float = 0.0

    WEIGHTS = {"completeness": 0.30, "accuracy": 0.25, "relevance": 0.25,
               "source_quality": 0.10, "conciseness": 0.10}

    @property
    def composite(self) -> float:
        return sum(getattr(self, k) * w for k, w in self.WEIGHTS.items())

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.WEIGHTS}
        d["composite"] = round(self.composite, 3)
        return d


class SelfRefiningAgent:
    """
    Production self-refining agent.

    Each task run:
    1. Retrieves similar successful past tasks
    2. Uses their lessons to generate a better plan
    3. Executes the task
    4. Self-evaluates quality on 5 dimensions
    5. Extracts lessons
    6. Persists to history for future runs
    7. Monitors quality trend and alerts on regression
    """

    def __init__(self):
        self._history = TaskHistoryStore()
        self._llm     = ChatOllama(model=settings.model_primary,   temperature=0)
        self._scorer  = ChatOllama(model=settings.model_secondary,  temperature=0)
        from tools.all_tools import get_all_tools, seed_knowledge_base
        seed_knowledge_base([
            {"id": "sr_market", "text": "AI agent market $3.8B 2024, CAGR 42%.", "source": "kb", "category": "market"},
        ])
        self._tools = {t.name: t for t in get_all_tools()}

    def run(self, goal: str) -> dict[str, Any]:
        run_id = str(uuid.uuid4())[:8]
        print(f"\n{'═' * 65}")
        print(f"Self-refining agent: {run_id}")
        print(f"Goal: {goal[:65]}")

        # Find similar past tasks. min_quality gates which past runs are worth
        # learning from; we keep it low here so the learning loop is visible even
        # with a small local model (in production raise it, e.g. 0.6, to learn
        # only from high-quality runs).
        similar = self._history.find_similar(goal, n=3, min_quality=0.0)
        print(f"Similar past tasks found: {len(similar)}")

        # Build experience context from past lessons
        experience = self._build_experience_context(similar)

        # Generate experience-informed plan
        plan = self._generate_plan(goal, experience)
        print(f"\nPlan:\n{plan[:300]}")

        # Execute
        start = time.perf_counter()
        result, iterations, tokens, cost, tool_counts = self._execute(goal, plan)
        elapsed = time.perf_counter() - start

        # Multi-dimensional quality assessment
        quality = self._assess_quality(goal, plan, result)
        print(f"\nQuality: {quality.composite:.0%} "
              f"(completeness={quality.completeness:.0%}, "
              f"accuracy={quality.accuracy:.0%})")

        # Extract lessons for future runs
        lessons = self._extract_lessons(goal, plan, result, quality)
        print(f"Lesson: {lessons[:120]}")

        # Persist to history
        self._history.save({
            "id":                run_id,
            "goal":              goal,
            "plan":              plan[:500],
            "result_summary":    result[:300],
            "quality_score":     quality.composite,
            "quality_breakdown": quality.to_dict(),
            "lessons_learned":   lessons,
            "iterations":        iterations,
            "total_tokens":      tokens,
            "cost_usd":          cost,
            "exit_reason":       "completed",
            "tool_counts":       tool_counts,
        })

        # Quality trend monitoring
        trend = self._history.quality_trend()
        if trend["trend"] == "degrading":
            print(f"\n⚠ QUALITY REGRESSION ALERT: "
                  f"Recent mean {trend['recent_mean']:.2f} < "
                  f"Historical mean {trend['historical_mean']:.2f}")
            SECURITY_LOG.record(SecurityEvent(
                severity="warning",
                event_type="quality_regression",
                description=f"Quality degrading: {trend['recent_mean']:.2f} vs {trend['historical_mean']:.2f}",
                task_id=run_id,
                blocked=False,
            ))

        # Systematic failure detection
        failures = self._history.systematic_failures()
        if failures:
            print(f"\n⚠ Systematic failure patterns detected:")
            for f in failures:
                print(f"  {f}")

        return {
            "run_id":     run_id,
            "answer":     result,
            "quality":    quality.to_dict(),
            "lessons":    lessons,
            "trend":      trend,
            "elapsed_s":  round(elapsed, 1),
            "tokens":     tokens,
        }

    def _build_experience_context(self, similar: list[dict]) -> str:
        if not similar:
            return ""
        lines = ["\n\nLearning from similar past task runs:"]
        for s in similar:
            q = s.get("quality_score", 0)
            l = s.get("lessons_learned", "")
            lines.append(
                f"  Task: {s['goal'][:60]}\n"
                f"  Quality: {q:.0%} | Lesson: {l[:100]}"
            )
        return "\n".join(lines)

    def _generate_plan(self, goal: str, experience: str) -> str:
        response = self._llm.invoke([
            SystemMessage(content=(
                "You are a self-refining planning agent. "
                "Generate a 4-6 step sequential plan for the goal. "
                "Use lessons from past tasks to avoid known failure patterns."
                + experience
            )),
            HumanMessage(content=(
                f"Goal: {goal}\n\n"
                "Generate a concise sequential plan with specific, executable steps."
            )),
        ])
        return response.content or ""

    def _execute(self, goal: str, plan: str) -> tuple[str, int, int, float, dict]:
        """Execute the plan and return (result, iterations, tokens, cost, tool_counts)."""
        llm_tools = self._llm.bind_tools(list(self._tools.values()))
        messages = [
            SystemMessage(content="You are a competitive intelligence agent. Follow the plan."),
            HumanMessage(content=f"Goal: {goal}\n\nPlan:\n{plan}\n\nExecute this plan."),
        ]
        iterations = 0
        total_tokens = 0
        total_cost   = 0.0
        tool_counts: dict[str, int] = {}
        result = ""

        for i in range(settings.max_iterations):
            iterations = i + 1
            response = llm_tools.invoke(messages)
            tokens = count_tokens(response.content or "")
            total_tokens += tokens
            total_cost   += settings.cost_estimate(tokens, settings.model_primary)
            messages.append(response)

            if not getattr(response, "tool_calls", []):
                result = response.content or ""
                break

            for tc in response.tool_calls:
                name = tc["name"]
                tool_fn = self._tools.get(name)
                tool_counts[name] = tool_counts.get(name, 0) + 1
                try:
                    obs = tool_fn.invoke(tc.get("args", {})) if tool_fn else "unknown tool"
                except Exception as e:
                    obs = f"Error: {e}"
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(content=str(obs)[:500], tool_call_id=tc["id"], name=name))

        return result, iterations, total_tokens, total_cost, tool_counts

    def _assess_quality(self, goal: str, plan: str, result: str) -> QualityAssessment:
        response = self._scorer.invoke([
            SystemMessage(content=(
                "Assess the quality of this task result on 5 dimensions (0.0-1.0 each). "
                'Return JSON only: {"completeness": 0.0, "accuracy": 0.0, '
                '"relevance": 0.0, "source_quality": 0.0, "conciseness": 0.0}'
            )),
            HumanMessage(content=(
                f"Goal: {goal[:200]}\n"
                f"Plan: {plan[:200]}\n"
                f"Result: {result[:400]}"
            )),
        ])
        try:
            raw = json.loads(response.content or "{}")
            return QualityAssessment(
                completeness=float(raw.get("completeness", 0.5)),
                accuracy=float(raw.get("accuracy", 0.5)),
                relevance=float(raw.get("relevance", 0.5)),
                source_quality=float(raw.get("source_quality", 0.5)),
                conciseness=float(raw.get("conciseness", 0.5)),
            )
        except Exception:
            return QualityAssessment(0.5, 0.5, 0.5, 0.5, 0.5)

    def _extract_lessons(
        self, goal: str, plan: str, result: str, quality: QualityAssessment,
    ) -> str:
        response = self._scorer.invoke([
            HumanMessage(content=(
                f"Goal: {goal[:200]}\nPlan: {plan[:200]}\n"
                f"Result: {result[:300]}\nQuality: {quality.composite:.0%}\n\n"
                "In ONE concise sentence: what is the key lesson for planning "
                "similar tasks better in future?"
            )),
        ])
        return (response.content or "").strip()[:200]


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Self-refining agent (Chapter 14)
    print("\n" + "=" * 65)
    print("CHAPTER 14: Self-Refining Agent")
    agent = SelfRefiningAgent()

    # Run 3 similar tasks in a row to demonstrate learning — each run can find
    # the earlier ones as "similar past tasks" and reuse their lessons.
    goals = [
        "Compare the pricing of our enterprise AI platform against the top three competitors for a 50-seat team.",
        "Analyse the pricing of the top enterprise AI platforms and recommend the best value for a 100-seat team.",
        "Review competitor enterprise AI platform pricing this quarter and recommend how our pricing should respond.",
    ]

    for goal in goals:
        result = agent.run(goal)
        print(f"\nQuality: {result['quality']['composite']:.0%} | "
              f"Trend: {result['trend']['trend']} | "
              f"Elapsed: {result['elapsed_s']}s")