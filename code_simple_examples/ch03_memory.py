"""
Chapter 3: Memory — Four Types Working Together
================================================
Demonstrates in-context, episodic, semantic (RAG via ChromaDB),
and structured memory — all feeding into one context window.

Run:  python ch03_memory.py
Need: ollama pull qwen2.5:7b
      pip install chromadb
"""

import json
from datetime import datetime, timezone
import ollama
import chromadb

MODEL = "qwen2.5:7b"


# ─── SETUP: SEMANTIC MEMORY STORE (RAG) ──────────────────────────────────────

def build_knowledge_base() -> chromadb.Collection:
    """
    Create and seed a ChromaDB vector store.
    In production: populated by document ingestion pipelines.
    """
    client = chromadb.Client()
    kb = client.create_collection("enterprise_knowledge")
    kb.add(
        documents=[
            "Our Enterprise tier is priced at $299 per seat per month.",
            "Competitor Alpha charges $350 per seat per month for similar features.",
            "Competitor Beta offers a freemium model capped at 5 users.",
            "Competitor Gamma recently cut prices by 15% to gain market share.",
            "Our product roadmap includes multi-agent orchestration in Q1 next year.",
            "Customer satisfaction scores average 4.6/5 across enterprise accounts.",
            "The sales cycle for enterprise accounts averages 45 days.",
        ],
        ids=[f"doc_{i}" for i in range(7)],
    )
    return kb


# ─── THE FOUR MEMORY TYPES ───────────────────────────────────────────────────

class AgentMemory:
    """
    Manages all four memory types and assembles
    them into a context window before each LLM call.
    """

    def __init__(self, kb: chromadb.Collection):
        # Type 1: In-context — assembled fresh each call (not stored here)

        # Type 2: Episodic — conversation history
        self.episodic: list[dict] = []

        # Type 3: Semantic — vector store handle
        self.knowledge_base = kb

        # Type 4: Structured — long-term facts with known schema
        self.structured: dict = {
            "user_name":        "Alex",
            "user_role":        "Sales Director",
            "preferred_format": "bullet points",
            "session_start":    datetime.now(timezone.utc).isoformat(),
        }

        # Scratchpad — temporary working notes, cleared each task
        self.scratchpad: dict = {}

    # ── Semantic retrieval ────────────────────────────────────────────────────

    def retrieve(self, query: str, n: int = 3) -> list[str]:
        """Query the vector store by semantic similarity (RAG)."""
        results = self.knowledge_base.query(
            query_texts=[query],
            n_results=n,
        )
        return results["documents"][0]

    # ── Context window assembly ───────────────────────────────────────────────

    def assemble(self, query: str) -> list[dict]:
        """
        Assemble in-context memory from all sources.
        This runs before EVERY LLM call.
        """
        # Retrieve semantically relevant facts
        semantic_facts = self.retrieve(query)

        # Build system message from structured + semantic memory
        system_content = (
            f"You are an assistant for {self.structured['user_name']}, "
            f"{self.structured['user_role']}. "
            f"Format responses as {self.structured['preferred_format']}.\n\n"
            "Relevant knowledge:\n" +
            "\n".join(f"- {fact}" for fact in semantic_facts)
        )

        # If scratchpad has notes, include them
        if self.scratchpad:
            system_content += (
                "\n\nWorking notes from this session:\n" +
                "\n".join(f"- {k}: {v}" for k, v in self.scratchpad.items())
            )

        messages = [{"role": "system", "content": system_content}]

        # Add recent episodic history (last 3 turns to control window size)
        for turn in self.episodic[-6:]:  # 3 pairs = 6 entries
            messages.append(turn)

        # Add current query
        messages.append({"role": "user", "content": query})
        return messages

    # ── Update after each turn ────────────────────────────────────────────────

    def record(self, user_msg: str, assistant_msg: str):
        """Store this turn in episodic memory."""
        self.episodic.append({"role": "user",      "content": user_msg})
        self.episodic.append({"role": "assistant",  "content": assistant_msg})

    def note(self, key: str, value: str):
        """Write to scratchpad — temporary task-scoped notes."""
        self.scratchpad[key] = value

    def clear_scratchpad(self):
        """Clear working notes at task completion."""
        self.scratchpad.clear()

    def stats(self) -> dict:
        """Observability: how much memory is in use."""
        return {
            "episodic_turns":    len(self.episodic) // 2,
            "scratchpad_keys":   len(self.scratchpad),
            "structured_keys":   len(self.structured),
        }


# ─── AGENT USING ALL FOUR MEMORY TYPES ───────────────────────────────────────

def run_memory_demo():
    print("Setting up knowledge base ...")
    kb = build_knowledge_base()
    memory = AgentMemory(kb)

    print(f"\nUser: {memory.structured['user_name']} "
          f"({memory.structured['user_role']})")
    print(f"Format preference: {memory.structured['preferred_format']}")

    queries = [
        "How does our pricing compare to competitors?",
        "Based on that comparison, what is our strongest argument to a price-sensitive prospect?",
        "What upcoming product capability could help close deals this quarter?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 55}")
        print(f"Turn {i}: {query}")

        # Assemble context from all memory types
        messages = memory.assemble(query)
        token_estimate = sum(len(m["content"].split()) for m in messages)
        print(f"Context: {len(messages)} messages, ~{token_estimate} tokens")

        # LLM call
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": 0},
        )
        answer = response.message.content

        # Update episodic memory
        memory.record(query, answer)

        # Write useful facts to scratchpad
        if "compet" in query.lower():
            memory.note(f"competitive_insight_{i}", answer[:120])

        print(f"\nAnswer:\n{answer}")

    print(f"\n{'─' * 55}")
    print("Memory stats at end of session:")
    print(json.dumps(memory.stats(), indent=2))

    print("\nClearing scratchpad for next task ...")
    memory.clear_scratchpad()
    print(f"Stats after clear: {memory.stats()}")


if __name__ == "__main__":
    run_memory_demo()
