"""
memory/assembler.py
====================
Production context window assembler.

Responsibilities:
  - Retrieve episodic history (with compression when needed)
  - Query semantic memory (RAG) for relevant knowledge
  - Assemble all sources into a context window within token budget
  - Compress when approaching the token limit
  - Run before EVERY LLM call — not just the first one
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_ollama import ChatOllama

from config.settings import settings
from core.logging import AgentLogger
from core.tokens import count_message_tokens, truncate_to_tokens


# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Enterprise Competitive Intelligence Agent.

IDENTITY
You research and analyse competitor pricing, products, market positioning,
and strategic developments on behalf of enterprise sales and strategy teams.

AUTHORISED SCOPE
- Competitor analysis: pricing models, product features, market positioning
- Market research: size, growth rates, trends, analyst reports
- Public information: announcements, press releases, job postings, SEC filings

PERMITTED ACTIONS
- web_search:    Search public web for current information
- rag_retrieve:  Query internal knowledge base
- code_execute:  Perform precise calculations (ALWAYS use for arithmetic)
- file_write:    Write reports and deliverables
- file_read:     Read previously written files
- human_escalate: Escalate ambiguous or high-stakes decisions

PROHIBITED ACTIONS
- Accessing competitor internal systems or confidential data
- Contacting competitor personnel
- Storing data outside the current task's output directory
- Sending any external communication
- Making commitments on behalf of the organisation

REASONING PRINCIPLES
1. Always use code_execute for calculations — never compute mentally
2. Distinguish verified facts from inferences explicitly
3. Express confidence levels for key claims
4. If uncertain about scope, escalate via human_escalate
5. When a task is complete, provide a clear, structured final answer

RESPONSE FORMAT
For tool calls: use the structured tool call format
For final answers: use the format requested by the user (or markdown if unspecified)"""


# ─── CONTEXT ASSEMBLER ───────────────────────────────────────────────────────

class ContextAssembler:
    """
    Assembles the context window before every LLM call.

    Design:
    - Starts with system prompt + semantic retrieval
    - Adds compressed episodic history
    - Adds current messages from this task run
    - Compresses if total exceeds the token budget
    """

    def __init__(self, logger: AgentLogger):
        self._logger     = logger
        self._compressor = ChatOllama(model=settings.model_secondary, temperature=0)

    def assemble(
        self,
        goal:           str,
        messages:       list[BaseMessage],
        episodic:       list[dict[str, str]],
        scratchpad:     dict[str, str],
        iteration:      int,
    ) -> list[BaseMessage]:
        """
        Build the complete context window for an LLM call.

        Returns a list of BaseMessage objects ready for llm.invoke().
        Always within the configured token budget.
        """
        # 1. Semantic retrieval (RAG)
        rag_context = self._retrieve_semantic(goal)

        # 2. Build system message
        system_content = SYSTEM_PROMPT
        if rag_context:
            system_content += f"\n\nRELEVANT KNOWLEDGE BASE CONTEXT:\n{rag_context}"
        if scratchpad:
            notes = "\n".join(f"- {k}: {v[:150]}" for k, v in scratchpad.items())
            system_content += f"\n\nWORKING NOTES (this session):\n{notes}"

        assembled: list[BaseMessage] = [SystemMessage(content=system_content)]

        # 3. Add episodic history (compressed if long)
        episodic_msgs = self._format_episodic(episodic)
        assembled.extend(episodic_msgs)

        # 4. Add current task messages
        assembled.extend(messages)

        # 5. Check token budget and compress if needed
        token_count = count_message_tokens(assembled)
        budget      = settings.max_tokens_context
        threshold   = int(budget * settings.context_compression_threshold)

        if token_count > threshold:
            assembled = self._compress(assembled, goal, iteration)
            token_count = count_message_tokens(assembled)

        rag_chunks = rag_context.count("---") + 1 if rag_context else 0
        self._logger.context_assembled(
            iteration=iteration,
            message_count=len(assembled),
            token_count=token_count,
            rag_chunks=rag_chunks,
        )
        return assembled

    # ── Private helpers ──────────────────────────────────────────────────

    def _retrieve_semantic(self, query: str) -> str:
        """Query the vector store and return formatted results."""
        try:
            from tools.all_tools import rag_retrieve
            result = rag_retrieve.invoke({"query": query})
            return truncate_to_tokens(result, settings.rag_top_k * settings.rag_max_chunk_tokens)
        except Exception:
            return ""   # degrade gracefully — no RAG is better than crashing

    def _format_episodic(self, episodic: list[dict[str, str]]) -> list[BaseMessage]:
        """
        Convert episodic history dicts to BaseMessage objects.
        If history is long, summarise the older portion.
        """
        if not episodic:
            return []

        keep = settings.episodic_keep_recent * 2   # pairs of user/assistant
        if len(episodic) <= keep:
            return self._dict_to_messages(episodic)

        # Summarise older history
        older   = episodic[:-keep]
        recent  = episodic[-keep:]

        older_text = "\n".join(
            f"{e['role'].upper()}: {e['content'][:200]}"
            for e in older
        )
        summary_resp = self._compressor.invoke([
            HumanMessage(content=(
                "Summarise this conversation history in 3-4 sentences. "
                "Preserve key decisions, facts found, and conclusions reached:\n\n"
                + older_text
            ))
        ])
        summary = summary_resp.content if hasattr(summary_resp, "content") else str(summary_resp)

        return [
            SystemMessage(content=f"[CONVERSATION HISTORY SUMMARY]: {summary}"),
            *self._dict_to_messages(recent),
        ]

    @staticmethod
    def _dict_to_messages(episodic: list[dict]) -> list[BaseMessage]:
        messages = []
        for e in episodic:
            role    = e.get("role", "user")
            content = e.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages

    def _compress(
        self,
        messages: list[BaseMessage],
        goal: str,
        iteration: int,
    ) -> list[BaseMessage]:
        """
        Compress the message list to fit within the token budget.

        Strategy:
        - Always preserve: system message, first user message, last 4 messages
        - Summarise the middle section
        """
        before_tokens = count_message_tokens(messages)

        if len(messages) < 6:
            return messages   # too short to compress meaningfully

        system_msg = messages[0]   # always keep
        first_user = messages[1]   # always keep
        recent     = messages[-4:] # always keep last 4
        middle     = messages[2:-4]

        if not middle:
            return messages

        middle_text = "\n".join(
            f"{type(m).__name__}: {getattr(m, 'content', '')[:300]}"
            for m in middle
        )
        summary_resp = self._compressor.invoke([
            HumanMessage(content=(
                f"Task goal: {goal[:200]}\n\n"
                "Summarise these intermediate steps, preserving key facts and results:\n\n"
                + middle_text
            ))
        ])
        summary = (
            summary_resp.content
            if hasattr(summary_resp, "content")
            else str(summary_resp)
        )

        compressed = [
            system_msg,
            first_user,
            SystemMessage(content=f"[COMPRESSED HISTORY — iterations {2} to {len(messages)-4}]: {summary}"),
            *recent,
        ]

        after_tokens = count_message_tokens(compressed)
        self._logger.context_compressed(
            iteration=iteration,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
        )
        return compressed
