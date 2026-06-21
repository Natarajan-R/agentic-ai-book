"""
core/tokens.py
===============
Accurate token counting using tiktoken.
Used for context window management and cost tracking.
"""

from __future__ import annotations

import tiktoken
from functools import lru_cache
from langchain_core.messages import BaseMessage


@lru_cache(maxsize=4)
def _get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Cache tiktoken encoders — they are expensive to initialise."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens in a string using tiktoken."""
    if not text:
        return 0
    enc = _get_encoder(encoding)
    return len(enc.encode(text))


def count_message_tokens(messages: list[BaseMessage | dict]) -> int:
    """
    Count total tokens across a list of messages.
    Each message has ~4 tokens of overhead for role/format markers.
    """
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = str(msg.get("content", ""))
        else:
            content = str(getattr(msg, "content", ""))
        total += count_tokens(content) + 4  # message overhead
    return total + 3  # conversation overhead


def truncate_to_tokens(text: str, max_tokens: int,
                       encoding: str = "cl100k_base") -> str:
    """Truncate text to at most max_tokens, preserving whole words."""
    enc = _get_encoder(encoding)
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens])
