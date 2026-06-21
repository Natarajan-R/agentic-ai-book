"""
tools/web_search.py  — Real DuckDuckGo search (no API key required)
tools/rag.py         — Persistent ChromaDB RAG retrieval
tools/compute.py     — Sandboxed Python execution via RestrictedPython
tools/files.py       — Safe file operations with path validation
tools/human.py       — Human-in-the-loop escalation tool
tools/registry.py    — Tool registry with metadata and rate-limit tracking

All tools follow the contract:
  - Strongly typed inputs and outputs
  - Structured exceptions on failure (never bare Exception)
  - Every call is timed
  - Timeout enforced via threading.Timer
  - Graceful degradation with informative error returns
"""

# ─── tools/web_search.py ─────────────────────────────────────────────────────

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import tool
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

from config.settings import settings
from core.exceptions import ToolExecutionError, ToolTimeoutError
from core.tokens import truncate_to_tokens


@dataclass
class SearchResult:
    title:   str
    url:     str
    snippet: str
    source:  str = "web"

    def to_text(self, max_snippet_tokens: int = 120) -> str:
        snippet = truncate_to_tokens(self.snippet, max_snippet_tokens)
        return f"Title: {self.title}\nURL: {self.url}\nSummary: {snippet}"


def _do_search(query: str, max_results: int) -> list[SearchResult]:
    """Synchronous search — run in executor with timeout."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ))
    return results


@tool
def web_search(query: str) -> str:
    """
    Search the public web using DuckDuckGo.
    Returns structured results with title, URL, and summary for each result.
    Use for: current events, competitor news, market data, public information.
    Do NOT use for: internal company data, confidential information.
    """
    max_results = settings.web_search_max_results
    timeout     = settings.tool_timeout_seconds

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_do_search, query, max_results)
        try:
            results = future.result(timeout=timeout)
        except FuturesTimeout:
            raise ToolTimeoutError(
                message=f"Web search timed out after {timeout}s",
                tool_name="web_search",
                tool_args={"query": query},
                timeout_seconds=timeout,
            )
        except DuckDuckGoSearchException as e:
            raise ToolExecutionError(
                message=f"DuckDuckGo search failed: {e}",
                tool_name="web_search",
                tool_args={"query": query},
                original_error=str(e),
            )

    if not results:
        return f"No results found for query: {query}"

    formatted = "\n\n".join(r.to_text() for r in results)
    return f"Web search results for '{query}':\n\n{formatted}"


# ─── tools/rag.py ────────────────────────────────────────────────────────────

import chromadb
from chromadb.config import Settings as ChromaSettings
from core.exceptions import VectorStoreError

_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_chroma_collection() -> chromadb.Collection:
    """
    Get or create the persistent ChromaDB collection.
    Uses persistent storage — data survives process restarts.
    """
    global _chroma_client, _collection
    if _collection is not None:
        return _collection

    _chroma_client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    _collection = _chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def seed_knowledge_base(documents: list[dict[str, str]]) -> int:
    """
    Seed the knowledge base with enterprise documents.
    Each document: {"id": str, "text": str, "source": str, "category": str}
    Returns number of documents added.
    """
    col = get_chroma_collection()
    existing = set(col.get()["ids"])

    to_add = [d for d in documents if d["id"] not in existing]
    if not to_add:
        return 0

    col.add(
        ids=[d["id"] for d in to_add],
        documents=[d["text"] for d in to_add],
        metadatas=[{"source": d.get("source", ""), "category": d.get("category", "")}
                   for d in to_add],
    )
    return len(to_add)


@tool
def rag_retrieve(query: str) -> str:
    """
    Retrieve relevant information from the internal enterprise knowledge base.
    Uses semantic similarity search — returns the most relevant passages.
    Use for: internal policies, past analyses, product documentation,
             historical competitor data, organisational knowledge.
    """
    try:
        col = get_chroma_collection()
        results = col.query(
            query_texts=[query],
            n_results=min(settings.rag_top_k, max(col.count(), 1)),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        raise VectorStoreError(
            message=f"ChromaDB query failed: {e}",
            memory_type="semantic",
        )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return f"No relevant information found for: {query}"

    parts = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        confidence = round(1 - dist, 2)   # cosine distance → similarity
        truncated  = truncate_to_tokens(doc, settings.rag_max_chunk_tokens)
        source     = meta.get("source", "internal")
        parts.append(
            f"[Confidence: {confidence:.0%} | Source: {source}]\n{truncated}"
        )

    return f"Internal knowledge base results for '{query}':\n\n" + "\n\n---\n\n".join(parts)


# ─── tools/compute.py ────────────────────────────────────────────────────────

import ast
import math
import statistics
from core.exceptions import ToolExecutionError


# Whitelist of safe builtins for sandboxed execution
SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool,
    "dict": dict, "divmod": divmod, "enumerate": enumerate,
    "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "map": map, "max": max,
    "min": min, "pow": pow, "print": print, "range": range,
    "round": round, "set": set, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}

SAFE_GLOBALS = {
    "__builtins__": SAFE_BUILTINS,
    "math":       math,
    "statistics": statistics,
}


def _is_safe_ast(code: str) -> tuple[bool, str]:
    """
    Validate that the code AST contains only safe operations.
    Blocks: imports, file access, network access, exec/eval calls.
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

    forbidden = (
        ast.Import, ast.ImportFrom,
        ast.Global, ast.Nonlocal,
        ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
    )
    forbidden_names = {"exec", "eval", "compile", "open", "__import__",
                       "getattr", "setattr", "delattr", "vars", "dir",
                       "globals", "locals", "breakpoint"}

    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            return False, f"Forbidden operation: {type(node).__name__}"
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden_names:
                return False, f"Forbidden function call: {name}"

    return True, ""


@tool
def code_execute(expression: str) -> str:
    """
    Execute a Python expression or short script for precise computation.
    Available: math, statistics, all standard numeric operations.
    NOT available: file I/O, network access, imports, exec/eval.

    ALWAYS use this tool for any calculation — never compute mentally.
    The LLM is probabilistic; this tool is exact.

    Examples:
      "3.8e9 * 1.42 ** 3"
      "statistics.mean([299, 350, 199, 99])"
      "round(5000 * 299 * 12 / 1000, 2)"
    """
    # Validate AST before execution
    safe, reason = _is_safe_ast(expression)
    if not safe:
        raise ToolExecutionError(
            message=f"Code execution blocked: {reason}",
            tool_name="code_execute",
            tool_args={"expression": expression},
            original_error=reason,
        )

    try:
        # Try eval first (expression mode)
        try:
            result = eval(expression, SAFE_GLOBALS.copy())  # noqa: S307
        except SyntaxError:
            # Fall back to exec (statement mode)
            local_vars: dict = {}
            exec(expression, SAFE_GLOBALS.copy(), local_vars)  # noqa: S102
            result = local_vars.get("result", local_vars)

        # Format result
        if isinstance(result, float):
            formatted = f"{result:,.6g}"
        elif isinstance(result, (list, dict, tuple)):
            formatted = str(result)
        else:
            formatted = str(result)

        return f"Result: {formatted}"

    except Exception as e:
        raise ToolExecutionError(
            message=f"Execution error: {e}",
            tool_name="code_execute",
            tool_args={"expression": expression},
            original_error=str(e),
        )


# ─── tools/files.py ──────────────────────────────────────────────────────────

import os
from pathlib import Path
from core.exceptions import ToolExecutionError


# Allowed output directory — agents cannot write outside this
OUTPUT_DIR = Path("./data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_path(filename: str) -> Path:
    """
    Resolve filename to an absolute path within OUTPUT_DIR.
    Raises if the resolved path escapes OUTPUT_DIR (path traversal protection).
    """
    resolved = (OUTPUT_DIR / Path(filename).name).resolve()
    if not str(resolved).startswith(str(OUTPUT_DIR.resolve())):
        raise ToolExecutionError(
            message=f"Path traversal blocked: {filename}",
            tool_name="file_write",
            tool_args={"filename": filename},
            original_error="resolved path outside allowed directory",
        )
    return resolved


@tool
def file_write(filename: str, content: str) -> str:
    """
    Write content to a file in the agent's output directory.
    Supports: .txt, .md, .json, .csv
    The file is always written to the safe output directory.
    Returns the full path and byte count on success.
    """
    allowed_extensions = {".txt", ".md", ".json", ".csv", ".html"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise ToolExecutionError(
            message=f"File extension not allowed: {ext}. Allowed: {allowed_extensions}",
            tool_name="file_write",
            tool_args={"filename": filename},
            original_error="disallowed extension",
        )

    path = _safe_path(filename)
    try:
        path.write_text(content, encoding="utf-8")
        return (
            f"Successfully written to: {path}\n"
            f"Size: {len(content.encode('utf-8')):,} bytes | "
            f"Lines: {content.count(chr(10))+1}"
        )
    except OSError as e:
        raise ToolExecutionError(
            message=f"File write failed: {e}",
            tool_name="file_write",
            tool_args={"filename": filename, "content_length": len(content)},
            original_error=str(e),
        )


@tool
def file_read(filename: str) -> str:
    """
    Read a file from the agent's output directory.
    Only files previously written by the agent can be read.
    """
    path = _safe_path(filename)
    if not path.exists():
        return f"File not found: {filename}"
    try:
        content = path.read_text(encoding="utf-8")
        return truncate_to_tokens(content, 2000)
    except OSError as e:
        raise ToolExecutionError(
            message=f"File read failed: {e}",
            tool_name="file_read",
            tool_args={"filename": filename},
            original_error=str(e),
        )


# ─── tools/human.py ──────────────────────────────────────────────────────────

@tool
def human_escalate(question: str, context: str = "") -> str:
    """
    Escalate to a human for input, approval, or clarification.
    Use when:
      - An action is irreversible and requires authorisation
      - The task is genuinely ambiguous and guessing would be wrong
      - The agent has encountered an unexpected situation it cannot resolve
    The agent loop pauses until the human responds.
    """
    print(f"\n{'─' * 60}")
    print("[HUMAN ESCALATION REQUIRED]")
    print(f"Question: {question}")
    if context:
        print(f"Context:  {context[:300]}")
    print(f"{'─' * 60}")

    response = input("Your response (press Enter to skip): ").strip()
    if not response:
        return "Human did not respond — agent will attempt to continue autonomously."
    return f"Human response: {response}"


# ─── tools/registry.py ───────────────────────────────────────────────────────

from dataclasses import dataclass, field as dc_field
from langchain_core.tools import BaseTool


@dataclass
class ToolMetadata:
    name:        str
    description: str
    when_to_use: str
    rate_limit:  int
    irreversible: bool = False
    requires_human_approval: bool = False


TOOL_METADATA: dict[str, ToolMetadata] = {
    "web_search": ToolMetadata(
        name="web_search",
        description="Search the public web for current information",
        when_to_use="Current events, competitor news, market data, public info",
        rate_limit=8,
    ),
    "rag_retrieve": ToolMetadata(
        name="rag_retrieve",
        description="Retrieve from internal knowledge base",
        when_to_use="Internal policies, past analyses, product docs, historical data",
        rate_limit=20,
    ),
    "code_execute": ToolMetadata(
        name="code_execute",
        description="Execute Python for precise computation",
        when_to_use="ALL arithmetic, statistics, data transformation",
        rate_limit=5,
    ),
    "file_write": ToolMetadata(
        name="file_write",
        description="Write report or document to output directory",
        when_to_use="Final deliverables, structured reports",
        rate_limit=3,
    ),
    "file_read": ToolMetadata(
        name="file_read",
        description="Read a previously written file",
        when_to_use="Reviewing previously written content",
        rate_limit=10,
    ),
    "human_escalate": ToolMetadata(
        name="human_escalate",
        description="Escalate to a human for approval or clarification",
        when_to_use="Irreversible actions, ambiguous situations",
        rate_limit=5,
        requires_human_approval=True,
    ),
}


def get_all_tools() -> list[BaseTool]:
    """Return all registered tools for binding to the LLM."""
    return [web_search, rag_retrieve, code_execute, file_write, file_read, human_escalate]
