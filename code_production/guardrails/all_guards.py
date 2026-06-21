"""
guardrails/
===========
All five guardrail layers with production-grade implementations.

Layer 1 — Input:     Injection detection + scope + length validation
Layer 2 — Reasoning: System prompt constraints (see agent/reference.py)
Layer 3 — Tool pre:  Rate limiting + irreversible action confirmation
Layer 3 — Tool post: PII redaction + injection in retrieved content
Layer 4 — Output:    Completeness + PII + policy compliance
Layer 5 — Observability: Logging (see core/logging.py)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from core.exceptions import (
    GuardrailError, InjectionDetectedError, OutOfScopeError,
    RateLimitError, OutputValidationError,
)
from core.logging import AgentLogger


# ─── INJECTION PATTERNS ──────────────────────────────────────────────────────
# Ordered by severity — most dangerous first

INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Direct system override attempts. The modifier group is repeatable so
    # "ignore all previous instructions" and "disregard your previous
    # instructions" both match.
    (r"ignore\s+(all\s+|previous\s+|your\s+|any\s+)*(instructions|constraints|rules|restrictions|guidelines|limitations|limits|scope)", "system_override"),
    (r"disregard\s+(all\s+|previous\s+|your\s+|any\s+)*(instructions|constraints|rules|restrictions|guidelines|limitations|limits|scope)", "system_override"),
    (r"forget\s+(everything|all|your instructions)", "system_override"),
    (r"override\s+(your|all|previous)\s+", "system_override"),

    # Role redefinition. "you are now ..." in a user request to a scoped
    # enterprise agent is an override attempt regardless of the following word.
    (r"you\s+are\s+now\b", "role_redefine"),
    (r"\b(unrestricted|unlimited|no\s+restrictions|without\s+restrictions)\b", "role_redefine"),
    (r"act\s+as\s+(if\s+)?(you\s+are|a|an)\s+", "role_redefine"),
    (r"pretend\s+(to\s+be|you\s+are)\s+", "role_redefine"),
    (r"your\s+new\s+(role|task|instruction|system|prompt)\s+is", "role_redefine"),

    # System prompt extraction
    (r"(reveal|show|print|output|repeat)\s+(your\s+)?(system\s+prompt|instructions)", "extraction"),
    (r"what\s+(are\s+)?your\s+(system\s+)?(prompt|instructions)", "extraction"),

    # Jailbreak patterns
    (r"(DAN|jailbreak|developer\s+mode|god\s+mode)", "jailbreak"),
    (r"hypothetically\s+(speaking\s+)?if\s+you\s+(had\s+no|were\s+without)", "jailbreak"),
]

_COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), category)
    for pattern, category in INJECTION_PATTERNS
]

# PII patterns for redaction
PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                      "ssn"),
    (re.compile(r"\b(?:\d{4}[\s-]?){3}\d{4}\b"),               "credit_card"),
    (re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),       "password"),
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "email"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "phone"),
    (re.compile(r"\b(?:API[-_]?KEY|api[-_]?key|sk-[A-Za-z0-9]{32,})\b"), "api_key"),
]


# ─── LAYER 1: INPUT GUARDRAIL ────────────────────────────────────────────────

@dataclass
class InputValidationResult:
    passed:         bool
    sanitised_text: str
    violation_type: str = ""
    violation_detail: str = ""


def validate_input(
    text: str,
    task_id: str,
    logger: AgentLogger,
) -> InputValidationResult:
    """
    Layer 1: Validate and sanitise incoming user request.

    Checks (in order):
      1. Length bounds
      2. Injection pattern detection (regex-based)
      3. Scope validation
      4. Basic sanitisation (strip null bytes, normalise whitespace)

    Returns InputValidationResult — never raises.
    Caller decides whether to proceed based on .passed.
    """

    # 1. Length check
    if len(text) > settings.max_input_length:
        logger.input_blocked(
            reason=f"Input too long: {len(text)} chars (max {settings.max_input_length})",
            violation="length",
        )
        return InputValidationResult(
            passed=False,
            sanitised_text=text,
            violation_type="length",
            violation_detail=f"{len(text)} chars exceeds {settings.max_input_length} limit",
        )

    # 2. Injection detection
    text_normalised = " ".join(text.lower().split())  # normalise whitespace
    for pattern, category in _COMPILED_PATTERNS:
        match = pattern.search(text_normalised)
        if match:
            logger.input_blocked(
                reason=f"Injection pattern detected: {category}",
                violation="injection",
            )
            return InputValidationResult(
                passed=False,
                sanitised_text=text,
                violation_type="injection",
                violation_detail=f"Category: {category}, match: {match.group()[:50]}",
            )

    # 3. Scope validation
    text_lower = text.lower()
    scope_matched = any(kw in text_lower for kw in settings.agent_scope_keywords)
    if not scope_matched:
        logger.input_blocked(
            reason="Request outside authorised scope",
            violation="scope",
        )
        return InputValidationResult(
            passed=False,
            sanitised_text=text,
            violation_type="scope",
            violation_detail=(
                "No scope keywords matched. This agent handles: "
                + ", ".join(sorted(settings.agent_scope_keywords)[:8])
            ),
        )

    # 4. Basic sanitisation
    sanitised = text.replace("\x00", "").strip()   # strip null bytes

    logger.input_passed(input_length=len(sanitised))
    return InputValidationResult(passed=True, sanitised_text=sanitised)


# ─── LAYER 3a: PRE-EXECUTION TOOL GUARDRAIL ──────────────────────────────────

@dataclass
class ToolPreCheckResult:
    allowed:     bool
    reason:      str = ""
    needs_human: bool = False


def check_tool_pre(
    tool_name: str,
    tool_args: dict[str, Any],
    tool_counts: dict[str, int],
    task_id: str,
    iteration: int,
    logger: AgentLogger,
) -> ToolPreCheckResult:
    """
    Layer 3a: Validate a proposed tool call before execution.

    Checks:
      1. Tool exists in metadata registry
      2. Rate limit not exceeded
      3. Args do not contain injection patterns
      4. Irreversible actions flagged for human confirmation
    """
    from tools.all_tools import TOOL_METADATA

    # 1. Tool exists
    if tool_name not in TOOL_METADATA:
        logger.tool_blocked(iteration, tool_name, "tool not in registry")
        return ToolPreCheckResult(allowed=False, reason=f"Unknown tool: {tool_name}")

    meta = TOOL_METADATA[tool_name]

    # 2. Rate limit
    current_count = tool_counts.get(tool_name, 0)
    if current_count >= meta.rate_limit:
        logger.tool_blocked(
            iteration, tool_name,
            f"rate limit {current_count}/{meta.rate_limit}",
        )
        return ToolPreCheckResult(
            allowed=False,
            reason=f"Rate limit reached: {meta.name} ({current_count}/{meta.rate_limit} calls)",
        )

    # 3. Injection in args
    for key, value in tool_args.items():
        if not isinstance(value, str):
            continue
        text_normalised = " ".join(value.lower().split())
        for pattern, category in _COMPILED_PATTERNS[:4]:   # check most critical patterns
            if pattern.search(text_normalised):
                logger.tool_blocked(
                    iteration, tool_name,
                    f"injection in arg '{key}': {category}",
                )
                return ToolPreCheckResult(
                    allowed=False,
                    reason=f"Argument '{key}' contains injection pattern: {category}",
                )

    # 4. Irreversible action
    if settings.is_irreversible(tool_name) or meta.requires_human_approval:
        return ToolPreCheckResult(
            allowed=True,
            needs_human=True,
            reason=f"Tool '{tool_name}' requires human confirmation",
        )

    return ToolPreCheckResult(allowed=True)


# ─── LAYER 3b: POST-EXECUTION TOOL GUARDRAIL ─────────────────────────────────

@dataclass
class ToolPostResult:
    content:     str
    was_redacted: bool = False
    was_filtered: bool = False
    patterns_found: list[str] = None

    def __post_init__(self):
        if self.patterns_found is None:
            self.patterns_found = []


def check_tool_post(
    tool_name: str,
    raw_result: str,
    task_id: str,
    iteration: int,
    logger: AgentLogger,
) -> ToolPostResult:
    """
    Layer 3b: Inspect tool result before injecting into context window.

    Checks:
      1. PII redaction
      2. Injection in retrieved content (indirect injection defence)
    """
    content = raw_result
    patterns_found: list[str] = []

    # 1. PII redaction
    if settings.pii_redaction_enabled:
        for pattern, pii_type in PII_PATTERNS:
            new_content = pattern.sub(f"[REDACTED:{pii_type.upper()}]", content)
            if new_content != content:
                patterns_found.append(pii_type)
                content = new_content

        if patterns_found:
            logger.pii_redacted("tool_post", iteration, patterns_found)

    # 2. Injection in retrieved content (indirect injection defence).
    # Check ALL injection patterns — including role-redefine and jailbreak
    # (e.g. "developer mode") which commonly appear in poisoned tool results.
    content_normalised = " ".join(content.lower().split())
    for pattern, category in _COMPILED_PATTERNS:
        if pattern.search(content_normalised):
            logger.error(
                iteration, "tool_post",
                "IndirectInjectionDetected",
                f"Injection pattern '{category}' in result from {tool_name}",
                recoverable=True,
            )
            # Replace the suspicious section rather than dropping the whole
            # result. The pattern is already compiled with re.IGNORECASE, so
            # flags must NOT be passed to .sub() here (that raises TypeError).
            content = pattern.sub("[FILTERED:INJECTION_ATTEMPT]", content)
            return ToolPostResult(
                content=content,
                was_redacted=bool(patterns_found),
                was_filtered=True,
                patterns_found=patterns_found,
            )

    return ToolPostResult(
        content=content,
        was_redacted=bool(patterns_found),
        patterns_found=patterns_found,
    )


# ─── LAYER 4: OUTPUT GUARDRAIL ────────────────────────────────────────────────

@dataclass
class OutputValidationResult:
    passed:          bool
    validated_output: str
    issues:          list[str] = None
    confidence:      float = 1.0

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


def validate_output(
    output: str,
    goal: str,
    task_id: str,
    iteration: int,
    logger: AgentLogger,
) -> OutputValidationResult:
    """
    Layer 4: Validate the final output before delivery.

    Checks:
      1. Minimum length / substance
      2. PII redaction
      3. Completeness check via LLM
      4. Confidence assessment
    """
    issues: list[str] = []

    # 1. Minimum substance
    if len(output.strip()) < 80:
        issues.append(f"Output too brief: {len(output)} chars")

    # 2. PII redaction
    content = output
    pii_found: list[str] = []
    for pattern, pii_type in PII_PATTERNS:
        new_content = pattern.sub(f"[REDACTED:{pii_type.upper()}]", content)
        if new_content != content:
            pii_found.append(pii_type)
            content = new_content

    if pii_found:
        logger.pii_redacted("output", iteration, pii_found)

    # 3. Completeness check via LLM (use secondary model — cheaper)
    llm = ChatOllama(model=settings.model_secondary, temperature=0)
    completeness_resp = llm.invoke([
        SystemMessage(content=(
            "You are an output quality checker. "
            "Assess whether the output adequately addresses the goal. "
            "Respond with JSON only: "
            '{"complete": true/false, "confidence": 0.0-1.0, "issues": ["list of issues"]}'
        )),
        HumanMessage(content=(
            f"Goal: {goal[:300]}\n\n"
            f"Output (first 500 chars): {content[:500]}\n\n"
            "Is the output complete and relevant?"
        )),
    ])

    try:
        import json
        assessment = json.loads(completeness_resp.content)
        complete   = assessment.get("complete", True)
        confidence = float(assessment.get("confidence", 0.8))
        llm_issues = assessment.get("issues", [])
    except Exception:
        complete   = True
        confidence = 0.7
        llm_issues = []

    if not complete:
        issues.extend(llm_issues)

    passed = len(issues) == 0 and complete
    logger.output_validated(iteration, len(content), passed)

    return OutputValidationResult(
        passed=passed,
        validated_output=content,
        issues=issues,
        confidence=confidence,
    )
