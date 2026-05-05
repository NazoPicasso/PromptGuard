"""
PromptGuard — Guard Layer
Rule-based guard wrapping any LLM pipeline: scores prompts, assigns risk tier, blocks on threshold.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from evaluation.runner import GUARD_BLOCK_PREFIX

# Terms often appearing in malicious USER prompts (not the same as output leak signals)
_PROMPT_SUSPICIOUS_TERMS = [
    "ignore all previous",
    "system prompt",
    "developer mode",
    "dan ",
    "unrestricted",
    "reveal your",
    "context window",
    "override",
    "sudo",
    "jailbreak",
    "end of instructions",
    "new directive",
    "base64",
    "disregard",
]

# Critical: almost always injection attempts
CRITICAL_PATTERNS = [
    r"first\s+.{0,40}characters\s+of\s+your\s+system",
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"(forget|disregard)\s+(everything|all|your)\s+(above|before|previous)\s+instructions?",
    r"(reveal|print|output|show|display|dump)\s+(your\s+)?(full\s+)?(system\s+prompt|instructions?|context)",
    r"(you are now|act as|pretend (you are|to be))\s+.*(unrestricted|dan|without restrictions)",
    r"(sudo|admin|developer|debug)\s*(mode|override|activated|access)",
    r"###\s*(end\s+of\s+instructions?|system\s+override)",
    r"\[ADMIN\]|\[SYSTEM OVERRIDE\]|\[DEBUG\]|\[OVERRIDE\]",
    r"new\s+(directive|instruction|task|objective)\s*:\s*",
    r"(bypass|override|disable|remove)\s+(safety|filter|restriction|guideline)",
]

HIGH_PATTERNS = [
    r"(confidential|secret|hidden)\s+(data|information|memory|variable|api\s*key)",
    r"(print|list|show|dump)\s+(all\s+)?(text|variables|secrets?|configuration)",
    r"(print|show)\s+all\s+text\s+(that\s+)?appears\s+before",
    r"(what|which)\s+.{0,40}(confidential|api\s*keys?|user\s+data|given\s+access)",
    r"\[Tool result\].*(SYSTEM\s*:|new\s+directive|ignore\s+all)",
    r"\[Retrieved document\].*(disregard|ignore\s+(all\s+)?previous)",
    r"(token|jwt|password)\s*[:=]\s*[`'\"]",
    r"base64[^\n]{0,80}(instruction|prompt|system)",
]

MEDIUM_PATTERNS = [
    r"(hypothetically|in\s+a\s+fictional|for\s+educational\s+purposes)",
    r"(base64|encode|decode)\s+.{0,80}(instruction|prompt|system)",
    r"(new\s+role|your\s+new\s+role|you\s+are\s+now\s+a)",
    r"(translate|convert)\s+.{0,120}(then\s+)?(follow|obey|execute|reveal)",
    r"roleplay|let'\s*s\s+roleplay",
    r"\\u[0-9a-f]{4}",  # unicode escapes in prompt
]

SOFT_PATTERNS = [
    r"\n{6,}",  # excessive newlines (smuggling)
]


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]


def _risk_tier_from_score(score: int) -> str:
    if score <= 2:
        return "low"
    if score <= 6:
        return "medium"
    return "high"


class PromptGuard:
    """
    Wraps a pipeline: inspects user prompts, blocks likely injections before the model runs.

    ``sensitivity`` controls which pattern sets are active and the score threshold for blocking.
    """

    def __init__(
        self,
        pipeline: Callable[[str], str],
        sensitivity: str = "medium",
        log_blocked: bool = True,
        custom_patterns: Optional[list[str]] = None,
    ):
        self.pipeline = pipeline
        self.sensitivity = sensitivity.lower()
        if self.sensitivity not in ("low", "medium", "high"):
            self.sensitivity = "medium"
        self.log_blocked = log_blocked
        self.blocked_log = []
        self.total_requests = 0
        self.blocked_count = 0

        self._critical = _compile(CRITICAL_PATTERNS)
        self._high = _compile(HIGH_PATTERNS)
        self._medium = _compile(MEDIUM_PATTERNS)
        self._soft = _compile(SOFT_PATTERNS)
        self._custom = _compile(custom_patterns) if custom_patterns else []

        # Base weights per match tier
        self._tier_weights = {
            "critical": 5,
            "high": 3,
            "medium": 2,
            "soft": 1,
        }

    def _collect_matches(self, prompt: str) -> dict:
        """Return matched pattern strings grouped by tier."""
        matched: dict[str, list[str]] = {"critical": [], "high": [], "medium": [], "soft": []}
        for pat in self._critical:
            if pat.search(prompt):
                matched["critical"].append(pat.pattern[:80])
        for pat in self._high:
            if pat.search(prompt):
                matched["high"].append(pat.pattern[:80])
        for pat in self._medium:
            if pat.search(prompt):
                matched["medium"].append(pat.pattern[:80])
        for pat in self._soft:
            if pat.search(prompt):
                matched["soft"].append(pat.pattern[:80])
        for pat in self._custom:
            if pat.search(prompt):
                matched["critical"].append(f"custom:{pat.pattern[:60]}")
        return matched

    def _score_prompt(self, prompt: str) -> dict:
        """Compute numeric risk score, tier label, and whether to block."""
        prompt_lower = prompt.lower()
        matches = self._collect_matches(prompt)

        score = 0
        for tier, names in matches.items():
            w = self._tier_weights.get(tier, 1)
            score += len(names) * w

        keyword_hits = sum(1 for kw in _PROMPT_SUSPICIOUS_TERMS if kw in prompt_lower)
        score += min(keyword_hits, 6)

        if prompt.count("\n") > 8 and any(
            k in prompt_lower for k in ("ignore", "system", "instruction", "override")
        ):
            score += 2

        risk_tier = _risk_tier_from_score(score)

        # Higher threshold for low sensitivity → fewer blocks; high sensitivity → more aggressive
        thresholds = {"low": 12, "medium": 7, "high": 5}
        threshold = thresholds[self.sensitivity]

        has_strong_tier = bool(matches["critical"] or matches["high"] or matches["medium"])

        if self.sensitivity == "low":
            # Ignore newline-only soft signal unless other tiers fired
            if not has_strong_tier and matches["soft"] and keyword_hits <= 1:
                should_block = False
            else:
                should_block = score >= threshold or (
                    bool(matches["critical"]) and score >= 9
                )
        elif self.sensitivity == "medium":
            should_block = score >= threshold or (
                bool(matches["critical"]) and score >= 5
            )
        else:
            should_block = score >= threshold or bool(matches["critical"])

        return {
            "risk_score": score,
            "risk_tier": risk_tier,
            "is_injection": should_block,
            "matched_by_tier": {k: v for k, v in matches.items() if v},
            "keyword_hits": keyword_hits,
            "threshold": threshold,
            "sensitivity": self.sensitivity,
        }

    def run(self, prompt: str) -> dict:
        """Run guard then optionally call the underlying pipeline."""
        self.total_requests += 1
        assessment = self._score_prompt(prompt)

        if assessment["is_injection"]:
            self.blocked_count += 1
            msg = (
                f"{GUARD_BLOCK_PREFIX} Potential prompt injection detected "
                f"(risk={assessment['risk_tier']}, score={assessment['risk_score']}). "
                "This incident has been logged."
            )
            blocked_entry = {
                "prompt_preview": prompt[:120] + "..." if len(prompt) > 120 else prompt,
                "risk_score": assessment["risk_score"],
                "risk_tier": assessment["risk_tier"],
                "matched": assessment["matched_by_tier"],
            }
            if self.log_blocked:
                self.blocked_log.append(blocked_entry)
                print(
                    f"[Guard] BLOCKED — tier={assessment['risk_tier']} score={assessment['risk_score']} "
                    f"threshold={assessment['threshold']}"
                )

            return {
                "response": msg,
                "blocked": True,
                "risk_assessment": assessment,
                "model_response": None,
            }

        model_response = self.pipeline(prompt)
        return {
            "response": model_response,
            "blocked": False,
            "risk_assessment": assessment,
            "model_response": model_response,
        }

    def get_stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "blocked": self.blocked_count,
            "passed": self.total_requests - self.blocked_count,
            "block_rate": round(self.blocked_count / max(self.total_requests, 1), 4),
            "sensitivity": self.sensitivity,
        }


def build_guarded_pipeline(
    pipeline: Callable[[str], str],
    sensitivity: str = "medium",
) -> Callable[[str], str]:
    """Return a simple ``prompt -> str`` wrapper for use with ``evaluate_pipeline``."""
    guard = PromptGuard(pipeline, sensitivity=sensitivity, log_blocked=False)

    def guarded(prompt: str) -> str:
        return guard.run(prompt)["response"]

    guarded.__name__ = f"guarded_{getattr(pipeline, '__name__', 'pipeline')}_{sensitivity}"
    return guarded
