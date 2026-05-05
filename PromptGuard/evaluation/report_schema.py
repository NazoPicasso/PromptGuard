"""
Audit-oriented JSON report helpers and schema version for PromptGuard.

Legacy consumers expect top-level keys: ``report_id``, ``generated_at``, ``pipeline``,
``baseline``, optional ``with_guard``, ``improvement``. v2 adds structured audit metadata
without removing those fields.
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA_VERSION = "2.0"

# Package tool name for audit trails
TOOL_NAME = "promptguard"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_audit_block(
    report_id: str,
    generated_at_utc: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Minimal audit record suitable for compliance exports."""
    block: dict[str, Any] = {
        "report_id": report_id,
        "generated_at_utc": generated_at_utc or utc_now_iso(),
        "tool": TOOL_NAME,
        "schema_version": SCHEMA_VERSION,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    if extra:
        block["notes"] = extra
    return block


def build_evaluation_config(
    sensitivity: str,
    attack_count: int,
    semantic_model_id: Optional[str],
    mutation_enabled: bool,
    benchmark_targets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Configuration snapshot embedded in reports for reproducibility."""
    cfg: dict[str, Any] = {
        "guard_sensitivity": sensitivity,
        "attack_case_count": attack_count,
        "mutation_enabled": mutation_enabled,
        "semantic_embedding_model": semantic_model_id or "unavailable",
    }
    if benchmark_targets:
        cfg["benchmark_targets"] = benchmark_targets
    return cfg


def attach_schema_wrappers(
    report: dict[str, Any],
    audit: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """Mutate report dict in place with v2 audit fields; preserves legacy keys."""
    report["schema_version"] = SCHEMA_VERSION
    report["audit"] = audit
    report["evaluation_config"] = evaluation_config
    if audit.get("generated_at_utc") and not report.get("generated_at"):
        report["generated_at"] = audit["generated_at_utc"]
    return report
