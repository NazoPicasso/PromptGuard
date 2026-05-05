"""
Multi-model benchmarking: run the same attack suite against several pipeline targets.

Results are merged into a single report via ``merge_benchmark_into_report`` in ``runner``.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from detector.guard import build_guarded_pipeline
from evaluation.runner import compute_metrics, evaluate_pipeline


def compute_improvement(baseline_metrics: dict, guard_metrics: dict) -> dict:
    b_asr = baseline_metrics.get("asr", 0)
    g_asr = guard_metrics.get("asr", 0)
    reduction = round((b_asr - g_asr) / max(b_asr, 0.001) * 100, 1)
    return {
        "asr_reduction_percent": reduction,
        "baseline_asr": b_asr,
        "guarded_asr": g_asr,
    }


def run_benchmark_for_target(
    label: str,
    pipeline: Callable[[str], str],
    attacks: list[dict],
    sensitivity: str,
    verbose: bool = False,
) -> dict:
    """
    Evaluate baseline + guarded for one callable pipeline.

    Returns a dict with ``target``, ``baseline``, ``with_guard``, ``improvement``.
    """
    pipeline_id = getattr(pipeline, "__name__", label)
    baseline_results = evaluate_pipeline(pipeline, attacks, verbose=verbose)
    guarded = build_guarded_pipeline(pipeline, sensitivity=sensitivity)
    guard_results = evaluate_pipeline(guarded, attacks, verbose=verbose)
    bm = compute_metrics(baseline_results)
    gm = compute_metrics(guard_results)
    return {
        "target": {
            "label": label,
            "model_id": pipeline_id,
        },
        "baseline": {"metrics": bm, "results": baseline_results},
        "with_guard": {"metrics": gm, "results": guard_results},
        "improvement": compute_improvement(bm, gm),
    }


def parse_benchmark_spec(spec: str) -> tuple[str, Optional[str]]:
    """
    Parse entries like ``mock``, ``ollama``, ``ollama:llama3``, ``openai``, ``openai:gpt-4o-mini``.

    Returns ``(kind, model_override)``.
    """
    if ":" in spec:
        kind, mid = spec.split(":", 1)
        return kind.strip().lower(), mid.strip() or None
    return spec.strip().lower(), None


def build_pipeline_callable(
    kind: str,
    model_override: Optional[str],
    system_prompt: str,
    ollama_base: str,
    mock_vulnerability: float,
):
    """
    Factory returning ``(label, pipeline)`` or raises if dependencies missing.

    * ``mock`` — always available
    * ``openai`` — requires ``openai`` package + ``OPENAI_API_KEY``
    * ``ollama`` — requires local Ollama and ``requests``; model default ``llama3`` or env ``OLLAMA_MODEL``
    """
    from pipelines import openai_pipeline as op

    if kind == "mock":
        pl = op.build_mock_pipeline(vulnerability_level=mock_vulnerability)
        return "mock", pl

    if kind == "openai":
        mid = model_override or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set for OpenAI benchmark target")
        pl = op.build_openai_pipeline(system_prompt=system_prompt, model=mid)
        return f"openai:{mid}", pl

    if kind == "ollama":
        mid = model_override or os.getenv("OLLAMA_MODEL", "llama3")
        pl = op.build_ollama_pipeline(
            system_prompt=system_prompt,
            model=mid,
            base_url=ollama_base,
        )
        return f"ollama:{mid}", pl

    raise ValueError(f"Unknown benchmark target kind: {kind}")


def run_benchmark_suite(
    spec_strings: list[str],
    attacks: list[dict],
    sensitivity: str,
    system_prompt: str,
    ollama_base: str,
    mock_vulnerability: float,
    verbose: bool = False,
) -> tuple[dict, dict[str, str], list[str]]:
    """
    Run ``run_benchmark_for_target`` for each spec like ``mock``, ``ollama:llama3``, ``openai``.

    Returns ``(models_payload, errors_by_spec, ordered_labels)``.
    """
    models_out: dict = {}
    errors: dict[str, str] = {}
    order: list[str] = []

    for raw in spec_strings:
        spec = raw.strip()
        if not spec:
            continue
        kind, mid = parse_benchmark_spec(spec)
        try:
            label, pl = build_pipeline_callable(
                kind, mid, system_prompt, ollama_base, mock_vulnerability
            )
            order.append(label)
            models_out[label] = run_benchmark_for_target(
                label, pl, attacks, sensitivity, verbose=verbose
            )
        except Exception as e:
            errors[spec] = str(e)

    return models_out, errors, order


def primary_benchmark_label(ordered_labels: list[str], models_out: dict) -> str:
    """First label from the user's benchmark order that produced results."""
    for lab in ordered_labels:
        if lab in models_out:
            return lab
    if models_out:
        return next(iter(models_out))
    raise ValueError("No successful benchmark targets")
