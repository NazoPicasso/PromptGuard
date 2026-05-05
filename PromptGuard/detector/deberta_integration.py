"""
Future DeBERTa-based injection classifier integration (stub / design notes).

This module documents the intended integration **without** adding heavy ML dependencies
to the default install. When you add ``transformers`` + fine-tuned weights, implement
:class:`DebertaInjectionClassifier` below and wire it from :class:`detector.guard.PromptGuard`.

Suggested layout for production:

```
models/
  deberta-injection/
    config.json
    model.safetensors (or pytorch_model.bin)
    tokenizer files
```

Integration points
------------------
1. **Guard (pre-inference)** — combine ``regex_score`` + ``deberta_prob`` with learned weights.
2. **Evaluation-only** — log classifier scores next to ``detection_debug`` for analysis.

API sketch::

    clf = DebertaInjectionClassifier.from_pretrained("models/deberta-injection")
    prob = clf.predict_proba(user_prompt)["injection"]

Protocol (optional): define ``predict_proba(text: str) -> dict[str, float]`` and call from
``PromptGuard._score_prompt`` when ``classifier=...`` is passed.

Implementation checklist for Phase 3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- [ ] Train sequence classifier on labeled prompts (binary or multi-label).
- [ ] Export HF ``AutoModelForSequenceClassification`` + tokenizer.
- [ ] Batch prompts in ``evaluate_pipeline`` for throughput (optional).
- [ ] Calibrate threshold on validation set; expose ``CALIBRATION_JSON``.
- [ ] Add CI job to regression-test ASR/FPR vs baseline harness.

For now, :meth:`DebertaInjectionClassifier.predict_proba` raises to make the gap explicit.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class InjectionClassifier(Protocol):
    """Minimal interface for a prompt-level injection classifier."""

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return label -> probability (e.g. ``{\"benign\": 0.2, \"injection\": 0.8}``)."""
        ...


class DebertaInjectionClassifier:
    """
    Placeholder for a Hugging Face DeBERTa-v3 sequence classifier.

    Replace ``NotImplementedError`` with ``AutoModelForSequenceClassification.from_pretrained``.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    @classmethod
    def from_pretrained(cls, path: str) -> "DebertaInjectionClassifier":
        return cls(model_path=path)

    def predict_proba(self, text: str) -> dict[str, float]:
        raise NotImplementedError(
            "DeBERTa classifier not loaded. Train/export a model and implement predict_proba "
            "(see detector/deberta_integration.py)."
        )

    def predict_label(self, text: str) -> str:
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)  # type: ignore
