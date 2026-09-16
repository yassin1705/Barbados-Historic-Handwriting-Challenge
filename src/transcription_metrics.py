"""Reusable transcription-error metrics and the Barbados Zindi score proxy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import unicodedata

import numpy as np
from rapidfuzz.distance import Levenshtein


@dataclass(frozen=True)
class ZindiScore:
    """Convert weighted WER/CER into the competition score proxy.

    The Barbados challenge uses an equal blend of normalized word and character
    errors. The defaults are the fixed normalization constants used by the
    existing experiments; expose them here so another benchmark can supply its
    own constants without replacing the metric implementation.
    """

    word_normalizer: float = 12.0
    character_normalizer: float = 55.0

    def __call__(self, *, weighted_wer: float, weighted_cer: float) -> float:
        return float(
            1.0
            - 0.5
            * (
                float(weighted_wer) / self.word_normalizer
                + float(weighted_cer) / self.character_normalizer
            )
        )


@dataclass(frozen=True)
class TranscriptionMetrics:
    """Compute normalized weighted WER/CER and an optional score function."""

    score: ZindiScore | None = ZindiScore()
    normalization: str = "NFC"
    strip: bool = True

    def _normalize(self, text: object) -> str:
        value = unicodedata.normalize(self.normalization, str(text))
        return value.strip() if self.strip else value

    def evaluate(self, references: Iterable[object], predictions: Iterable[object]) -> dict[str, float]:
        refs = [self._normalize(value) for value in references]
        hyps = [self._normalize(value) for value in predictions]
        if len(refs) != len(hyps):
            raise ValueError(f"Expected equal reference/prediction counts, got {len(refs)} and {len(hyps)}.")
        if not refs:
            raise ValueError("Cannot score an empty prediction set.")

        word_errors = np.asarray(
            [Levenshtein.distance(ref.split(), hyp.split()) for ref, hyp in zip(refs, hyps)], dtype=float
        )
        character_errors = np.asarray(
            [Levenshtein.distance(ref, hyp) for ref, hyp in zip(refs, hyps)], dtype=float
        )
        word_weights = np.sqrt(np.asarray([max(1, len(ref.split())) for ref in refs], dtype=float))
        character_weights = np.sqrt(np.asarray([max(1, len(ref)) for ref in refs], dtype=float))

        weighted_wer = float(np.dot(word_errors, word_weights) / word_weights.sum())
        weighted_cer = float(np.dot(character_errors, character_weights) / character_weights.sum())
        result = {
            "weighted_wer": weighted_wer,
            "weighted_cer": weighted_cer,
            "mean_weighted_error": (weighted_wer + weighted_cer) / 2.0,
        }
        if self.score is not None:
            result["zindi_score_proxy"] = self.score(
                weighted_wer=weighted_wer, weighted_cer=weighted_cer
            )
        return result

    def __call__(self, references: Iterable[object], predictions: Iterable[object]) -> dict[str, float]:
        return self.evaluate(references, predictions)
