"""
WER computation with domain-specific breakdowns.

Overall WER is a coarse metric. For domain adaptation the interesting question is:
  1. How much did WER on domain terms (medical/financial jargon) improve?
  2. Did fine-tuning degrade WER on general/common words (catastrophic forgetting)?

We compute three numbers:
  - wer_overall     : standard WER across the full eval set
  - wer_domain_terms: WER computed only on transcripts containing domain terms
  - wer_common_terms: WER on transcripts with no domain terms (regression check)

The split is done at the utterance level: an utterance is "domain" if it
contains any word from the domain vocabulary (medical_terms.txt or financial
terms from the synthesis script).

jiwer is used for WER computation because it handles normalisation (punctuation
removal, case folding) out of the box. The transforms applied match what base
Whisper applies during its own eval (no contractions expansion needed for
medical/financial text).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jiwer
import numpy as np
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)

# jiwer transform — same normalisation Whisper uses during evaluation
_WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


@dataclass
class WERReport:
    wer_overall: float
    wer_domain_terms: float
    wer_common_terms: float
    n_total: int
    n_domain: int
    n_common: int
    per_term_wer: dict[str, float] = field(default_factory=dict)


def compute_wer_metrics(pred, processor: WhisperProcessor) -> dict[str, float]:
    """
    Trainer-compatible compute_metrics callback.

    Called by Seq2SeqTrainer after each evaluation step. Returns a dict
    that the Trainer logs and uses for model selection.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 (ignore index) with pad token for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_strs = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_strs = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer = jiwer.wer(
        label_strs, pred_strs,
        truth_transform=_WER_TRANSFORM,
        hypothesis_transform=_WER_TRANSFORM,
    )

    return {"wer": round(wer, 4)}


class DomainWERAnalyzer:
    """
    Compute WER broken down by utterances containing domain vocabulary.

    Parameters
    ----------
    domain_vocab : set or list of domain terms (lower-cased internally)
    """

    def __init__(self, domain_vocab: set[str] | list[str]):
        # Normalise to lowercase tokens for matching
        self._vocab = {w.lower().strip() for w in domain_vocab}

    def _contains_domain_term(self, text: str) -> bool:
        """True if the text contains any domain vocabulary term."""
        text_lower = text.lower()
        for term in self._vocab:
            if term in text_lower:
                return True
        return False

    def analyze(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> WERReport:
        """
        Compute overall, domain, and common WER.

        Parameters
        ----------
        references  : ground-truth transcripts
        hypotheses  : model predictions

        Returns
        -------
        WERReport with per-category WER values
        """
        assert len(references) == len(hypotheses), "lengths must match"

        domain_refs, domain_hyps = [], []
        common_refs, common_hyps = [], []

        for ref, hyp in zip(references, hypotheses):
            if self._contains_domain_term(ref):
                domain_refs.append(ref)
                domain_hyps.append(hyp)
            else:
                common_refs.append(ref)
                common_hyps.append(hyp)

        wer_overall = jiwer.wer(
            references, hypotheses,
            truth_transform=_WER_TRANSFORM,
            hypothesis_transform=_WER_TRANSFORM,
        )

        wer_domain = jiwer.wer(
            domain_refs, domain_hyps,
            truth_transform=_WER_TRANSFORM,
            hypothesis_transform=_WER_TRANSFORM,
        ) if domain_refs else float("nan")

        wer_common = jiwer.wer(
            common_refs, common_hyps,
            truth_transform=_WER_TRANSFORM,
            hypothesis_transform=_WER_TRANSFORM,
        ) if common_refs else float("nan")

        # Per-term WER: score each term independently
        per_term = self._per_term_wer(references, hypotheses)

        return WERReport(
            wer_overall=round(wer_overall, 4),
            wer_domain_terms=round(wer_domain, 4) if not np.isnan(wer_domain) else float("nan"),
            wer_common_terms=round(wer_common, 4) if not np.isnan(wer_common) else float("nan"),
            n_total=len(references),
            n_domain=len(domain_refs),
            n_common=len(common_refs),
            per_term_wer=per_term,
        )

    def _per_term_wer(
        self, references: list[str], hypotheses: list[str]
    ) -> dict[str, float]:
        """WER for utterances containing each individual term."""
        per_term: dict[str, tuple[list, list]] = {t: ([], []) for t in self._vocab}

        for ref, hyp in zip(references, hypotheses):
            ref_lower = ref.lower()
            for term in self._vocab:
                if term in ref_lower:
                    per_term[term][0].append(ref)
                    per_term[term][1].append(hyp)

        result = {}
        for term, (refs, hyps) in per_term.items():
            if refs:
                w = jiwer.wer(
                    refs, hyps,
                    truth_transform=_WER_TRANSFORM,
                    hypothesis_transform=_WER_TRANSFORM,
                )
                result[term] = round(w, 4)

        return result


def load_domain_vocab(vocab_file: str | Path) -> set[str]:
    """Load vocabulary from a plain-text file (one term per line)."""
    path = Path(vocab_file)
    if not path.exists():
        raise FileNotFoundError(f"Domain vocab file not found: {path}")
    with open(path) as f:
        terms = {line.strip().lower() for line in f if line.strip()}
    logger.info("Loaded %d domain terms from %s", len(terms), path)
    return terms
