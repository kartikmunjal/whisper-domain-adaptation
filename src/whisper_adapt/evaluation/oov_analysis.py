"""
Out-of-vocabulary (OOV) analysis for domain adaptation.

The standard WER metric treats all substitution errors equally, but for domain
adaptation the interesting errors are specifically on OOV terms — words that
are rare or absent in Whisper's training distribution.

This module:
  1. Identifies which domain terms were transcribed incorrectly
  2. Clusters error patterns (e.g., "EBITDA" → "EBB it da" is a phonetic
     confuse; "echocardiogram" → "echo cardio gram" is a segmentation error)
  3. Computes term-level recall: did the term appear at all in the hypothesis?

Term-level recall is often more informative than WER for downstream use:
a medical transcription system that transcribes "myocardial infarction" as
"my card eel infarction" has 67% WER on that term but 100% term recall if
the downstream NLP step can tolerate minor variations.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import jiwer
import pandas as pd


@dataclass
class TermAnalysis:
    term: str
    n_occurrences: int
    term_recall: float           # fraction of times term appeared in hypothesis
    wer_on_term_utterances: float
    common_substitutions: list[tuple[str, int]]  # (substituted_text, count)


@dataclass
class OOVReport:
    summary: pd.DataFrame   # columns: term, n_occurrences, term_recall, wer
    worst_terms: list[str]  # terms with lowest recall
    best_terms: list[str]   # terms with highest recall after fine-tuning


class OOVAnalyzer:
    """
    Detailed per-term analysis for domain OOV vocabulary.

    Parameters
    ----------
    domain_terms : list of domain-specific terms to track
    """

    def __init__(self, domain_terms: list[str]):
        self.domain_terms = [t.lower().strip() for t in domain_terms]

    def analyze(
        self,
        references: list[str],
        hypotheses: list[str],
    ) -> OOVReport:
        """
        Run per-term OOV analysis.

        Returns a report with term-level recall, WER, and common substitutions.
        """
        rows = []
        for term in self.domain_terms:
            analysis = self._analyze_term(term, references, hypotheses)
            if analysis.n_occurrences == 0:
                continue
            rows.append({
                "term": analysis.term,
                "n_occurrences": analysis.n_occurrences,
                "term_recall": round(analysis.term_recall, 3),
                "wer": round(analysis.wer_on_term_utterances, 3),
                "top_substitution": (
                    analysis.common_substitutions[0][0]
                    if analysis.common_substitutions else ""
                ),
            })

        df = pd.DataFrame(rows).sort_values("wer", ascending=False)

        worst = df.head(5)["term"].tolist() if len(df) >= 5 else df["term"].tolist()
        best = df.tail(5)["term"].tolist() if len(df) >= 5 else df["term"].tolist()

        return OOVReport(summary=df, worst_terms=worst, best_terms=best)

    def _analyze_term(
        self, term: str, references: list[str], hypotheses: list[str]
    ) -> TermAnalysis:
        term_refs = []
        term_hyps = []
        recall_hits = 0
        substitutions: Counter = Counter()

        for ref, hyp in zip(references, hypotheses):
            if term not in ref.lower():
                continue

            term_refs.append(ref)
            term_hyps.append(hyp)

            if term in hyp.lower():
                recall_hits += 1
            else:
                # Extract what the model said instead
                ref_words = re.sub(r"[^\w\s]", "", ref.lower()).split()
                hyp_words = re.sub(r"[^\w\s]", "", hyp.lower()).split()
                term_words = term.split()

                # Find the window in ref that contains the term
                for i, w in enumerate(ref_words):
                    window = " ".join(ref_words[i:i + len(term_words)])
                    if window == term:
                        hyp_window = " ".join(hyp_words[i:i + len(term_words)])
                        substitutions[hyp_window] += 1
                        break

        n = len(term_refs)
        if n == 0:
            return TermAnalysis(term=term, n_occurrences=0, term_recall=0.0,
                                wer_on_term_utterances=float("nan"),
                                common_substitutions=[])

        recall = recall_hits / n

        wer = jiwer.wer(
            term_refs, term_hyps,
            truth_transform=jiwer.Compose([
                jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]),
            hypothesis_transform=jiwer.Compose([
                jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                jiwer.ReduceToListOfListOfWords(),
            ]),
        ) if term_refs else float("nan")

        return TermAnalysis(
            term=term,
            n_occurrences=n,
            term_recall=recall,
            wer_on_term_utterances=wer,
            common_substitutions=substitutions.most_common(3),
        )

    def compare(
        self,
        baseline_refs: list[str],
        baseline_hyps: list[str],
        finetuned_hyps: list[str],
    ) -> pd.DataFrame:
        """
        Side-by-side comparison of baseline vs. fine-tuned model on each term.

        Returns a DataFrame with columns:
          term, baseline_recall, finetuned_recall, baseline_wer, finetuned_wer,
          recall_delta, wer_delta
        """
        baseline_report = self.analyze(baseline_refs, baseline_hyps)
        finetuned_report = self.analyze(baseline_refs, finetuned_hyps)

        bl = baseline_report.summary.set_index("term")
        ft = finetuned_report.summary.set_index("term")

        combined = bl.join(ft, lsuffix="_baseline", rsuffix="_finetuned", how="outer")
        combined["recall_delta"] = (
            combined["term_recall_finetuned"] - combined["term_recall_baseline"]
        ).round(3)
        combined["wer_delta"] = (
            combined["wer_finetuned"] - combined["wer_baseline"]
        ).round(3)

        return combined.reset_index().sort_values("wer_delta")
