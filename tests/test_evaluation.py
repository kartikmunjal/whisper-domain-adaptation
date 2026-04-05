"""
Tests for WER computation and OOV analysis.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from whisper_adapt.evaluation.wer import DomainWERAnalyzer
from whisper_adapt.evaluation.oov_analysis import OOVAnalyzer


DOMAIN_VOCAB = {"echocardiogram", "atrial fibrillation", "myocardial infarction"}


class TestDomainWERAnalyzer:
    def setup_method(self):
        self.analyzer = DomainWERAnalyzer(DOMAIN_VOCAB)

    def test_perfect_transcription(self):
        refs = [
            "The patient presented with atrial fibrillation.",
            "Good morning how are you.",
        ]
        hyps = [
            "The patient presented with atrial fibrillation.",
            "Good morning how are you.",
        ]
        report = self.analyzer.analyze(refs, hyps)
        assert report.wer_overall == 0.0
        assert report.wer_domain_terms == 0.0
        assert report.wer_common_terms == 0.0

    def test_domain_term_error(self):
        refs = ["The echocardiogram was normal."]
        hyps = ["The echo cardio gram was normal."]  # Segmented incorrectly
        report = self.analyzer.analyze(refs, hyps)
        assert report.wer_domain_terms > 0.0
        assert report.n_domain == 1
        assert report.n_common == 0

    def test_common_term_no_domain(self):
        refs = ["The patient is stable."]
        hyps = ["The patient is stable."]
        report = self.analyzer.analyze(refs, hyps)
        assert report.n_domain == 0
        assert report.n_common == 1
        assert report.wer_common_terms == 0.0

    def test_split_is_correct(self):
        refs = [
            "Echocardiogram showed normal function.",
            "Please follow up tomorrow.",
            "Myocardial infarction ruled out.",
        ]
        hyps = refs[:]  # All correct
        report = self.analyzer.analyze(refs, hyps)
        assert report.n_domain == 2
        assert report.n_common == 1

    def test_case_insensitive(self):
        # Domain vocab is lower-cased internally
        refs = ["ATRIAL FIBRILLATION was detected."]
        hyps = ["atrial fibrillation was detected."]
        report = self.analyzer.analyze(refs, hyps)
        # Reference contains domain term; WER should be 0 (same content)
        assert report.n_domain == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            self.analyzer.analyze(["a", "b"], ["a"])


class TestOOVAnalyzer:
    def setup_method(self):
        self.analyzer = OOVAnalyzer(list(DOMAIN_VOCAB))

    def test_perfect_recall(self):
        refs = ["Patient had atrial fibrillation.", "Echocardiogram result negative."]
        hyps = refs[:]
        report = self.analyzer.analyze(refs, hyps)
        # Both terms should have recall = 1.0
        df = report.summary
        for term in ["atrial fibrillation", "echocardiogram"]:
            row = df[df["term"] == term]
            if len(row) > 0:
                assert row.iloc[0]["term_recall"] == pytest.approx(1.0)

    def test_zero_recall_on_miss(self):
        refs = ["The echocardiogram showed normal systolic function."]
        hyps = ["The echo showed normal systolic function."]
        report = self.analyzer.analyze(refs, hyps)
        df = report.summary
        row = df[df["term"] == "echocardiogram"]
        if len(row) > 0:
            assert row.iloc[0]["term_recall"] < 1.0

    def test_missing_term_excluded(self):
        # myocardial infarction not in references → should not appear in report
        refs = ["Atrial fibrillation detected."]
        hyps = ["Atrial fibrillation detected."]
        report = self.analyzer.analyze(refs, hyps)
        assert "myocardial infarction" not in report.summary["term"].values

    def test_compare_returns_dataframe(self):
        import pandas as pd
        refs = ["The echocardiogram was normal."] * 5
        baseline_hyps = ["The echo was normal."] * 5
        finetuned_hyps = ["The echocardiogram was normal."] * 5
        df = self.analyzer.compare(refs, baseline_hyps, finetuned_hyps)
        assert isinstance(df, pd.DataFrame)
        assert "recall_delta" in df.columns
        assert "wer_delta" in df.columns
