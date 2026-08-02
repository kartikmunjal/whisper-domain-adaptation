from collections import Counter

import pytest
import torch

from scripts.reconstruct_encodec_eval import (
    codebook_report,
    entropy_from_counts,
    merge_code_counts,
)


def test_entropy_and_codebook_utilization_are_mechanical():
    counts = [Counter({0: 2, 1: 2}), Counter({0: 4})]

    report = codebook_report(counts, codebook_size=4, frame_rate=10.0)

    assert entropy_from_counts(counts[0]) == 1.0
    assert report["nominal_fixed_width_bps"] == 40.0
    assert report["empirical_entropy_bps"] == 10.0
    assert report["entropy_utilization"] == 0.25
    assert report["codebooks"][0]["unique_fraction"] == 0.5


def test_merge_code_counts_validates_and_preserves_codebooks():
    totals = [Counter(), Counter()]
    codes = torch.tensor([[[[0, 1, 1], [2, 2, 3]]]])

    merge_code_counts(totals, codes)

    assert totals[0] == Counter({1: 2, 0: 1})
    assert totals[1] == Counter({2: 2, 3: 1})
    with pytest.raises(ValueError, match="container"):
        merge_code_counts([Counter()], codes)
