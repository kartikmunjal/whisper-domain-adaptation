import numpy as np
import pytest

from scripts.summarize_encodec_benchmark import paired_ratio_ci, trial_mean_ci


def test_trial_intervals_are_deterministic_and_keep_trial_unit():
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    assert trial_mean_ci(values, 200) == trial_mean_ci(values, 200)
    assert trial_mean_ci(values, 200)[0] == 3.0


def test_paired_ratio_uses_aligned_adapter_trials():
    ratio = paired_ratio_ci([2, 4, 6, 8, 10], [1, 2, 3, 4, 5], 200)

    assert ratio == [2.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="positive aligned"):
        paired_ratio_ci([1, 2], [1, 0], 20)
