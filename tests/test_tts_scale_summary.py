import numpy as np

from scripts.summarize_tts_scale_study import N_BOOTSTRAP, ci


def test_seed_bootstrap_ci_is_deterministic_and_reports_mean():
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    first = ci(values, n_resamples=200)
    second = ci(values, n_resamples=200)

    assert first == second
    assert first[0] == 3.0
    assert first[1] <= first[0] <= first[2]
    assert N_BOOTSTRAP == 10_000


def test_paired_effect_uses_seedwise_differences():
    stage_a = np.asarray([1.0, 10.0, 2.0, 20.0, 3.0])
    stage_b = np.asarray([2.0, 9.0, 4.0, 18.0, 6.0])

    effect = ci(stage_b - stage_a, n_resamples=200)

    assert effect[0] == np.mean([1.0, -1.0, 2.0, -2.0, 3.0])


def test_report_text_is_portable_utf8(tmp_path):
    report = tmp_path / "REPORT.md"
    report.write_text(
        "Scaled phonemes − scaled bytes", encoding="utf-8", newline="\n"
    )

    assert report.read_bytes() == "Scaled phonemes − scaled bytes".encode("utf-8")
