from scripts.synthesize_codec_tts import optional_metric


def test_optional_metric_marks_no_decodable_outputs():
    assert optional_metric([]) == {
        "mean": None,
        "clip_bootstrap_95_ci": None,
        "n_valid": 0,
    }


def test_optional_metric_reports_valid_count_and_interval():
    result = optional_metric([1.0, 2.0, 3.0])
    assert result["mean"] == 2.0
    assert result["n_valid"] == 3
    assert len(result["clip_bootstrap_95_ci"]) == 2
