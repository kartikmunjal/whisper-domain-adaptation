from scripts.summarize_codec_tts_conditioning import compact_report


def test_compact_report_keeps_primary_evidence_without_attention_matrix():
    report = {
        "seed": 11,
        "n_examples": 98,
        "conditioning": {"shuffled_minus_true_nll": 0.07},
        "free_running_position_error": {"error_counts": [1], "event_counts": [2]},
        "attention": [
            {
                "id": "clip",
                "layer": 0,
                "centroid_monotonicity_r": 0.4,
                "attention_entropy": 0.9,
                "matrix": [[0.25, 0.75]],
            }
        ],
        "checkpoint_sha256": "abc",
        "provenance": {"git_dirty": False},
        "examples": [{"large": "payload"}],
    }

    compact = compact_report(report)

    assert compact["seed"] == 11
    assert compact["conditioning"] == report["conditioning"]
    assert compact["checkpoint_sha256"] == "abc"
    assert compact["provenance"] == {"git_dirty": False}
    assert "matrix" not in compact["attention_summaries"][0]
    assert "examples" not in compact
