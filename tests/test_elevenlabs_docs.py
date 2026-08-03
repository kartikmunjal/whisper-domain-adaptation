from scripts.update_elevenlabs_docs import replace_block, wer_ci


def test_replace_block_is_deterministic():
    source = "before\n<!-- S -->\nold\n<!-- E -->\nafter\n"
    expected = "before\n<!-- S -->\nnew\n<!-- E -->\nafter\n"
    assert replace_block(source, "<!-- S -->", "<!-- E -->", "new") == expected


def test_wer_ci_formats_fraction_as_percent():
    assert wer_ci([0.00484, 0.0031, 0.00658]) == (
        "0.484% (95% seed-bootstrap CI 0.310–0.658)"
    )
