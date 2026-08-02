from scripts.update_encodec_docs import replace_block


def test_replace_block_is_deterministic_and_preserves_surroundings():
    source = "before\n<!-- S -->\nold\n<!-- E -->\nafter\n"
    expected = "before\n<!-- S -->\nnew\n<!-- E -->\nafter\n"

    assert replace_block(source, "<!-- S -->", "<!-- E -->", "new") == expected
