from r2.stimuli.transforms import SELF_LABEL_TOKENS, self_label_tokens


def test_self_label_tokens_whole_word_case_insensitive():
    assert set(SELF_LABEL_TOKENS) == {"fake", "fabricated", "sham", "bogus", "dummy", "placeholder",
                                      "made-up", "made up", "not real"}
    assert self_label_tokens("DOI:10.9999/FAKE.1") == ["fake"]
    assert self_label_tokens("line one\nfake line") == ["fake"]
    assert self_label_tokens("Hisham, fakery, shampoo, dummyvar, placeholders, unreal") == []
    assert self_label_tokens("a Made-Up trial; data made\nup; NOT REAL; a Sham arm") == [
        "made up", "made-up", "not real", "sham"]
