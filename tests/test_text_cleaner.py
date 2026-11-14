from contextual_intelligence.preprocessing.text_cleaner import clean_text


def test_decodes_html_entities():
    text = "doesn&#39;t"
    assert clean_text(text) == "doesn't"


def test_normalises_whitespace():
    text = "News about\ntechnology"
    assert clean_text(text) == "news about technology"


def test_converts_to_lowercase():
    assert clean_text("WORLD NEWS") == "world news"


def test_handles_empty_values():
    assert clean_text("") == ""


def test_handles_non_string_values():
    assert clean_text(None) == ""
