from src.api.stream_helpers import resolve_job_id


def test_resolve_job_id_passes_through_non_empty_string():
    assert resolve_job_id("deck-42", "fallback") == "deck-42"


def test_resolve_job_id_decodes_utf8_bytes():
    assert resolve_job_id("deck-42".encode("utf-8"), "fallback") == "deck-42"


def test_resolve_job_id_uses_fallback_for_undecodable_bytes():
    assert resolve_job_id(b"\xff", "fallback") == "fallback"


def test_resolve_job_id_uses_fallback_for_empty_string():
    assert resolve_job_id("", "fallback") == "fallback"


def test_resolve_job_id_uses_fallback_for_empty_bytes():
    assert resolve_job_id(b"", "fallback") == "fallback"


def test_resolve_job_id_uses_fallback_for_missing_value():
    assert resolve_job_id(None, "fallback") == "fallback"
