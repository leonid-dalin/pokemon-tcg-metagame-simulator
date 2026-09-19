import pytest

from src.core.urls import POR_URLS


@pytest.mark.unit
def test_mega_starmie_por_url_uses_the_matchups_path():
    url = next(url for url in POR_URLS if "mega-starmie-ex" in url)
    assert "/mega-starmie-ex/matchups?" in url
    assert "?matchups?" not in url