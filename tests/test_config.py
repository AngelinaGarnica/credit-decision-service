from credit.config import Settings


def test_database_url_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite://")

    assert Settings().database_url == "sqlite://"
