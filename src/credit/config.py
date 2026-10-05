"""Application settings, read from environment variables (or a local `.env` file)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://credit:credit@localhost:5432/credit"
    raw_data_path: Path = Path("data/raw/cs-training.csv")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    anthropic_api_key: str | None = None  # monitoring summary only; None lets the SDK resolve it


@lru_cache
def get_settings() -> Settings:
    """Return the settings, built once per process."""
    return Settings()
