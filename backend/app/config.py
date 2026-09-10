"""Application configuration.

All settings are environment-driven so the same image runs locally and on Render.
Secrets (ANTHROPIC_API_KEY) are only ever read here, server-side.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- General -------------------------------------------------------
    app_name: str = "FMN Manufacturing Machine Risk API"
    app_version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = "INFO"

    # --- Paths ---------------------------------------------------------
    data_path: Path = REPO_ROOT / "data" / "project2_manufacturing_sensors.csv"
    model_dir: Path = REPO_ROOT / "models" / "manufacturing"

    # --- CORS ----------------------------------------------------------
    # Comma-separated list is also accepted (FRONTEND_URL=https://a.app,https://b.app)
    frontend_url: str = "http://localhost:3000"

    # --- LLM -----------------------------------------------------------
    anthropic_api_key: str | None = None
    model_name: str = "claude-opus-5"
    llm_max_tokens: int = 900
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    # --- Risk banding (documented in reports/model_summary.md) ---------
    # Thresholds are written by the training job into metadata.json; these are
    # only fallbacks used if metadata is unavailable.
    fallback_threshold: float = 0.5

    @property
    def cors_origins(self) -> list[str]:
        raw = [o.strip().rstrip("/") for o in self.frontend_url.split(",")]
        origins = [o for o in raw if o]

        # Local development convenience. Next.js silently increments its port
        # when 3000 is taken, which otherwise turns into a confusing CORS
        # failure that looks like the backend is down. Allowing the few ports
        # it actually drifts to costs nothing: these origins are only reachable
        # from the developer's own machine.
        #
        # Production is unaffected - it is pinned to FRONTEND_URL alone.
        if self.environment.lower() != "production":
            for port in (3000, 3001, 3002):
                for host in ("localhost", "127.0.0.1"):
                    origins.append(f"http://{host}:{port}")

        # De-duplicate while preserving order.
        return list(dict.fromkeys(origins))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
