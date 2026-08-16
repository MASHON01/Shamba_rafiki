from functools import lru_cache
from pathlib import Path
from typing import Annotated, List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.config import paths


class Settings(BaseSettings):
    """
    Global application configuration.
    Values are loaded from the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # APPLICATION
    # ------------------------------------------------------------------

    APP_NAME: str = "Shamba Rafiki Backend"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Kiosk frontend runs locally in the same browser, so CORS is
    # permissive by default for local origins. Tighten via env in any
    # networked deployment. NoDecode keeps pydantic-settings from
    # JSON-parsing the env value, so a friendly comma-separated form works
    # alongside a JSON array; the validator below splits it.
    CORS_ALLOW_ORIGINS: Annotated[list[str], NoDecode] = ["*"]

    # Max size (bytes) for an uploaded corpus document.
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "console"

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    LLM_PROVIDER: str = "llama_cpp"
    LLM_SERVER_URL: str = "http://localhost:8080"

    MODEL_PATH: str = "models/llama.gguf"
    MODEL_CONTEXT_SIZE: int = 4096
    MODEL_THREADS: int = 8

    # Resilience: split connect vs read so a dead server fails fast while a
    # slow CPU generation is still allowed to finish.
    LLM_CONNECT_TIMEOUT_SECONDS: float = 5.0
    LLM_READ_TIMEOUT_SECONDS: float = 120.0
    LLM_HEALTH_TIMEOUT_SECONDS: float = 5.0

    LLM_RETRY_MAX_ATTEMPTS: int = 3
    LLM_CIRCUIT_FAILURE_THRESHOLD: int = 3
    LLM_CIRCUIT_RESET_SECONDS: float = 30.0

    LLM_RESPONSE_CACHE_ENABLED: bool = True
    LLM_RESPONSE_CACHE_MAX_ENTRIES: int = 256
    WARMUP_ON_STARTUP: bool = True

    MODEL_ID: str = "llama-3.2-3b-instruct-q4_k_m"
    LLAMA_SERVER_BIN: str = "llama-server"

    PROMPT_VERSION: str = "v1"
    SWAHILI_PROMPT_ENRICH: bool = True

    # ------------------------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------------------------

    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # ------------------------------------------------------------------
    # VECTOR STORE
    # ------------------------------------------------------------------

    VECTOR_STORE: str = "data/vector_store"
    EMBEDDING_CACHE: str = "data/embeddings"

    # ------------------------------------------------------------------
    # KNOWLEDGE
    # ------------------------------------------------------------------

    RAW_DOCUMENTS: str = "data/raw_documents"
    PROCESSED_DOCUMENTS: str = "data/processed_documents"

    # ------------------------------------------------------------------
    # IMAGE MODEL
    # ------------------------------------------------------------------

    ONNX_MODEL: str = "models/plant_classifier.onnx"

    # ------------------------------------------------------------------
    # LANGUAGE
    # ------------------------------------------------------------------

    DEFAULT_LANGUAGE: str = "en"

    SUPPORTED_LANGUAGES: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["en", "sw"]
    )

    @field_validator("SUPPORTED_LANGUAGES", "CORS_ALLOW_ORIGINS", mode="before")
    @classmethod
    def _split_csv(cls, value):
        """
        Accept both a JSON array and a plain comma-separated string from
        .env, so SUPPORTED_LANGUAGES=en,sw works as well as ["en","sw"].
        """
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                import json

                return json.loads(text)
            return [item.strip() for item in text.split(",") if item.strip()]
        return value

    # ------------------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------------------

    TOP_K: int = 5
    MAX_CONTEXT_CHUNKS: int = 5
    SIMILARITY_THRESHOLD: float = 0.45

    # ------------------------------------------------------------------
    # VERIFICATION
    # ------------------------------------------------------------------

    ENABLE_VERIFICATION: bool = True
    ENABLE_CONFIDENCE_SCORING: bool = True

    # ------------------------------------------------------------------
    # RESOLVED PATHS (absolute, CWD-independent)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (paths.ROOT_DIR / p)

    @property
    def model_path(self) -> Path:
        return self._resolve(self.MODEL_PATH)

    @property
    def onnx_model_path(self) -> Path:
        return self._resolve(self.ONNX_MODEL)

    @property
    def vector_store_path(self) -> Path:
        return self._resolve(self.VECTOR_STORE)

    @property
    def embedding_cache_path(self) -> Path:
        return self._resolve(self.EMBEDDING_CACHE)

    @property
    def raw_documents_path(self) -> Path:
        return self._resolve(self.RAW_DOCUMENTS)

    @property
    def processed_documents_path(self) -> Path:
        return self._resolve(self.PROCESSED_DOCUMENTS)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton settings instance."""
    return Settings()


settings = get_settings()
