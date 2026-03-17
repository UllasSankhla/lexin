from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8001
    app_env: str = "development"

    database_url: str = "sqlite:///./data/data_plane.db"

    storage_base_path: str = "./storage"
    transcripts_path: str = "./storage/transcripts"
    recordings_path: str = "./storage/recordings"

    control_plane_url: str = "http://localhost:8000"
    control_plane_api_key: str = "change-me-in-production"
    config_cache_ttl_sec: int = 30

    deepgram_api_key: str = ""
    deepgram_stt_model: str = "flux-general-en"
    deepgram_tts_model: str = "aura-2-thalia-en"
    deepgram_stt_language: str = "en-US"
    deepgram_eot_threshold: float = 0.9
    deepgram_eot_timeout_ms: int = 1200
    deepgram_eot_hold_ms: int = 500   # extra hold after EndOfTurn before firing on_final

    cerebras_api_key: str = ""
    cerebras_model: str = "llama-4-scout-17b-16e-instruct"
    cerebras_max_tokens: int = 512
    cerebras_temperature: float = 0.4

    max_call_duration_sec: int = 600   # 10 minutes — also controls receive loop timeout
    silence_timeout_sec: int = 30      # Deepgram end-of-utterance detection window
    session_token_ttl_sec: int = 300
    max_concurrent_calls: int = 50

    cors_origins: str = "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500"

    calendly_api_key: str = ""
    calendly_scheduling_link: str = ""
    calendly_timezone: str = "America/Los_Angeles"

    customer_keys_path: str = "./customer_keys.json"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    def ensure_directories(self):
        for path in [self.transcripts_path, self.recordings_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)


settings = Settings()
