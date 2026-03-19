from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"

    database_url: str = "sqlite:///./data/control_plane.db"

    storage_base_path: str = "./storage"
    context_files_path: str = "./storage/context_files"
    system_prompts_path: str = "./storage/system_prompts"
    policy_documents_path: str = "./storage/policy_documents"
    max_context_file_size_mb: int = 10

    control_plane_api_key: str = "change-me-in-production"
    cors_origins: str = "http://localhost:3000,http://localhost:8001,http://localhost:5500"

    supabase_url: str = "https://tnuwpzjppoumdmqovxwh.supabase.co"
    supabase_anon_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRudXdwempwcG91bWRtcW92eHdoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM3NzM5MjMsImV4cCI6MjA4OTM0OTkyM30.N5W6y2x3CU6cxJ8XYFIMyT9b__0LFzLzj2Siikg9hk0"

    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    def ensure_directories(self):
        for path in [self.context_files_path, self.system_prompts_path, self.policy_documents_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        Path("./data").mkdir(parents=True, exist_ok=True)


settings = Settings()
