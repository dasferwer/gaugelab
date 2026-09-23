from pydantic import Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://gauge:gauge@localhost:5432/gauge"
    admin_token: SecretStr = Field(min_length=24)
    demo_url: HttpUrl = HttpUrl("http://model:8000")
    ollama_url: HttpUrl = HttpUrl("http://host.docker.internal:11434")
    request_timeout: float = Field(default=10, ge=0.05, le=120)
    lease_seconds: int = Field(default=30, ge=2, le=600)
    poll_seconds: float = Field(default=0.3, ge=0.05, le=30)
    max_attempts: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def safe_configuration(self):
        if self.lease_seconds <= 2 * self.request_timeout + 1:
            raise ValueError("Аренда должна превышать два таймаута запроса минимум на секунду")
        for url in (self.demo_url, self.ollama_url):
            if (
                url.username
                or url.password
                or url.query
                or url.fragment
                or url.path not in (None, "/")
            ):
                raise ValueError("Адрес провайдера должен содержать только схему, хост и порт")
        return self
