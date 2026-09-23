from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Case(StrictModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    question: str = Field(min_length=1, max_length=4000)
    expected: str = Field(min_length=1, max_length=4000)
    required_terms: list[str] = Field(min_length=1, max_length=20)

    @field_validator("required_terms")
    @classmethod
    def valid_terms(cls, value):
        if any(not term.strip() or len(term) > 200 for term in value):
            raise ValueError("Ключевые фразы должны содержать от 1 до 200 символов")
        if len(set(term.casefold().strip() for term in value)) != len(value):
            raise ValueError("Ключевые фразы не должны повторяться")
        return value


class Dataset(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=40)
    split: Literal["validation", "test"] = "validation"
    cases: list[Case] = Field(min_length=1, max_length=200)

    @field_validator("cases")
    @classmethod
    def unique_ids(cls, value):
        if len({case.id for case in value}) != len(value):
            raise ValueError("Идентификаторы примеров должны быть уникальными")
        return value


class Prompt(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=40)
    template: str = Field(min_length=1, max_length=5000)

    @field_validator("template")
    @classmethod
    def question_slot(cls, value):
        if value.count("{question}") != 1:
            raise ValueError("Шаблон должен содержать ровно одну подстановку {question}")
        return value


class Model(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    version: str = Field(min_length=1, max_length=40)
    backend: Literal["demo", "ollama"] = "demo"
    model_name: str = Field(default="demo-v1", min_length=1, max_length=100)
    artifact_digest: str = Field(default="demo-v1", min_length=1, max_length=100)
    input_per_million: Decimal = Field(
        default=Decimal("0"), ge=0, le=10000, max_digits=12, decimal_places=6
    )
    output_per_million: Decimal = Field(
        default=Decimal("0"), ge=0, le=10000, max_digits=12, decimal_places=6
    )

    @model_validator(mode="after")
    def digest_required(self):
        if self.backend == "demo" and (
            self.model_name != "demo-v1" or self.artifact_digest != "demo-v1"
        ):
            raise ValueError("Эмулятор поддерживает только demo-v1")
        if self.backend == "ollama" and (
            len(self.artifact_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.artifact_digest)
        ):
            raise ValueError("Для Ollama нужен полный SHA-256 из /api/tags")
        return self


class RunInput(StrictModel):
    request_id: UUID
    dataset_id: UUID
    prompt_id: UUID
    model_id: UUID
    judge_model_id: UUID | None = None
    repeats: int = Field(default=1, ge=1, le=5)
    seed: int = Field(default=42, ge=0, le=2_000_000_000)
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=256, ge=1, le=4096)


class GatePolicy(StrictModel):
    metric: Literal["exact_match", "token_f1", "keyword_recall", "judge_score"] = "exact_match"
    minimum_score: float = Field(default=0.8, ge=0, le=1)
    allowed_regression: float = Field(default=0.05, ge=0, le=1)
    minimum_cases: int = Field(default=5, ge=2, le=200)
    maximum_p95_seconds: float | None = Field(default=None, gt=0, le=3600)
    maximum_cost: Decimal | None = Field(default=None, ge=0, le=1_000_000)


class GateInput(StrictModel):
    baseline_id: UUID
    candidate_id: UUID
    policy: GatePolicy = Field(default_factory=GatePolicy)
