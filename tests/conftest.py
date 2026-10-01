# ruff: noqa: E402
from safety import ensure_test_environment

# До импорта модулей с engine/settings проверяем все ресурсы тестового профиля.
ensure_test_environment()

import os
import subprocess
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gaugelab.app import create_app
from gaugelab.config import Settings
from gaugelab.demo_model import ANSWERS
from gaugelab.demo_model import app as demo
from gaugelab.schemas import Dataset, Model, Prompt, RunInput
from gaugelab.service import create_run, save_artifact


@pytest.fixture(scope="session", autouse=True)
def migrations():
    subprocess.run(["alembic", "upgrade", "head"], check=True)


@pytest.fixture
def settings():
    return Settings(request_timeout=0.1, lease_seconds=2)


@pytest_asyncio.fixture
async def sessions():
    engine = create_async_engine(os.environ["DATABASE_URL"])
    async with engine.begin() as db:
        await db.execute(text("TRUNCATE artifacts, runs, samples, gates CASCADE"))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest_asyncio.fixture
async def api(sessions, settings):
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer " + settings.admin_token.get_secret_value()},
        ) as client,
    ):
        yield client


@pytest_asyncio.fixture
async def model_client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=demo), base_url="http://model"
    ) as client:
        yield client


def dataset_payload(name=None):
    return {
        "name": name or uuid4().hex,
        "version": "1",
        "cases": [
            {"id": f"case-{i}", "question": q, "expected": a, "required_terms": [a]}
            for i, (q, a) in enumerate(ANSWERS.items())
        ],
    }


async def artifacts(sessions, template="{question}"):
    async with sessions.begin() as db:
        dataset, _ = await save_artifact(db, "dataset", Dataset.model_validate(dataset_payload()))
        prompt, _ = await save_artifact(
            db, "prompt", Prompt(name=uuid4().hex, version="1", template=template)
        )
        model, _ = await save_artifact(db, "model", Model(name=uuid4().hex, version="1"))
        return dataset.id, prompt.id, model.id


async def new_run(sessions, ids, **options):
    request = RunInput(
        request_id=uuid4(), dataset_id=ids[0], prompt_id=ids[1], model_id=ids[2], **options
    )
    async with sessions.begin() as db:
        row, _ = await create_run(db, request)
        return row.id
