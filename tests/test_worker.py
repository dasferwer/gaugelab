import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from conftest import artifacts, new_run
from sqlalchemy import func, select, update

from gaugelab.models import Sample
from gaugelab.provider import ProviderError, evaluate, generate
from gaugelab.service import claim, finish, report
from gaugelab.worker import work_once


async def drain(sessions, settings, client):
    for _ in range(100):
        if not await work_once(sessions, settings, client):
            break


async def test_complete_run_and_judge(sessions, settings, model_client):
    ids = await artifacts(sessions)
    run = await new_run(sessions, ids, judge_model_id=ids[2], repeats=2)
    await asyncio.gather(
        drain(sessions, settings, model_client), drain(sessions, settings, model_client)
    )
    async with sessions() as db:
        data = await report(db, run)
    assert data["summary"]["status"] == "completed"
    assert data["summary"]["successful"] == 20
    assert data["summary"]["metrics"]["judge_score"] == 1
    assert data["summary"]["metrics"]["exact_match"] == 1
    assert data["summary"]["cost_complete"]
    assert all(s["attempts"] == 1 for s in data["results"])


async def test_no_expected_answer_in_generation_request(sessions, settings, model_client):
    ids = await artifacts(sessions)
    await new_run(sessions, ids)
    job = await claim(sessions, settings)
    output = await evaluate(model_client, settings, job)
    assert job["case"]["expected"] not in output["rendered_prompt"]
    assert output["rendered_prompt"] == job["case"]["question"]


async def test_concurrent_claims_and_expired_fencing(sessions, settings):
    ids = await artifacts(sessions)
    await new_run(sessions, ids)
    jobs = await asyncio.gather(*(claim(sessions, settings) for _ in range(15)))
    actual = [j for j in jobs if j]
    assert len(actual) == len({j["id"] for j in actual}) == 10
    first = actual[0]
    async with sessions.begin() as db:
        await db.execute(
            update(Sample)
            .where(Sample.id == first["id"])
            .values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
        )
    second = await claim(sessions, settings)
    assert first["id"] == second["id"] and first["token"] != second["token"]
    assert not await finish(sessions, settings, first, error="stale")
    assert await finish(sessions, settings, second, error="temporary")


async def test_expired_last_attempt_is_terminal(sessions, settings):
    settings.max_attempts = 1
    ids = await artifacts(sessions)
    await new_run(sessions, ids)
    jobs = [await claim(sessions, settings) for _ in range(10)]
    async with sessions.begin() as db:
        await db.execute(
            update(Sample).values(lease_until=datetime.now(UTC) - timedelta(seconds=1))
        )
    for _ in jobs:
        assert await claim(sessions, settings) is None
    async with sessions() as db:
        assert (
            await db.scalar(
                select(func.count()).select_from(Sample).where(Sample.status == "error")
            )
            == 10
        )


async def test_unavailable_model_eventually_fails(sessions, settings):
    settings.max_attempts = 1
    ids = await artifacts(sessions)
    run = await new_run(sessions, ids)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as client:
        await drain(sessions, settings, client)
    async with sessions() as db:
        data = await report(db, run)
    assert data["summary"]["failed"] == 10
    assert data["summary"]["metrics"]["exact_match"] == 0
    assert not data["summary"]["cost_complete"]


def ollama_model():
    return {
        "backend": "ollama",
        "model_name": "test:fixed",
        "artifact_digest": "a" * 64,
        "input_per_million": "2",
        "output_per_million": "4",
    }


@pytest.mark.parametrize("changed", [False, True])
async def test_ollama_adapter_and_digest_validation(settings, changed):
    calls = []

    def response(request):
        calls.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200,
                json={"models": [{"name": "test:fixed", "digest": ("b" if changed else "a") * 64}]},
            )
        return httpx.Response(
            200,
            json={"done": True, "response": "answer", "prompt_eval_count": 100, "eval_count": 20},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        if changed:
            with pytest.raises(ProviderError, match="model_digest_changed"):
                await generate(client, settings, ollama_model(), "question", 42, 0, 100)
            assert len(calls) == 1
        else:
            result = await generate(client, settings, ollama_model(), "question", 42, 0, 100)
            assert len(calls) == 3 and result["text"] == "answer"
            assert result["cost"] == "0.00028"


@pytest.mark.parametrize("usage", [{}, {"prompt_eval_count": -1, "eval_count": 2}])
async def test_missing_usage_never_becomes_known_zero(settings, usage):
    def response(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "test:fixed", "digest": "a" * 64}]}
            )
        return httpx.Response(200, json={"done": True, "response": "answer", **usage})

    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        result = await generate(client, settings, ollama_model(), "q", 1, 0, 100)
    assert not result["usage_complete"] and result["input_tokens"] is None


async def test_judge_invalid_json_fails_sample(sessions, settings):
    ids = await artifacts(sessions)
    await new_run(sessions, ids, judge_model_id=ids[2])
    job = await claim(sessions, settings)
    response = {
        "model_digest": "demo-v1",
        "text": "not json",
        "input_tokens": 1,
        "output_tokens": 1,
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as client:
        with pytest.raises(ProviderError, match="invalid_judge_response"):
            await evaluate(client, settings, job)


async def test_repeat_variability_is_reported(sessions, settings, model_client):
    ids = await artifacts(sessions, "DEMO_NOISY {question}")
    run = await new_run(sessions, ids, repeats=2)
    await drain(sessions, settings, model_client)
    async with sessions() as db:
        data = await report(db, run)
    assert data["summary"]["metrics"]["exact_match"] == 0.5
    assert data["summary"]["mean_repeat_stddev"]["exact_match"] == 0.5
