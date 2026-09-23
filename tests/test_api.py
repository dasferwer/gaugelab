import asyncio
from uuid import uuid4

from conftest import dataset_payload
from sqlalchemy import func, select

from gaugelab.models import Sample
from gaugelab.worker import work_once


async def prepare(api, template="{question}"):
    dataset = await api.post("/v1/datasets", json=dataset_payload())
    prompt = await api.post(
        "/v1/prompts", json={"name": uuid4().hex, "version": "1", "template": template}
    )
    model = await api.post("/v1/models", json={"name": uuid4().hex, "version": "1"})
    assert dataset.status_code == prompt.status_code == model.status_code == 201
    return {
        "request_id": str(uuid4()),
        "dataset_id": dataset.json()["id"],
        "prompt_id": prompt.json()["id"],
        "model_id": model.json()["id"],
    }


async def test_auth_version_conflicts_and_validation(api):
    assert (await api.get("/metrics", headers={"Authorization": "wrong"})).status_code == 401
    body = dataset_payload()
    first = await api.post("/v1/datasets", json=body)
    second = await api.post("/v1/datasets", json=body)
    assert first.status_code == 201 and second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    body["cases"][0]["expected"] = "Changed"
    assert (await api.post("/v1/datasets", json=body)).status_code == 409
    assert (
        await api.post("/v1/prompts", json={"name": "x", "version": "1", "template": "No slot"})
    ).status_code == 422
    assert (await api.get(f"/v1/artifacts/{uuid4()}")).status_code == 404
    assert (await api.get("/health/ready")).status_code == 200
    assert (await api.get("/health/live")).status_code == 200


async def test_concurrent_run_creation_is_idempotent(api, sessions):
    body = await prepare(api)
    responses = await asyncio.gather(*(api.post("/v1/runs", json=body) for _ in range(12)))
    assert sum(r.status_code == 202 for r in responses) == 1
    assert all(r.status_code in (200, 202) for r in responses)
    async with sessions() as db:
        assert await db.scalar(select(func.count()).select_from(Sample)) == 10
    assert (await api.post("/v1/runs", json={**body, "seed": 100})).status_code == 409
    assert (
        await api.post(
            "/v1/runs", json={**body, "request_id": str(uuid4()), "dataset_id": body["model_id"]}
        )
    ).status_code == 404


async def test_end_to_end_report_and_gate(api, sessions, settings, model_client):
    body = await prepare(api)
    baseline = (await api.post("/v1/runs", json=body)).json()["id"]
    prompt = (
        await api.post(
            "/v1/prompts", json={"name": "bad", "version": "1", "template": "DEMO_WRONG {question}"}
        )
    ).json()
    candidate = (
        await api.post(
            "/v1/runs", json={**body, "request_id": str(uuid4()), "prompt_id": prompt["id"]}
        )
    ).json()["id"]
    early = await api.post("/v1/gates", json={"baseline_id": baseline, "candidate_id": candidate})
    assert early.json()["decision"] == "blocked"
    for _ in range(20):
        assert await work_once(sessions, settings, model_client)
    gate = await api.post("/v1/gates", json={"baseline_id": baseline, "candidate_id": candidate})
    assert gate.status_code == 201 and gate.json()["decision"] == "blocked"
    assert (await api.get("/v1/gates/" + gate.json()["id"])).json() == gate.json()
    report = (await api.get("/v1/runs/" + baseline)).json()
    assert report["summary"]["metrics"]["exact_match"] == 1
    assert report["snapshot"]["dataset_hash"]
    assert 'gaugelab_samples{status="done"} 20' in (await api.get("/metrics")).text
