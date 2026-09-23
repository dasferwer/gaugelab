"""Сравнивает исходный промпт с намеренно ухудшенным и проверяет блокировку регрессии."""

import asyncio
import json
import os
import time
from uuid import uuid4

import httpx

from gaugelab.demo_model import ANSWERS


async def prepare(client, template="{question}"):
    data = {
        "name": "demo-" + uuid4().hex,
        "version": "1",
        "cases": [
            {"id": f"q{i}", "question": q, "expected": a, "required_terms": [a]}
            for i, (q, a) in enumerate(ANSWERS.items())
        ],
    }
    dataset = await client.post("/v1/datasets", json=data)
    prompt = await client.post(
        "/v1/prompts", json={"name": uuid4().hex, "version": "1", "template": template}
    )
    model = await client.post("/v1/models", json={"name": uuid4().hex, "version": "1"})
    for response in (dataset, prompt, model):
        response.raise_for_status()
    return {
        "dataset_id": dataset.json()["id"],
        "prompt_id": prompt.json()["id"],
        "model_id": model.json()["id"],
    }


async def submit(client, config):
    response = await client.post("/v1/runs", json={"request_id": str(uuid4()), **config})
    response.raise_for_status()
    return response.json()["id"]


async def wait_complete(client, identifier, timeout=90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get("/v1/runs/" + identifier)
        response.raise_for_status()
        report = response.json()
        if report["summary"]["status"] == "completed":
            return report
        await asyncio.sleep(0.2)
    raise AssertionError("Прогон не завершился за отведённое время")


async def main():
    headers = {"Authorization": "Bearer " + os.environ["ADMIN_TOKEN"]}
    async with httpx.AsyncClient(base_url="http://api:8000", headers=headers, timeout=20) as client:
        config = await prepare(client)
        config.update(repeats=2, judge_model_id=config["model_id"])
        baseline = await submit(client, config)
        same = await submit(client, config)
        bad_prompt = await client.post(
            "/v1/prompts",
            json={"name": uuid4().hex, "version": "1", "template": "DEMO_WRONG {question}"},
        )
        bad_prompt.raise_for_status()
        changed = await submit(client, {**config, "prompt_id": bad_prompt.json()["id"]})
        reports = await asyncio.gather(
            *(wait_complete(client, identifier) for identifier in (baseline, same, changed))
        )
        assert all(r["summary"]["failed"] == 0 for r in reports)
        assert reports[0]["summary"]["metrics"]["exact_match"] == 1
        assert reports[2]["summary"]["metrics"]["exact_match"] == 0
        passed = await client.post(
            "/v1/gates", json={"baseline_id": baseline, "candidate_id": same}
        )
        blocked = await client.post(
            "/v1/gates", json={"baseline_id": baseline, "candidate_id": changed}
        )
        assert passed.json()["decision"] == "passed"
        assert blocked.json()["decision"] == "blocked"
        judge = await client.post(
            "/v1/gates",
            json={
                "baseline_id": baseline,
                "candidate_id": changed,
                "policy": {"metric": "judge_score"},
            },
        )
        assert judge.json()["decision"] == "blocked"
    print(
        json.dumps(
            {
                "runs": 3,
                "samples": 60,
                "same_prompt": "passed",
                "regressed_prompt": "blocked",
                "judge_regression": "blocked",
                "baseline_id": baseline,
                "candidate_id": changed,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
