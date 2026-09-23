"""Проверяет повторные запросы после сбоя модели и восстановление заданий после остановки воркеров."""

import asyncio
import json
import subprocess
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values
from smoke import prepare, submit, wait_complete

ROOT = Path(__file__).resolve().parents[1]


def compose(*args):
    subprocess.run(["docker", "compose", *args], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


async def wait_state(client, identifier, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = await client.get("/v1/runs/" + identifier)
        response.raise_for_status()
        report = response.json()
        if predicate(report):
            return report
        await asyncio.sleep(0.05)
    raise AssertionError("Не удалось воспроизвести нужное состояние")


async def main():
    headers = {"Authorization": "Bearer " + dotenv_values(ROOT / ".env")["ADMIN_TOKEN"]}
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8260", headers=headers, timeout=15
    ) as client:
        try:
            config = await prepare(client, "DEMO_SLOW {question}")
            run = await submit(client, config)
            await wait_state(
                client, run, lambda r: sum(s["status"] == "running" for s in r["results"]) >= 2
            )
            compose("kill", "-s", "SIGKILL", "worker", "worker-2")
            compose("start", "worker", "worker-2")
            recovered = await wait_complete(client, run)
            assert recovered["summary"]["successful"] == 10 and recovered["summary"]["failed"] == 0
            assert any(s["attempts"] >= 2 for s in recovered["results"])
            assert len({s["id"] for s in recovered["results"]}) == 10
            assert recovered["summary"]["cost_complete"] is False
            config = await prepare(client)
            compose("stop", "model")
            unavailable = await submit(client, config)
            await wait_state(client, unavailable, lambda r: any(s["error"] for s in r["results"]))
            compose("start", "model")
            restored = await wait_complete(client, unavailable)
            assert restored["summary"]["successful"] == 10 and restored["summary"]["failed"] == 0
        finally:
            compose("start", "worker", "worker-2", "model")
    print(
        json.dumps(
            {
                "worker_sigkill": "recovered",
                "samples_not_duplicated": True,
                "uncertain_cost_marked": True,
                "provider_outage": "recovered",
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
