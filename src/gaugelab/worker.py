import asyncio
import logging
import time
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gaugelab.config import Settings
from gaugelab.provider import ProviderError, evaluate
from gaugelab.service import claim, finish

log = logging.getLogger("gaugelab.worker")


async def work_once(sessions, settings, client):
    job = await claim(sessions, settings)
    if not job:
        return False
    try:
        result = await evaluate(client, settings, job)
        await finish(sessions, settings, job, result=result)
    except ProviderError as exc:
        await finish(sessions, settings, job, error=str(exc))
        log.warning("sample=%s error=%s", job["id"], str(exc))
    return True


async def main():
    settings = Settings()
    engine = create_async_engine(settings.database_url, pool_pre_ping=True)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with httpx.AsyncClient(
            timeout=settings.request_timeout, follow_redirects=False, trust_env=False
        ) as client:
            while True:
                Path("/tmp/gaugelab-heartbeat").write_text(str(time.time()))
                try:
                    worked = await work_once(sessions, settings, client)
                except Exception as exc:
                    log.error("worker_error=%s", type(exc).__name__)
                    worked = False
                if not worked:
                    await asyncio.sleep(settings.poll_seconds)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
