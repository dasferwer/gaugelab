import asyncio
import secrets
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, Header
from fastapi.responses import JSONResponse, Response
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gaugelab.config import Settings
from gaugelab.models import Artifact, Gate, Sample
from gaugelab.schemas import Dataset, GateInput, Model, Prompt, RunInput
from gaugelab.scoring import compare
from gaugelab.service import Conflict, NotFound, create_run, report, save_artifact


def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        engine = create_async_engine(settings.database_url, pool_pre_ping=True)
        app.state.sessions = async_sessionmaker(engine, expire_on_commit=False)
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(title="GaugeLab", version="0.1.0", lifespan=lifespan)

    async def admin(authorization: str = Header(default="")):
        if not secrets.compare_digest(
            authorization.encode(), ("Bearer " + settings.admin_token.get_secret_value()).encode()
        ):
            from fastapi import HTTPException

            raise HTTPException(401, "Неверный токен")

    async def session():
        async with app.state.sessions() as db:
            yield db

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        return JSONResponse({"detail": "База данных недоступна"}, status_code=503)

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(NotFound)
    async def not_found(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.get("/health/live")
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready(db=Depends(session)):
        await db.execute(text("SELECT 1"))
        return {"status": "ready"}

    router = APIRouter(prefix="/v1", dependencies=[Depends(admin)])

    async def artifact_response(db, kind, body):
        row, created = await save_artifact(db, kind, body)
        await db.commit()
        return JSONResponse(
            {
                "id": str(row.id),
                "kind": row.kind,
                "content_hash": row.content_hash,
                "payload": row.payload,
            },
            status_code=201 if created else 200,
        )

    @router.post("/datasets")
    async def datasets(body: Dataset, db=Depends(session)):
        return await artifact_response(db, "dataset", body)

    @router.post("/prompts")
    async def prompts(body: Prompt, db=Depends(session)):
        return await artifact_response(db, "prompt", body)

    @router.post("/models")
    async def models(body: Model, db=Depends(session)):
        return await artifact_response(db, "model", body)

    @router.get("/artifacts/{artifact_id}")
    async def artifact(artifact_id: UUID, db=Depends(session)):
        row = await db.get(Artifact, artifact_id)
        if not row:
            raise NotFound("Артефакт не найден")
        return {
            "id": row.id,
            "kind": row.kind,
            "content_hash": row.content_hash,
            "payload": row.payload,
        }

    @router.post("/runs")
    async def runs(body: RunInput, db=Depends(session)):
        row, created = await create_run(db, body)
        await db.commit()
        return JSONResponse(
            {"id": str(row.id), "report_url": f"/v1/runs/{row.id}"},
            status_code=202 if created else 200,
        )

    @router.get("/runs/{run_id}")
    async def get_run(run_id: UUID, db=Depends(session)):
        return await report(db, run_id)

    @router.post("/gates", status_code=201)
    async def gates(body: GateInput, db=Depends(session)):
        baseline = await report(db, body.baseline_id)
        candidate = await report(db, body.candidate_id)
        result = await asyncio.to_thread(compare, baseline, candidate, body.policy)
        gate = Gate(
            baseline_id=body.baseline_id,
            candidate_id=body.candidate_id,
            result={
                **result,
                "baseline": baseline["summary"],
                "candidate": candidate["summary"],
                "baseline_request_hash": baseline["request_hash"],
                "candidate_request_hash": candidate["request_hash"],
            },
        )
        db.add(gate)
        await db.commit()
        return {
            "id": gate.id,
            "baseline_id": gate.baseline_id,
            "candidate_id": gate.candidate_id,
            **gate.result,
        }

    @router.get("/gates/{gate_id}")
    async def get_gate(gate_id: UUID, db=Depends(session)):
        row = await db.get(Gate, gate_id)
        if not row:
            raise NotFound("Проверка не найдена")
        return {
            "id": row.id,
            "baseline_id": row.baseline_id,
            "candidate_id": row.candidate_id,
            **row.result,
        }

    @app.get("/metrics", dependencies=[Depends(admin)])
    async def metrics(db=Depends(session)):
        counts = dict(
            (await db.execute(select(Sample.status, func.count()).group_by(Sample.status))).all()
        )
        lines = [
            "# HELP gaugelab_samples Количество заданий по состояниям.",
            "# TYPE gaugelab_samples gauge",
        ]
        lines += [
            f'gaugelab_samples{{status="{s}"}} {counts.get(s, 0)}'
            for s in ("pending", "running", "done", "error")
        ]
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    app.include_router(router)
    return app
