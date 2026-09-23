import hashlib
import json
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert

from gaugelab.models import Artifact, Run, Sample
from gaugelab.scoring import EVALUATOR_VERSION, summarize


class Conflict(Exception):
    pass


class NotFound(Exception):
    pass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


async def save_artifact(db, kind, body):
    payload = body.model_dump(mode="json")
    hashed = digest(payload)
    identifier = await db.scalar(
        insert(Artifact)
        .values(
            kind=kind, name=body.name, version=body.version, content_hash=hashed, payload=payload
        )
        .on_conflict_do_nothing(constraint="uq_artifact_version")
        .returning(Artifact.id)
    )
    row = await db.scalar(
        select(Artifact).where(
            Artifact.kind == kind, Artifact.name == body.name, Artifact.version == body.version
        )
    )
    if row.content_hash != hashed:
        raise Conflict("Эта версия уже существует с другим содержимым")
    return row, identifier is not None


async def create_run(db, body):
    config = body.model_dump(mode="json")
    hashed = digest(config)
    snapshot = {
        "config": config,
        "evaluator_version": EVALUATOR_VERSION,
        "judge": None,
        "judge_hash": None,
    }
    for kind, identifier in (
        ("dataset", body.dataset_id),
        ("prompt", body.prompt_id),
        ("model", body.model_id),
        ("judge", body.judge_model_id),
    ):
        if identifier is None:
            continue
        artifact = await db.get(Artifact, identifier)
        if not artifact or artifact.kind != ("model" if kind == "judge" else kind):
            raise NotFound(f"Не найдена версия артефакта: {kind}")
        snapshot[kind], snapshot[kind + "_hash"] = artifact.payload, artifact.content_hash
    created = await db.scalar(
        insert(Run)
        .values(id=body.request_id, request_hash=hashed, snapshot=snapshot)
        .on_conflict_do_nothing()
        .returning(Run.id)
    )
    row = await db.get(Run, body.request_id)
    if not created:
        if row.request_hash != hashed:
            raise Conflict("request_id уже использован с другими параметрами")
        return row, False
    for index, case in enumerate(snapshot["dataset"]["cases"]):
        for repeat in range(body.repeats):
            db.add(
                Sample(
                    run_id=row.id,
                    case_id=case["id"],
                    repeat=repeat,
                    seed=body.seed + index * body.repeats + repeat,
                )
            )
    await db.flush()
    return row, True


async def claim(sessions, settings):
    async with sessions.begin() as db:
        now = await db.scalar(select(func.clock_timestamp()))
        row = (
            await db.execute(
                select(Sample, Run.snapshot)
                .join(Run, Sample.run_id == Run.id)
                .where(
                    or_(
                        and_(Sample.status == "pending", Sample.available_at <= now),
                        and_(Sample.status == "running", Sample.lease_until <= now),
                    )
                )
                .order_by(Sample.available_at, Sample.id)
                .with_for_update(skip_locked=True, of=Sample)
                .limit(1)
            )
        ).first()
        if not row:
            return None
        sample, snapshot = row
        if (
            sample.attempts >= settings.max_attempts
            or snapshot["evaluator_version"] != EVALUATOR_VERSION
        ):
            sample.status = "error"
            sample.error = (
                "attempts_exhausted"
                if sample.attempts >= settings.max_attempts
                else "evaluator_version_changed"
            )
            sample.lease_token, sample.lease_until = None, None
            return None
        sample.status, sample.lease_token = "running", uuid4()
        sample.lease_until = now + timedelta(seconds=settings.lease_seconds)
        sample.attempts += 1
        return {
            "id": sample.id,
            "token": sample.lease_token,
            "seed": sample.seed,
            "snapshot": snapshot,
            "case": next(c for c in snapshot["dataset"]["cases"] if c["id"] == sample.case_id),
        }


async def finish(sessions, settings, job, result=None, error=None):
    async with sessions.begin() as db:
        now = await db.scalar(select(func.clock_timestamp()))
        sample = await db.scalar(
            select(Sample)
            .where(
                Sample.id == job["id"],
                Sample.status == "running",
                Sample.lease_token == job["token"],
                Sample.lease_until > now,
            )
            .with_for_update()
        )
        if sample is None:
            return False
        sample.lease_token, sample.lease_until = None, None
        if result is not None:
            sample.status, sample.result, sample.error = "done", result, None
        else:
            sample.error = error
            if sample.attempts >= settings.max_attempts or error == "model_digest_changed":
                sample.status = "error"
            else:
                sample.status = "pending"
                sample.available_at = now + timedelta(seconds=min(30, 2**sample.attempts))
        return True


async def report(db, identifier):
    run = await db.get(Run, identifier)
    if not run:
        raise NotFound("Прогон не найден")
    samples = (
        await db.scalars(
            select(Sample).where(Sample.run_id == run.id).order_by(Sample.case_id, Sample.repeat)
        )
    ).all()
    results = [
        {
            "id": str(s.id),
            "case_id": s.case_id,
            "repeat": s.repeat,
            "seed": s.seed,
            "status": s.status,
            "attempts": s.attempts,
            "error": s.error,
            "result": s.result,
        }
        for s in samples
    ]
    return {
        "id": str(run.id),
        "request_hash": run.request_hash,
        "snapshot": run.snapshot,
        "summary": summarize(run.snapshot, results),
        "results": results,
    }
