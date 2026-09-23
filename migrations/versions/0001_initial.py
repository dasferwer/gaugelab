"""Создаёт таблицы версий артефактов, прогонов, заданий и результатов проверки регрессий."""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

DDL = """
CREATE TABLE artifacts (id UUID PRIMARY KEY, kind VARCHAR(20) NOT NULL, name VARCHAR(80) NOT NULL,
 version VARCHAR(40) NOT NULL, content_hash VARCHAR(64) NOT NULL, payload JSON NOT NULL,
 CONSTRAINT uq_artifact_version UNIQUE (kind, name, version));
CREATE TABLE runs (id UUID PRIMARY KEY, request_hash VARCHAR(64) NOT NULL, snapshot JSON NOT NULL,
 created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE samples (id UUID PRIMARY KEY, run_id UUID NOT NULL REFERENCES runs(id), case_id VARCHAR(80) NOT NULL,
 repeat INTEGER NOT NULL, seed INTEGER NOT NULL, status VARCHAR(20) NOT NULL, attempts INTEGER NOT NULL,
 lease_token UUID, lease_until TIMESTAMPTZ, available_at TIMESTAMPTZ NOT NULL DEFAULT now(), result JSON, error VARCHAR(80),
 CONSTRAINT uq_run_sample UNIQUE (run_id, case_id, repeat),
 CONSTRAINT ck_sample_status CHECK (status IN ('pending','running','done','error')));
CREATE INDEX ix_samples_run_id ON samples (run_id);
CREATE INDEX ix_samples_work ON samples (status, available_at);
CREATE TABLE gates (id UUID PRIMARY KEY, baseline_id UUID NOT NULL REFERENCES runs(id),
 candidate_id UUID NOT NULL REFERENCES runs(id), result JSON NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now());
"""


def upgrade():
    for statement in DDL.split(";"):
        if statement.strip():
            op.execute(statement)


def downgrade():
    for table in ("gates", "samples", "runs", "artifacts"):
        op.drop_table(table)
