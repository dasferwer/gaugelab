"""Проверяет, что основной цикл процесса недавно обновлял отметку времени."""

import time
from pathlib import Path

from gaugelab.config import Settings

assert (
    time.time() - float(Path("/tmp/gaugelab-heartbeat").read_text())
    < 2 * Settings().request_timeout + 10
)
