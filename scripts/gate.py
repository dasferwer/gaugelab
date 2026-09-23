"""Возвращает код 0 при успешной проверке, 2 при регрессии и 1 при технической ошибке."""

import argparse
import json
import os
import sys

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("baseline")
parser.add_argument("candidate")
parser.add_argument("--url", default="http://127.0.0.1:8260")
parser.add_argument("--max-drop", type=float, default=0.05)
args = parser.parse_args()
try:
    with httpx.Client(timeout=30, trust_env=False) as client:
        response = client.post(
            args.url + "/v1/gates",
            headers={"Authorization": "Bearer " + os.environ["ADMIN_TOKEN"]},
            json={
                "baseline_id": args.baseline,
                "candidate_id": args.candidate,
                "policy": {"allowed_regression": args.max_drop},
            },
        )
        response.raise_for_status()
        result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["decision"] == "passed" else 2)
except (httpx.HTTPError, KeyError, ValueError):
    print("Не удалось выполнить проверку регрессии", file=sys.stderr)
    sys.exit(1)
