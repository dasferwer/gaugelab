import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal

EVALUATOR_VERSION = "gaugelab-v1"
JUDGE_RUBRIC = "judge-v1"
METRICS = ("exact_match", "token_f1", "keyword_recall", "judge_score")


def tokens(value):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold(), flags=re.UNICODE)


def score(answer, case):
    actual, expected = tokens(answer), tokens(case["expected"])
    common = sum((Counter(actual) & Counter(expected)).values())
    f1 = 2 * common / (len(actual) + len(expected)) if actual or expected else 1.0
    # Проверяем фразы по целым токенам: слово «кот» не должно совпадать со словом «котлета».
    haystack = " " + " ".join(actual) + " "
    hits = sum(
        bool(tokens(term)) and (" " + " ".join(tokens(term)) + " ") in haystack
        for term in case["required_terms"]
    )
    return {
        "exact_match": float(actual == expected),
        "token_f1": f1,
        "keyword_recall": hits / len(case["required_terms"]),
    }


def percentile95(values):
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)] if values else None


def summarize(snapshot, samples):
    successful = [s for s in samples if s["status"] == "done"]
    complete = all(s["status"] in ("done", "error") for s in samples)
    metrics = {}
    for metric in METRICS:
        if metric == "judge_score" and snapshot.get("judge") is None:
            metrics[metric] = None
            continue
        metrics[metric] = statistics.mean(
            s["result"]["metrics"].get(metric, 0) if s["status"] == "done" else 0 for s in samples
        )
    variability = {}
    for metric in METRICS:
        if snapshot["config"]["repeats"] < 2 or not complete or metrics[metric] is None:
            variability[metric] = None
            continue
        grouped = defaultdict(list)
        for sample in samples:
            value = sample["result"]["metrics"].get(metric, 0) if sample["status"] == "done" else 0
            grouped[sample["case_id"]].append(value)
        variability[metric] = statistics.mean(
            statistics.pstdev(values) for values in grouped.values()
        )
    costs = [Decimal(s["result"]["cost_lower_bound"]) for s in successful]
    cost_complete = (
        complete
        and len(successful) == len(samples)
        and all(s["attempts"] == 1 and s["result"]["usage_complete"] for s in successful)
    )
    return {
        "status": "completed" if complete else "running",
        "samples": len(samples),
        "successful": len(successful),
        "failed": sum(s["status"] == "error" for s in samples),
        "metrics": metrics,
        "mean_repeat_stddev": variability,
        "p95_seconds": percentile95([s["result"]["latency_seconds"] for s in successful]),
        "cost_lower_bound": str(sum(costs, Decimal(0))),
        "cost_complete": cost_complete,
    }


def compare(baseline, candidate, policy):
    checks = []

    def check(name, passed, detail):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    b, c = baseline["summary"], candidate["summary"]
    bs, cs = baseline["snapshot"], candidate["snapshot"]
    check("different_runs", baseline["id"] != candidate["id"], "Сравниваются два разных прогона")
    check("complete", b["status"] == c["status"] == "completed", "Оба прогона должны завершиться")
    check(
        "no_errors", b["failed"] == c["failed"] == 0, "Ошибки провайдера не исключаются из оценки"
    )
    compatible = (
        bs["dataset_hash"] == cs["dataset_hash"]
        and bs["evaluator_version"] == cs["evaluator_version"]
        and bs["config"]["repeats"] == cs["config"]["repeats"]
        and bs["config"]["seed"] == cs["config"]["seed"]
    )
    if policy.metric == "judge_score":
        compatible = (
            compatible
            and bs.get("judge_hash") == cs.get("judge_hash")
            and bs.get("judge") is not None
        )
    check(
        "compatible",
        compatible,
        "Совпадают датасет, версия оценки, повторы и seed; для судьи — также его версия",
    )
    check(
        "minimum_cases",
        len(cs["dataset"]["cases"]) >= policy.minimum_cases,
        "Порог применяется к вопросам, а не к числу повторов",
    )
    value = c["metrics"][policy.metric]
    check(
        "minimum_score",
        value is not None and value >= policy.minimum_score,
        {"observed": value, "minimum": policy.minimum_score},
    )
    paired = None
    if compatible and b["status"] == c["status"] == "completed":
        by_case = []
        case_values = []
        for report in (baseline, candidate):
            grouped = defaultdict(list)
            for sample in report["results"]:
                metric = (
                    sample["result"]["metrics"].get(policy.metric, 0)
                    if sample["status"] == "done"
                    else 0
                )
                grouped[sample["case_id"]].append(metric)
            case_values.append({key: statistics.mean(values) for key, values in grouped.items()})
        for key in sorted(case_values[0]):
            by_case.append(case_values[1][key] - case_values[0][key])
        rng = random.Random(2026)
        estimates = sorted(
            statistics.mean(rng.choices(by_case, k=len(by_case))) for _ in range(1000)
        )
        paired = {
            "mean_delta": statistics.mean(by_case),
            "ci95_low": estimates[24],
            "ci95_high": estimates[974],
            "wins": sum(d > 1e-12 for d in by_case),
            "ties": sum(abs(d) <= 1e-12 for d in by_case),
            "losses": sum(d < -1e-12 for d in by_case),
            "unit": "case",
            "bootstrap_seed": 2026,
        }
    check(
        "regression",
        paired is not None and paired["ci95_low"] >= -policy.allowed_regression,
        {"paired": paired, "allowed_regression": policy.allowed_regression},
    )
    if policy.maximum_p95_seconds is not None:
        check(
            "latency",
            c["p95_seconds"] is not None and c["p95_seconds"] <= policy.maximum_p95_seconds,
            {"observed": c["p95_seconds"], "maximum": policy.maximum_p95_seconds},
        )
    if policy.maximum_cost is not None:
        check(
            "cost",
            c["cost_complete"] and Decimal(c["cost_lower_bound"]) <= policy.maximum_cost,
            {
                "observed_lower_bound": c["cost_lower_bound"],
                "complete": c["cost_complete"],
                "maximum": str(policy.maximum_cost),
            },
        )
    return {
        "decision": "passed" if all(c["passed"] for c in checks) else "blocked",
        "checks": checks,
        "policy": policy.model_dump(mode="json"),
        "evaluator_version": EVALUATOR_VERSION,
    }
