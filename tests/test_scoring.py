import copy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from gaugelab.schemas import Dataset, GatePolicy, Model, Prompt
from gaugelab.scoring import compare, percentile95, score, summarize


@pytest.mark.parametrize(
    "answer,expected,exact,f1",
    [
        ("ПАРИЖ!", "Париж", 1, 1),
        ("a a b", "a b b", 0, 2 / 3),
        ("", "answer", 0, 0),
        ("Ａ", "a", 1, 1),
    ],
)
def test_normalized_metrics(answer, expected, exact, f1):
    result = score(answer, {"expected": expected, "required_terms": [expected]})
    assert result["exact_match"] == exact
    assert result["token_f1"] == pytest.approx(f1)


def test_keywords_require_whole_tokens():
    assert score("котлета", {"expected": "кот", "required_terms": ["кот"]})["keyword_recall"] == 0
    assert score("Это кот.", {"expected": "кот", "required_terms": ["кот"]})["keyword_recall"] == 1


def test_p95_nearest_rank():
    assert percentile95(list(range(1, 101))) == 95
    assert percentile95([]) is None


def fixture_report(identifier="baseline", values=None, repeats=1):
    values = values or [1.0] * 10
    snapshot = {
        "dataset_hash": "same",
        "dataset": {"cases": [{"id": str(i)} for i in range(len(values))]},
        "evaluator_version": "v1",
        "config": {"repeats": repeats, "seed": 42},
        "judge": None,
    }
    results = [
        {
            "case_id": str(i),
            "status": "done",
            "attempts": 1,
            "result": {
                "metrics": {"exact_match": v, "token_f1": v, "keyword_recall": v},
                "latency_seconds": 0.5,
                "cost_lower_bound": "0.01",
                "usage_complete": True,
            },
        }
        for i, v in enumerate(values)
        for _ in range(repeats)
    ]
    return {
        "id": identifier,
        "snapshot": snapshot,
        "results": results,
        "summary": summarize(snapshot, results),
    }


def test_gate_pass_and_regression():
    baseline = fixture_report()
    assert compare(baseline, fixture_report("candidate"), GatePolicy())["decision"] == "passed"
    regressed = fixture_report("candidate", [0.0] * 10)
    gate = compare(baseline, regressed, GatePolicy())
    assert gate["decision"] == "blocked"
    paired = next(c for c in gate["checks"] if c["name"] == "regression")["detail"]["paired"]
    assert paired["losses"] == 10 and paired["ci95_high"] == -1


def test_repeated_trials_are_not_independent_cases():
    baseline = fixture_report(values=[1.0] * 2, repeats=5)
    candidate = fixture_report("candidate", values=[1.0] * 2, repeats=5)
    assert compare(baseline, candidate, GatePolicy(minimum_cases=5))["decision"] == "blocked"


@pytest.mark.parametrize("change", ["dataset", "seed", "unfinished", "same_id", "judge_missing"])
def test_gate_rejects_invalid_comparisons(change):
    baseline, candidate = fixture_report(), fixture_report("candidate")
    policy = GatePolicy()
    if change == "dataset":
        candidate["snapshot"]["dataset_hash"] = "different"
    if change == "seed":
        candidate["snapshot"]["config"]["seed"] = 99
    if change == "unfinished":
        candidate["summary"]["status"] = "running"
    if change == "same_id":
        candidate["id"] = "baseline"
    if change == "judge_missing":
        policy.metric = "judge_score"
    assert compare(baseline, candidate, policy)["decision"] == "blocked"


def test_errors_count_as_zero_and_block_gate():
    baseline, candidate = fixture_report(), fixture_report("candidate")
    candidate["results"][0].update(status="error", result=None)
    candidate["summary"] = summarize(candidate["snapshot"], candidate["results"])
    assert candidate["summary"]["metrics"]["exact_match"] == 0.9
    assert compare(baseline, candidate, GatePolicy())["decision"] == "blocked"


def test_cost_unknown_after_retry_and_missing_usage():
    baseline, candidate = fixture_report(), fixture_report("candidate")
    candidate["results"][0]["attempts"] = 2
    candidate["summary"] = summarize(candidate["snapshot"], candidate["results"])
    assert candidate["summary"]["cost_complete"] is False
    assert (
        compare(baseline, candidate, GatePolicy(maximum_cost=Decimal("10")))["decision"]
        == "blocked"
    )
    candidate["results"][0]["attempts"] = 1
    candidate["results"][0]["result"]["usage_complete"] = False
    assert not summarize(candidate["snapshot"], candidate["results"])["cost_complete"]


def test_latency_and_cost_limits():
    baseline, candidate = fixture_report(), fixture_report("candidate")
    assert (
        compare(baseline, candidate, GatePolicy(maximum_p95_seconds=0.1))["decision"] == "blocked"
    )
    assert (
        compare(baseline, candidate, GatePolicy(maximum_cost=Decimal("0.01")))["decision"]
        == "blocked"
    )


def test_bootstrap_is_reproducible_and_does_not_mutate_reports():
    b, c = fixture_report(), fixture_report("candidate", [1] * 9 + [0.5])
    saved = copy.deepcopy(c)
    assert compare(b, c, GatePolicy()) == compare(b, c, GatePolicy())
    assert c == saved


@pytest.mark.parametrize("template", ["Нет вопроса", "{question} {question}"])
def test_invalid_prompt(template):
    with pytest.raises(ValidationError):
        Prompt(name="x", version="1", template=template)


def test_model_requires_digest():
    with pytest.raises(ValidationError):
        Model(name="x", version="1", backend="ollama", model_name="model:tag")


def test_duplicate_dataset_cases_rejected():
    case = {"id": "x", "question": "q", "expected": "a", "required_terms": ["a"]}
    with pytest.raises(ValidationError):
        Dataset(name="x", version="1", cases=[case, case])
