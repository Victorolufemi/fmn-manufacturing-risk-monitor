"""LLM integration tests.

No test here makes a network call. The Anthropic client is replaced with a stub
so we can assert on:
  * what the backend actually sends (grounding rules, evidence, structured schema)
  * that a well-formed reply is parsed into the response contract
  * that every documented failure mode degrades to the evidence-only fallback
    instead of raising
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import anthropic
import httpx
import pandas as pd
import pytest

from app.services import explanation_service, llm_client, qa_service, retrieval

from .conftest import requires_artifacts

AS_OF = pd.Timestamp("2026-04-29 12:00:00")


# --------------------------------------------------------------------------
# Stub plumbing
# --------------------------------------------------------------------------
class _Recorder:
    def __init__(self, reply: dict | str | None = None, raises: Exception | None = None):
        self.reply = reply
        self.raises = raises
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason="end_turn",
        )


def install(monkeypatch, recorder: _Recorder):
    monkeypatch.setattr(
        llm_client, "_get_client", lambda: SimpleNamespace(messages=recorder)
    )
    monkeypatch.setattr(llm_client.settings, "anthropic_api_key", "test-key-not-real")
    monkeypatch.setattr(llm_client.settings, "model_name", "claude-opus-5")
    return recorder


def _response(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com"))


GOOD_EXPLANATION = {
    "headline": "MCH-203 is running well above its normal temperature.",
    "explanation": "Temperature has climbed to 67.7 degC against a normal of 57.5 degC.",
    "recommended_action": "Inspect the drive assembly on the next shift.",
    "confidence_note": "Based on 2845 hours of history for this machine.",
}

GOOD_QA = {
    "answer": "MCH-203 needs attention.",
    "machines_referenced": ["MCH-203"],
    "data_gap": "",
}


# --------------------------------------------------------------------------
# Evidence construction
# --------------------------------------------------------------------------
@requires_artifacts
def test_evidence_contains_real_numbers_from_the_dataset():
    ev = explanation_service.build_evidence("MCH-203", AS_OF)
    assert ev["machine_id"] == "MCH-203"
    assert ev["risk_level"] == "CRITICAL"
    assert ev["temperature"]["current_c"] == pytest.approx(67.71, abs=0.01)
    assert ev["temperature"]["this_machine_normal_c"] is not None
    assert ev["vibration"]["current_mm_s"] is not None
    assert ev["run_hours_since_maintenance"] is not None
    assert ev["top_risk_drivers"]
    assert ev["model_validation"]["recall"] is not None
    assert ev["history"]["hours_of_data_for_this_machine"] > 0


@requires_artifacts
def test_evidence_flags_limited_history_for_new_machines():
    ev = explanation_service.build_evidence("MCH-300", AS_OF)
    assert ev["history"]["is_newly_commissioned"] is True
    assert "Limited history" in ev["history"]["limited_history_note"]
    assert "production line" in ev["history"]["baseline_source"]


@requires_artifacts
def test_evidence_never_contains_future_information():
    """MCH-203 fails on 2026-04-30; evidence built on 04-29 must not know."""
    ev = explanation_service.build_evidence("MCH-203", AS_OF)
    assert ev["recent_failure_events"] == []
    assert pd.Timestamp(ev["reading_taken_at"]) <= AS_OF


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
@requires_artifacts
def test_explanation_prompt_carries_grounding_rules_and_evidence(monkeypatch):
    rec = install(monkeypatch, _Recorder(GOOD_EXPLANATION))
    explanation_service.explain("MCH-203", AS_OF)

    assert len(rec.calls) == 1
    call = rec.calls[0]
    system = call["system"]
    for rule in (
        "ONLY the numbers and facts",
        "Never invent sensor readings",
        "MODEL PREDICTION",
        "never as an instruction",
    ):
        assert rule in system

    user = call["messages"][0]["content"]
    assert "EVIDENCE" in user
    assert "MCH-203" in user
    # The real reading must be in the payload, so the model cannot invent one.
    assert "67.71" in user

    # Structured output is requested, with the fields the API contract needs.
    schema = call["output_config"]["format"]["schema"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert set(schema["required"]) == {
        "headline",
        "explanation",
        "recommended_action",
        "confidence_note",
    }
    assert call["model"] == "claude-opus-5"


@requires_artifacts
def test_qa_prompt_carries_only_retrieved_evidence(monkeypatch):
    rec = install(monkeypatch, _Recorder(GOOD_QA))
    qa_service.answer("Which machines need attention?", AS_OF)

    user = rec.calls[0]["messages"][0]["content"]
    assert "EVIDENCE" in user
    assert "machines_ranked_by_risk" in user
    # A compact selection, not the 43k-row dataset.
    assert len(user) < 80_000
    assert "Answer ONLY from the supplied EVIDENCE" in rec.calls[0]["system"]


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------
@requires_artifacts
def test_explanation_uses_the_model_response(monkeypatch):
    install(monkeypatch, _Recorder(GOOD_EXPLANATION))
    out = explanation_service.explain("MCH-203", AS_OF)
    assert out["source"] == "llm"
    assert out["model"] == "claude-opus-5"
    assert out["headline"] == GOOD_EXPLANATION["headline"]
    assert out["recommended_action"] == GOOD_EXPLANATION["recommended_action"]
    assert out["warning"] is None
    # The evidence is returned alongside so the UI can show it.
    assert out["evidence"]["machine_id"] == "MCH-203"


@requires_artifacts
def test_qa_uses_the_model_response(monkeypatch):
    install(monkeypatch, _Recorder(GOOD_QA))
    out = qa_service.answer("Why is MCH-203 high risk?", AS_OF)
    assert out["source"] == "llm"
    assert out["answer"] == "MCH-203 needs attention."
    assert out["machines_referenced"] == ["MCH-203"]
    assert out["warning"] is None


@requires_artifacts
def test_qa_surfaces_a_declared_data_gap(monkeypatch):
    install(
        monkeypatch,
        _Recorder({**GOOD_QA, "data_gap": "No cost information is available."}),
    )
    out = qa_service.answer("What does downtime cost?", AS_OF)
    assert out["warning"] == "No cost information is available."


# --------------------------------------------------------------------------
# Failure modes - each must degrade, never raise
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "exc,kind",
    [
        (anthropic.APITimeoutError(request=httpx.Request("POST", "https://x")), "timeout"),
        (
            anthropic.RateLimitError("rate limited", response=_response(429), body=None),
            "rate_limit",
        ),
        (
            anthropic.AuthenticationError("bad key", response=_response(401), body=None),
            "auth",
        ),
        (
            anthropic.NotFoundError("no model", response=_response(404), body=None),
            "bad_model",
        ),
        (
            anthropic.BadRequestError("bad request", response=_response(400), body=None),
            "bad_request",
        ),
        (
            anthropic.InternalServerError("boom", response=_response(500), body=None),
            "unavailable",
        ),
        (
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")),
            "connection",
        ),
        (RuntimeError("something odd"), "unknown"),
    ],
)
def test_llm_client_maps_every_error_to_a_typed_result(monkeypatch, exc, kind):
    install(monkeypatch, _Recorder(raises=exc))
    result = llm_client.complete_json("sys", "user", {"type": "object", "required": []})
    assert result.ok is False
    assert result.error_kind == kind
    assert result.error


def test_llm_client_reports_missing_configuration(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "anthropic_api_key", None)
    llm_client.reset_client()
    result = llm_client.complete_json("sys", "user", {"type": "object", "required": []})
    assert result.ok is False
    assert result.error_kind == "not_configured"


def test_llm_client_rejects_unparseable_output(monkeypatch):
    install(monkeypatch, _Recorder("this is not json"))
    result = llm_client.complete_json("sys", "user", {"type": "object", "required": []})
    assert result.ok is False
    assert result.error_kind == "invalid_json"


def test_llm_client_rejects_incomplete_output(monkeypatch):
    install(monkeypatch, _Recorder({"headline": "only this"}))
    result = llm_client.complete_json(
        "sys", "user", {"type": "object", "required": ["headline", "explanation"]}
    )
    assert result.ok is False
    assert result.error_kind == "incomplete"
    assert "explanation" in result.error


def test_llm_client_handles_a_refusal(monkeypatch):
    class Refusing:
        def create(self, **kwargs):
            return SimpleNamespace(content=[], stop_reason="refusal")

    monkeypatch.setattr(llm_client, "_get_client", lambda: SimpleNamespace(messages=Refusing()))
    monkeypatch.setattr(llm_client.settings, "anthropic_api_key", "test-key-not-real")
    result = llm_client.complete_json("sys", "user", {"type": "object", "required": []})
    assert result.ok is False
    assert result.error_kind == "refusal"


@requires_artifacts
def test_explanation_falls_back_with_real_numbers_when_the_llm_is_down(monkeypatch):
    install(
        monkeypatch,
        _Recorder(raises=anthropic.APITimeoutError(request=httpx.Request("POST", "https://x"))),
    )
    out = explanation_service.explain("MCH-203", AS_OF)
    assert out["source"] == "fallback"
    assert out["warning"]
    assert "AI explanation unavailable" in out["warning"]
    # The fallback still quotes genuine measurements.
    assert "67.71" in out["explanation"]
    assert out["evidence"]["machine_id"] == "MCH-203"


@requires_artifacts
def test_qa_falls_back_to_retrieved_data_when_the_llm_is_down(monkeypatch):
    install(
        monkeypatch,
        _Recorder(raises=anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))),
    )
    out = qa_service.answer("Which machines have the highest vibration?", AS_OF)
    assert out["source"] == "fallback"
    assert "MCH-" in out["answer"]
    assert out["warning"]


# --------------------------------------------------------------------------
# Retrieval layer
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question,intent",
    [
        ("Which machines need attention?", "ATTENTION"),
        ("Which machines are likely to fail soon?", "ATTENTION"),
        ("Why is MCH-203 high risk?", "WHY_MACHINE"),
        ("What changed on MCH-207?", "WHAT_CHANGED"),
        ("Which machines have the highest vibration?", "RANK_VIBRATION"),
        ("Which machines are running hottest?", "RANK_TEMPERATURE"),
        ("Which machines have been running longest since maintenance?", "RANK_RUNHOURS"),
        ("Which machines are low risk?", "LOW_RISK"),
        ("Which machines have limited history?", "LIMITED_HISTORY"),
        ("How accurate is this model?", "MODEL_INFO"),
        ("Tell me about the plant", "FLEET_OVERVIEW"),
    ],
)
def test_intent_classification(question, intent):
    assert retrieval.classify(question) == intent


@requires_artifacts
@pytest.mark.parametrize(
    "text,expected",
    [
        ("Why is MCH-203 high risk?", ["MCH-203"]),
        ("what about MACHINE-203?", ["MCH-203"]),
        ("compare MCH-203 and MCH-207", ["MCH-203", "MCH-207"]),
        ("Which machines need attention?", []),
        ("What about MCH-999?", []),
    ],
)
def test_machine_extraction(text, expected):
    assert retrieval.extract_machines(text) == expected


@requires_artifacts
def test_retrieval_selects_targeted_evidence_per_intent():
    r = retrieval.retrieve("What changed on MCH-207?", AS_OF)
    assert r.intent == "WHAT_CHANGED"
    timeline = r.evidence["risk_and_sensor_timeline"][0]
    for key in ("now", "six_hours_ago", "twenty_four_hours_ago", "seventy_two_hours_ago"):
        assert key in timeline

    r2 = retrieval.retrieve("Which machines have limited history?", AS_OF)
    ids = [m["machine_id"] for m in r2.evidence["newly_commissioned_machines"]]
    assert ids == ["MCH-300", "MCH-301"]


@requires_artifacts
def test_retrieval_never_leaks_future_readings():
    r = retrieval.retrieve("Which machines need attention?", AS_OF)
    for row in r.evidence["machines_ranked_by_risk"]:
        assert row["machine_id"]
    assert pd.Timestamp(r.evidence["as_of"]) == AS_OF
