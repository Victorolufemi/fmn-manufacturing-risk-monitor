"""API contract tests."""
from __future__ import annotations

import pandas as pd

from .conftest import requires_artifacts

AS_OF = "2026-04-29T12:00:00"


def test_health_always_responds(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "llm_configured" in body
    # The health endpoint must never leak the key itself.
    assert "anthropic_api_key" not in str(body).lower()


@requires_artifacts
def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["machines"] == 17
    assert body["prediction_horizon_hours"] > 0


@requires_artifacts
def test_metadata(client):
    body = client.get("/api/metadata").json()
    assert len(body["machines"]) == 17
    assert body["cold_start_machines"] == ["MCH-300", "MCH-301"]
    assert body["risk_bands"]["medium"] < body["risk_bands"]["high"]
    assert len(body["suggested_questions"]) >= 3


@requires_artifacts
def test_model_info_exposes_metrics_but_no_secrets(client):
    body = client.get("/api/model-info").json()
    assert body["model_name"]
    assert body["headline_metrics"]["test_recall"] is not None
    assert body["validation"]["test_failure_episodes"] == 7
    blob = str(body).lower()
    assert "api_key" not in blob
    assert "sk-ant" not in blob


@requires_artifacts
def test_dashboard_shape_and_counts(client):
    r = client.get(f"/api/dashboard?as_of={AS_OF}")
    assert r.status_code == 200
    d = r.json()
    c = d["counts"]
    assert c["total_machines"] == 17
    assert c["critical"] + c["high"] + c["medium"] + c["low"] == 17
    assert c["newly_commissioned"] == 2
    assert len(d["machines"]) == 17
    # Ranked by risk, descending.
    scores = [m["risk_score"] for m in d["machines"]]
    assert scores == sorted(scores, reverse=True)
    assert d["attention_required"], "attention list is never empty"


@requires_artifacts
def test_dashboard_never_shows_data_after_the_review_time(client):
    d = client.get(f"/api/dashboard?as_of={AS_OF}").json()
    cutoff = pd.Timestamp(AS_OF)
    for m in d["machines"]:
        assert pd.Timestamp(m["timestamp"]) <= cutoff


@requires_artifacts
def test_dashboard_defaults_to_latest_timestamp(client):
    d = client.get("/api/dashboard").json()
    assert d["as_of"] == d["data_range"]["end"]


@requires_artifacts
def test_dashboard_rejects_a_bad_timestamp(client):
    assert client.get("/api/dashboard?as_of=not-a-date").status_code == 400


@requires_artifacts
def test_machines_filters(client):
    base = client.get(f"/api/machines?as_of={AS_OF}").json()
    assert len(base) == 17

    new_only = client.get(f"/api/machines?as_of={AS_OF}&cohort=new").json()
    assert {m["machine_id"] for m in new_only} == {"MCH-300", "MCH-301"}

    established = client.get(f"/api/machines?as_of={AS_OF}&cohort=established").json()
    assert len(established) == 15

    crit = client.get(f"/api/machines?as_of={AS_OF}&risk_level=CRITICAL").json()
    assert all(m["risk_level"] == "CRITICAL" for m in crit)

    searched = client.get(f"/api/machines?as_of={AS_OF}&search=203").json()
    assert [m["machine_id"] for m in searched] == ["MCH-203"]


@requires_artifacts
def test_machine_detail_contains_everything_the_page_needs(client):
    d = client.get(f"/api/machines/MCH-203?as_of={AS_OF}&history_hours=168").json()
    assert d["machine_id"] == "MCH-203"
    assert d["risk_level"] == "CRITICAL"
    assert d["prediction_horizon_hours"] > 0
    assert len(d["history"]) == 168
    assert d["temperature"]["baseline"] is not None
    assert d["vibration"]["baseline"] is not None
    assert d["drivers"], "drivers must be computed from the model"
    assert d["cold_start"]["is_cold_start"] is False
    # History is ordered and bounded by the review time.
    stamps = [pd.Timestamp(p["timestamp"]) for p in d["history"]]
    assert stamps == sorted(stamps)
    assert stamps[-1] <= pd.Timestamp(AS_OF)


@requires_artifacts
def test_machine_detail_drivers_are_ranked_and_bounded(client):
    d = client.get(f"/api/machines/MCH-203?as_of={AS_OF}").json()
    importances = [x["importance"] for x in d["drivers"]]
    assert importances == sorted(importances, reverse=True)
    assert all(0.0 <= v <= 1.0 for v in importances)
    for x in d["drivers"]:
        assert 0.0 <= x["risk_if_normal"] <= 1.0
        assert x["label"] and x["description"]


@requires_artifacts
def test_cold_start_machine_carries_a_warning(client):
    d = client.get(f"/api/machines/MCH-300?as_of={AS_OF}").json()
    assert d["cold_start"]["is_cold_start"] is True
    assert d["cold_start"]["history_hours"] <= 72
    assert "Limited history" in d["cold_start"]["message"]
    assert d["cold_start"]["baseline_source"] == "line_pooled"


@requires_artifacts
def test_unknown_machine_returns_404(client):
    r = client.get("/api/machines/MCH-999")
    assert r.status_code == 404
    assert "MCH-999" in r.json()["detail"]


@requires_artifacts
def test_explanation_endpoint_returns_grounded_evidence(client):
    d = client.get(f"/api/machines/MCH-203/explanation?as_of={AS_OF}").json()
    assert d["machine_id"] == "MCH-203"
    assert d["source"] in ("llm", "fallback")
    assert d["headline"] and d["explanation"] and d["recommended_action"]
    ev = d["evidence"]
    assert ev["machine_id"] == "MCH-203"
    assert ev["risk_level"] == "CRITICAL"
    assert ev["temperature"]["current_c"] is not None
    assert ev["vibration"]["this_machine_normal_mm_s"] is not None
    assert ev["top_risk_drivers"]


@requires_artifacts
def test_explanation_for_unknown_machine_returns_404(client):
    assert client.get("/api/machines/NOPE/explanation").status_code == 404


@requires_artifacts
def test_qa_returns_an_answer_and_its_evidence(client):
    r = client.post(
        "/api/qa", json={"question": "Which machines need attention?", "as_of": AS_OF}
    )
    assert r.status_code == 200
    d = r.json()
    assert d["intent"] == "ATTENTION"
    assert d["answer"].strip()
    assert "machines_ranked_by_risk" in d["evidence"]


@requires_artifacts
def test_qa_routes_machine_specific_questions(client):
    d = client.post(
        "/api/qa", json={"question": "Why is MCH-203 high risk?", "as_of": AS_OF}
    ).json()
    assert d["intent"] == "WHY_MACHINE"
    assert "MCH-203" in d["machines_referenced"]
    assert d["evidence"]["machines"][0]["machine_id"] == "MCH-203"


@requires_artifacts
def test_qa_validates_input(client):
    assert client.post("/api/qa", json={"question": ""}).status_code == 422
    assert client.post("/api/qa", json={}).status_code == 422


@requires_artifacts
def test_qa_evidence_is_a_selection_not_the_whole_dataset(client):
    """The retrieval layer must not ship 43k rows to the model."""
    import json

    d = client.post(
        "/api/qa", json={"question": "Which machines need attention?", "as_of": AS_OF}
    ).json()
    payload = json.dumps(d["evidence"])
    assert len(payload) < 60_000
    assert "machines_ranked_by_risk" in d["evidence"]
    assert len(d["evidence"]["machines_ranked_by_risk"]) <= 17


def test_cors_allows_the_configured_frontend(client):
    r = client.options(
        "/api/dashboard",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code in (200, 204)
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_rejects_an_unconfigured_origin(client):
    r = client.options(
        "/api/dashboard",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") != "https://evil.example.com"

def test_cors_allows_next_fallback_ports_in_development():
    """Next.js drifts to 3001/3002 when 3000 is taken; dev must tolerate that."""
    from app.config import Settings

    dev = Settings(environment="development", frontend_url="http://localhost:3000")
    assert "http://localhost:3001" in dev.cors_origins
    assert "http://127.0.0.1:3002" in dev.cors_origins


def test_cors_in_production_is_pinned_to_frontend_url():
    """No localhost fallbacks may leak into a production allowlist."""
    from app.config import Settings

    prod = Settings(environment="production", frontend_url="https://my-app.vercel.app")
    assert prod.cors_origins == ["https://my-app.vercel.app"]


def test_cors_accepts_a_comma_separated_list():
    from app.config import Settings

    s = Settings(environment="production", frontend_url="https://a.app, https://b.app/")
    assert s.cors_origins == ["https://a.app", "https://b.app"]
