"""Data-selection layer for grounded Q&A.

The CSV is never sent to the LLM. Instead a question is classified into an
intent, the machines it refers to are resolved, and a compact evidence bundle is
assembled from the scored history - typically a few hundred numbers rather than
43,000 rows.

Intent detection is deliberately rule-based: it is deterministic, auditable,
costs nothing, and its failure mode is a broader retrieval rather than a wrong
answer. When no rule matches, the fleet overview is retrieved, which is enough
context to answer most general questions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from app.ml.constants import COL_TIMESTAMP
from app.services import data_service as ds
from app.services import machine_service as ms

MACHINE_RE = re.compile(r"\b(?:MCH|MACHINE)[\s_-]?(\d{2,4})\b", re.IGNORECASE)

INTENTS = {
    "ATTENTION": [
        "need attention", "needs attention", "require attention", "at risk",
        "which machines are risky", "what should i look at", "highest risk",
        "most at risk", "worry", "urgent", "priority", "act on", "check first",
        "likely to fail", "going to fail", "fail soon",
    ],
    "WHY_MACHINE": ["why is", "why does", "why has", "reason", "explain", "what is wrong with"],
    "WHAT_CHANGED": ["what changed", "what has changed", "changed on", "recently changed",
                     "what happened"],
    "RISK_INCREASED": ["risk increase", "risk increased", "risk go up", "risk rise",
                       "getting worse", "trending up", "rising", "why did the risk"],
    "RANK_VIBRATION": ["highest vibration", "most vibration", "vibrating", "vibration ranking",
                       "worst vibration"],
    "RANK_TEMPERATURE": ["highest temperature", "hottest", "running hot", "temperature ranking",
                         "warmest"],
    "RANK_RUNHOURS": ["longest since maintenance", "running longest", "most run hours",
                      "overdue", "run hours", "since maintenance", "due for maintenance",
                      "due maintenance"],
    "LOW_RISK": ["low risk", "which machines are fine", "healthy", "all clear", "safe"],
    "LIMITED_HISTORY": ["limited history", "new machine", "newly commissioned", "little history",
                        "just installed", "recently installed", "cold start"],
    "MODEL_INFO": ["how accurate", "how good is the model", "how does this work", "model performance",
                   "how reliable", "what does the score mean", "how is risk calculated",
                   "prediction horizon"],
}


@dataclass
class Retrieval:
    intent: str
    machines: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    description: str = ""


def _normalise_machine_id(token: str) -> str | None:
    art = ds.load_artifacts()
    for candidate in (f"MCH-{token}", token, f"MACHINE-{token}"):
        if candidate in art.machine_ids:
            return candidate
    return None


def extract_machines(question: str) -> list:
    found = []
    for m in MACHINE_RE.finditer(question):
        mid = _normalise_machine_id(m.group(1))
        if mid and mid not in found:
            found.append(mid)
    return found


def classify(question: str) -> str:
    q = question.lower()
    scores = {}
    for intent, phrases in INTENTS.items():
        hits = sum(1 for p in phrases if p in q)
        if hits:
            scores[intent] = hits
    if not scores:
        return "FLEET_OVERVIEW"
    # Machine-specific intents win when a machine is named.
    if extract_machines(question):
        for preferred in ("WHAT_CHANGED", "RISK_INCREASED", "WHY_MACHINE"):
            if preferred in scores:
                return preferred
    return max(scores, key=lambda k: (scores[k], k))


# --------------------------------------------------------------------------
# Evidence builders - each returns only what the intent needs
# --------------------------------------------------------------------------
def _machine_card(machine_id: str, as_of: pd.Timestamp, deep: bool = False) -> dict:
    d = ms.machine_detail(machine_id, as_of, history_hours=72)
    card = {
        "machine_id": machine_id,
        "production_line": d["line"],
        "risk_score": round(d["risk_score"], 3),
        "risk_level": d["risk_level"],
        "risk_trend": d["trend"]["direction"],
        "risk_change_over_24h": d["trend"]["delta"],
        "temperature_c": d["temperature"]["current"],
        "temperature_normal_for_this_machine_c": d["temperature"]["baseline"],
        "temperature_change_24h_c": d["temperature"]["change_24h"],
        "vibration_mm_s": d["vibration"]["current"],
        "vibration_normal_for_this_machine_mm_s": d["vibration"]["baseline"],
        "vibration_change_24h_mm_s": d["vibration"]["change_24h"],
        "run_hours_since_maintenance": d["run_hours_since_maintenance"],
        "hours_of_history": d["cold_start"]["history_hours"],
        "is_newly_commissioned": d["cold_start"]["is_cold_start"],
        "limited_history_note": d["cold_start"]["message"],
        "reading_taken_at": d["timestamp"],
    }
    if deep:
        card["top_risk_drivers"] = [
            {
                "driver": x["label"],
                "share_of_risk": x["importance"],
                "risk_if_this_were_normal": x["risk_if_normal"],
                "readings": [
                    {
                        "name": e["label"],
                        "current_value": e["value"],
                        "typical_value_for_this_machine": e["typical_value"],
                    }
                    for e in x["evidence"]
                ],
            }
            for x in d["drivers"]
        ]
        card["failure_events_on_record"] = d["failure_history"][-5:]
        card["maintenance_events_on_record"] = d["maintenance_history"][-5:]
    return card


def _sensor_change_card(machine_id: str, as_of: pd.Timestamp) -> dict:
    """Now vs 6h vs 24h vs 72h ago - enough to answer 'what changed'."""
    hist = ds.machine_history(machine_id, as_of, hours=73)
    if hist.empty:
        return {"machine_id": machine_id, "error": "no readings available"}
    hist = hist.set_index(COL_TIMESTAMP)
    latest_ts = hist.index[-1]

    def at(hours_ago: int):
        target = latest_ts - pd.Timedelta(hours=hours_ago)
        sub = hist[hist.index <= target]
        if sub.empty:
            return None
        r = sub.iloc[-1]
        return {
            "at": r.name.isoformat(),
            "temperature_c": round(float(r["temperature_c"]), 2),
            "vibration_mm_s": round(float(r["vibration_mm_s"]), 3),
            "risk_score": round(float(r["risk_score"]), 3),
            "risk_level": r["risk_level"],
        }

    return {
        "machine_id": machine_id,
        "now": at(0),
        "six_hours_ago": at(6),
        "twenty_four_hours_ago": at(24),
        "seventy_two_hours_ago": at(72),
        "readings_available_hours": int(len(hist)),
    }


def _fleet_table(as_of: pd.Timestamp, sort_key: str, top: int = 17) -> list:
    machines = ms.list_machines(as_of)
    rows = [
        {
            "machine_id": m["machine_id"],
            "production_line": m["line"],
            "risk_score": m["risk_score"],
            "risk_level": m["risk_level"],
            "risk_trend": m["trend"]["direction"],
            "temperature_c": m["temperature_c"],
            "vibration_mm_s": m["vibration_mm_s"],
            "run_hours_since_maintenance": m["run_hours_since_maintenance"],
            "main_driver": m["main_driver"],
            "is_newly_commissioned": m["cold_start"]["is_cold_start"],
            "hours_of_history": m["cold_start"]["history_hours"],
        }
        for m in machines
    ]
    rows.sort(key=lambda r: (r.get(sort_key) is None, -(r.get(sort_key) or 0)))
    return rows[:top]


def retrieve(question: str, as_of: pd.Timestamp) -> Retrieval:
    art = ds.load_artifacts()
    intent = classify(question)
    machines = extract_machines(question)
    common = {
        "as_of": as_of.isoformat(),
        "prediction_horizon_hours": art.horizon_hours,
        "risk_bands": art.bands.to_dict(),
        "total_machines_monitored": len(art.machine_ids),
    }

    if intent in ("WHY_MACHINE", "RISK_INCREASED") and machines:
        ev = {
            **common,
            "machines": [_machine_card(m, as_of, deep=True) for m in machines],
        }
        if intent == "RISK_INCREASED":
            ev["risk_and_sensor_timeline"] = [
                _sensor_change_card(m, as_of) for m in machines
            ]
        return Retrieval(intent, machines, ev, "Full risk profile and drivers for the named machines.")

    if intent == "WHAT_CHANGED" and machines:
        return Retrieval(
            intent,
            machines,
            {
                **common,
                "machines": [_machine_card(m, as_of) for m in machines],
                "risk_and_sensor_timeline": [_sensor_change_card(m, as_of) for m in machines],
            },
            "Recent sensor and risk timeline for the named machines.",
        )

    if machines and intent in ("FLEET_OVERVIEW", "ATTENTION"):
        return Retrieval(
            intent,
            machines,
            {**common, "machines": [_machine_card(m, as_of, deep=True) for m in machines]},
            "Risk profile for the named machines.",
        )

    if intent == "ATTENTION":
        dash = ms.dashboard(as_of)
        return Retrieval(
            intent,
            [m["machine_id"] for m in dash["attention_required"]],
            {
                **common,
                "fleet_counts": dash["counts"],
                "machines_ranked_by_risk": _fleet_table(as_of, "risk_score"),
                "recent_risk_events": dash["recent_risk_events"],
            },
            "Fleet ranked by risk, plus recent alert history.",
        )

    if intent == "RANK_VIBRATION":
        return Retrieval(intent, [], {**common, "machines_ranked_by_vibration": _fleet_table(as_of, "vibration_mm_s")},
                         "Fleet ranked by current vibration.")

    if intent == "RANK_TEMPERATURE":
        return Retrieval(intent, [], {**common, "machines_ranked_by_temperature": _fleet_table(as_of, "temperature_c")},
                         "Fleet ranked by current temperature.")

    if intent == "RANK_RUNHOURS":
        return Retrieval(
            intent,
            [],
            {**common, "machines_ranked_by_run_hours_since_maintenance":
                _fleet_table(as_of, "run_hours_since_maintenance")},
            "Fleet ranked by run hours since last maintenance.",
        )

    if intent == "LOW_RISK":
        rows = [r for r in _fleet_table(as_of, "risk_score") if r["risk_level"] == "LOW"]
        return Retrieval(intent, [], {**common, "low_risk_machines": rows},
                         "Machines currently in the LOW band.")

    if intent == "LIMITED_HISTORY":
        cold = art.metadata.get("cold_start", {})
        cards = [_machine_card(m, as_of) for m in cold.get("machines", []) if m in art.machine_ids]
        return Retrieval(
            intent,
            cold.get("machines", []),
            {
                **common,
                "newly_commissioned_machines": cards,
                "cold_start_threshold_hours": cold.get("threshold_hours"),
                "how_these_are_handled": cold.get("strategy"),
                "known_limitation": cold.get("measurable_limitation"),
            },
            "Cold-start machines and how the system handles them.",
        )

    if intent == "MODEL_INFO":
        meta = art.metadata
        return Retrieval(
            intent,
            [],
            {
                **common,
                "model_type": meta.get("model_name"),
                "what_it_predicts": (
                    f"Probability of a failure event within "
                    f"{art.horizon_hours} hours of the reading."
                ),
                "horizon_rationale": meta.get("horizon_rationale"),
                "validation": meta.get("validation"),
                "measured_performance": meta.get("headline_metrics"),
                "threshold_choice": meta.get("threshold"),
                "what_the_model_uses_most": meta.get("group_importance", [])[:4],
                "cold_start_handling": meta.get("cold_start", {}).get("strategy"),
            },
            "Model design and its measured performance.",
        )

    dash = ms.dashboard(as_of)
    return Retrieval(
        "FLEET_OVERVIEW",
        [],
        {
            **common,
            "fleet_counts": dash["counts"],
            "machines_ranked_by_risk": _fleet_table(as_of, "risk_score"),
            "recent_risk_events": dash["recent_risk_events"],
        },
        "Whole-fleet snapshot ranked by risk.",
    )
