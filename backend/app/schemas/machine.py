"""Response models. These define the exact contract the frontend types mirror."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
TrendDirection = Literal["RISING", "STABLE", "FALLING"]


class DriverEvidence(BaseModel):
    feature: str
    label: str
    value: float | None = None
    typical_value: float | None = None


class RiskDriver(BaseModel):
    group: str
    label: str
    description: str
    importance: float = Field(..., description="Share of the current risk score, 0-1")
    contribution: float = Field(
        ..., description="Absolute drop in risk score if this were normal"
    )
    risk_if_normal: float
    evidence: list[DriverEvidence] = []


class SensorReading(BaseModel):
    current: float | None = None
    rolling_mean_24h: float | None = None
    change_24h: float | None = None
    baseline: float | None = None
    deviation_from_baseline: float | None = None
    z_vs_baseline: float | None = None
    unit: str


class RiskTrend(BaseModel):
    direction: TrendDirection
    delta: float
    previous_score: float | None = None
    lookback_hours: int = 24


class ColdStartInfo(BaseModel):
    is_cold_start: bool
    history_hours: int
    baseline_source: str = Field(
        ..., description="machine = own baseline; line_pooled = borrowed from the line"
    )
    message: str | None = None


class MachineSummary(BaseModel):
    machine_id: str
    line: str
    timestamp: str
    risk_score: float
    risk_level: RiskLevel
    prediction_horizon_hours: int
    temperature_c: float | None = None
    vibration_mm_s: float | None = None
    run_hours_since_maintenance: float | None = None
    trend: RiskTrend
    main_driver: str | None = None
    cold_start: ColdStartInfo


class SensorPoint(BaseModel):
    timestamp: str
    temperature_c: float | None = None
    vibration_mm_s: float | None = None
    run_hours_since_maintenance: float | None = None
    risk_score: float | None = None
    risk_level: RiskLevel | None = None
    failure_event: int = 0


class MaintenanceEvent(BaseModel):
    timestamp: str
    type: Literal["FAILURE_REPAIR", "PLANNED_MAINTENANCE"]


class MachineDetail(BaseModel):
    machine_id: str
    line: str
    timestamp: str
    risk_score: float
    risk_level: RiskLevel
    prediction_horizon_hours: int
    trend: RiskTrend
    temperature: SensorReading
    vibration: SensorReading
    run_hours_since_maintenance: float | None = None
    drivers: list[RiskDriver]
    history: list[SensorPoint]
    failure_history: list[str]
    maintenance_history: list[MaintenanceEvent]
    cold_start: ColdStartInfo
    score_is_out_of_sample: bool


class DashboardCounts(BaseModel):
    total_machines: int
    critical: int
    high: int
    medium: int
    low: int
    newly_commissioned: int
    trending_up: int


class RecentRiskEvent(BaseModel):
    machine_id: str
    started_at: str
    ended_at: str
    peak_risk_score: float
    peak_risk_level: RiskLevel
    resulted_in_failure: bool
    failure_at: str | None = None
    warning_hours: float | None = None


class Dashboard(BaseModel):
    as_of: str
    data_range: dict
    prediction_horizon_hours: int
    counts: DashboardCounts
    machines: list[MachineSummary]
    attention_required: list[MachineSummary]
    recent_risk_events: list[RecentRiskEvent]
    risk_bands: dict


class Explanation(BaseModel):
    machine_id: str
    generated_at: str
    source: Literal["llm", "fallback"]
    model: str | None = None
    headline: str
    explanation: str
    recommended_action: str
    confidence_note: str | None = None
    evidence: dict
    warning: str | None = None


class QARequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=500)
    as_of: str | None = None


class QASource(BaseModel):
    kind: str
    machine_id: str | None = None
    detail: str


class QAResponse(BaseModel):
    question: str
    answer: str
    source: Literal["llm", "fallback"]
    model: str | None = None
    intent: str
    machines_referenced: list[str] = []
    evidence: dict
    warning: str | None = None
