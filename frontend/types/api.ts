/** Types mirroring the FastAPI response models in backend/app/schemas/machine.py. */

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type TrendDirection = "RISING" | "STABLE" | "FALLING";

export interface RiskTrend {
  direction: TrendDirection;
  delta: number;
  previous_score: number | null;
  lookback_hours: number;
}

export interface ColdStartInfo {
  is_cold_start: boolean;
  history_hours: number;
  baseline_source: string;
  message: string | null;
}

export interface MachineSummary {
  machine_id: string;
  line: string;
  timestamp: string;
  risk_score: number;
  risk_level: RiskLevel;
  prediction_horizon_hours: number;
  temperature_c: number | null;
  vibration_mm_s: number | null;
  run_hours_since_maintenance: number | null;
  trend: RiskTrend;
  main_driver: string | null;
  cold_start: ColdStartInfo;
}

export interface SensorReading {
  current: number | null;
  rolling_mean_24h: number | null;
  change_24h: number | null;
  baseline: number | null;
  deviation_from_baseline: number | null;
  z_vs_baseline: number | null;
  unit: string;
}

export interface DriverEvidence {
  feature: string;
  label: string;
  value: number | null;
  typical_value: number | null;
}

export interface RiskDriver {
  group: string;
  label: string;
  description: string;
  importance: number;
  contribution: number;
  risk_if_normal: number;
  evidence: DriverEvidence[];
}

export interface SensorPoint {
  timestamp: string;
  temperature_c: number | null;
  vibration_mm_s: number | null;
  run_hours_since_maintenance: number | null;
  risk_score: number | null;
  risk_level: RiskLevel | null;
  failure_event: number;
}

export interface MaintenanceEvent {
  timestamp: string;
  type: "FAILURE_REPAIR" | "PLANNED_MAINTENANCE";
}

export interface MachineDetail {
  machine_id: string;
  line: string;
  timestamp: string;
  risk_score: number;
  risk_level: RiskLevel;
  prediction_horizon_hours: number;
  trend: RiskTrend;
  temperature: SensorReading;
  vibration: SensorReading;
  run_hours_since_maintenance: number | null;
  drivers: RiskDriver[];
  history: SensorPoint[];
  failure_history: string[];
  maintenance_history: MaintenanceEvent[];
  cold_start: ColdStartInfo;
  score_is_out_of_sample: boolean;
}

export interface DashboardCounts {
  total_machines: number;
  critical: number;
  high: number;
  medium: number;
  low: number;
  newly_commissioned: number;
  trending_up: number;
}

export interface RecentRiskEvent {
  machine_id: string;
  started_at: string;
  ended_at: string;
  peak_risk_score: number;
  peak_risk_level: RiskLevel;
  resulted_in_failure: boolean;
  failure_at: string | null;
  warning_hours: number | null;
}

export interface RiskBands {
  medium: number;
  high: number;
  critical: number;
}

export interface Dashboard {
  as_of: string;
  data_range: { start: string; end: string };
  prediction_horizon_hours: number;
  counts: DashboardCounts;
  machines: MachineSummary[];
  attention_required: MachineSummary[];
  recent_risk_events: RecentRiskEvent[];
  risk_bands: RiskBands;
}

export interface Explanation {
  machine_id: string;
  generated_at: string;
  source: "llm" | "fallback";
  model: string | null;
  headline: string;
  explanation: string;
  recommended_action: string;
  confidence_note: string | null;
  evidence: Record<string, unknown>;
  warning: string | null;
}

export interface QAResponse {
  question: string;
  answer: string;
  source: "llm" | "fallback";
  model: string | null;
  intent: string;
  machines_referenced: string[];
  evidence: Record<string, unknown>;
  warning: string | null;
}

export interface AppMetadata {
  data_range: { start: string; end: string };
  default_as_of: string;
  machines: string[];
  lines: string[];
  prediction_horizon_hours: number;
  risk_bands: RiskBands;
  cold_start_machines: string[];
  llm_configured: boolean;
  suggested_questions: string[];
}

export interface ModelInfo {
  model_name: string;
  model_version: string;
  trained_at: string;
  prediction_horizon_hours: number;
  horizon_rationale: string;
  n_features: number;
  risk_bands: RiskBands;
  threshold: Record<string, unknown>;
  validation: Record<string, unknown>;
  headline_metrics: Record<string, number | string | null>;
  group_importance: { group: string; label: string; description: string; importance: number }[];
  cold_start: Record<string, unknown>;
  dataset: Record<string, unknown>;
  llm: { configured: boolean; model: string | null };
}
