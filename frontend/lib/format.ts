import type { RiskLevel, TrendDirection } from "@/types/api";

/** Risk scores are probabilities; plant staff read percentages far faster. */
export function formatRiskScore(score: number | null | undefined): string {
  if (score === null || score === undefined || Number.isNaN(score)) return "-";
  const pct = score * 100;
  if (pct >= 99.5 && pct < 100) return ">99%";
  if (pct > 0 && pct < 1) return "<1%";
  return `${Math.round(pct)}%`;
}

export function formatNumber(
  value: number | null | undefined,
  digits = 1,
  unit?: string,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const s = value.toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return unit ? `${s} ${unit}` : s;
}

export function formatSigned(value: number | null | undefined, digits = 1, unit?: string): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value, digits, unit)}`;
}

export function formatHours(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  return `${Math.round(value).toLocaleString()} h`;
}

export function formatTimestamp(iso: string | null | undefined): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function formatDateShort(iso: string | null | undefined): string {
  if (!iso) return "-";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
  });
}

/** ISO string a datetime-local input can consume, and the API can parse. */
export function toLocalInputValue(iso: string): string {
  return iso.slice(0, 16);
}

export const RISK_ORDER: RiskLevel[] = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];

export const RISK_COPY: Record<RiskLevel, { label: string; meaning: string }> = {
  CRITICAL: {
    label: "Critical",
    meaning: "Strong evidence of impending failure. Investigate now.",
  },
  HIGH: {
    label: "High",
    meaning: "Above the alert threshold. Schedule an inspection this shift.",
  },
  MEDIUM: {
    label: "Medium",
    meaning: "Elevated but below the alert threshold. Keep on the watch list.",
  },
  LOW: {
    label: "Low",
    meaning: "No elevated risk signal in the sensor data.",
  },
};

export const TREND_COPY: Record<TrendDirection, string> = {
  RISING: "Risk has increased over the last 24 hours",
  FALLING: "Risk has decreased over the last 24 hours",
  STABLE: "Risk is broadly unchanged over the last 24 hours",
};
