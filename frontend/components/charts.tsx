"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatDateShort, formatNumber, formatRiskScore, formatTimestamp } from "@/lib/format";
import type { RiskBands, SensorPoint } from "@/types/api";

type Row = SensorPoint & { t: number };

function toRows(history: SensorPoint[]): Row[] {
  return history.map((p) => ({ ...p, t: new Date(p.timestamp).getTime() }));
}

function failureTimes(history: SensorPoint[]): number[] {
  return history.filter((p) => p.failure_event === 1).map((p) => new Date(p.timestamp).getTime());
}

const AXIS = {
  stroke: "hsl(var(--muted-foreground))",
  fontSize: 11,
} as const;

function ChartTooltip({
  active,
  payload,
  label,
  unit,
  digits,
  isRisk,
}: {
  active?: boolean;
  payload?: { value: number | null; name?: string }[];
  label?: number;
  unit?: string;
  digits: number;
  isRisk?: boolean;
}) {
  if (!active || !payload?.length) return null;
  const v = payload[0]?.value;
  return (
    <div className="rounded-md border bg-popover px-3 py-2 text-xs shadow-md">
      <p className="font-medium text-muted-foreground">{formatTimestamp(new Date(label ?? 0).toISOString())}</p>
      <p className="mt-0.5 text-sm font-semibold tabular">
        {isRisk ? formatRiskScore(v) : formatNumber(v, digits, unit)}
      </p>
    </div>
  );
}

/** Temperature / vibration / run-hours over time, with failure markers. */
export function SensorChart({
  history,
  dataKey,
  color,
  unit,
  digits = 1,
  baseline,
  height = 200,
}: {
  history: SensorPoint[];
  dataKey: "temperature_c" | "vibration_mm_s" | "run_hours_since_maintenance";
  color: string;
  unit: string;
  digits?: number;
  baseline?: number | null;
  height?: number;
}) {
  const rows = toRows(history);
  const failures = failureTimes(history);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
        <XAxis
          dataKey="t"
          type="number"
          scale="time"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(v) => formatDateShort(new Date(v).toISOString())}
          {...AXIS}
          tickLine={false}
          minTickGap={48}
        />
        <YAxis {...AXIS} tickLine={false} axisLine={false} width={52} />
        <Tooltip
          content={<ChartTooltip unit={unit} digits={digits} />}
          cursor={{ stroke: "hsl(var(--muted-foreground))", strokeDasharray: "3 3" }}
        />
        {baseline != null && (
          <ReferenceLine
            y={baseline}
            stroke="hsl(var(--muted-foreground))"
            strokeDasharray="4 4"
            label={{
              value: "machine normal",
              position: "insideTopLeft",
              fontSize: 10,
              fill: "hsl(var(--muted-foreground))",
            }}
          />
        )}
        {failures.map((t) => (
          <ReferenceLine
            key={t}
            x={t}
            stroke="hsl(var(--risk-critical))"
            strokeWidth={2}
            label={{
              value: "failure",
              position: "top",
              fontSize: 10,
              fill: "hsl(var(--risk-critical))",
            }}
          />
        ))}
        <Line
          type="monotone"
          dataKey={dataKey}
          stroke={color}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/**
 * Risk score over time. The banded background is what makes "is this machine
 * getting worse" readable at a glance, rather than asking the user to interpret
 * a bare probability curve.
 */
export function RiskChart({
  history,
  bands,
  height = 240,
}: {
  history: SensorPoint[];
  bands: RiskBands;
  height?: number;
}) {
  const rows = toRows(history);
  const failures = failureTimes(history);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
        <defs>
          <linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="hsl(var(--risk-critical))" stopOpacity={0.35} />
            <stop offset="100%" stopColor="hsl(var(--risk-critical))" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />

        {/* Risk bands as background shading. */}
        <ReferenceArea y1={bands.critical} y2={1} fill="hsl(var(--risk-critical))" fillOpacity={0.07} />
        <ReferenceArea y1={bands.high} y2={bands.critical} fill="hsl(var(--risk-high))" fillOpacity={0.07} />
        <ReferenceArea y1={bands.medium} y2={bands.high} fill="hsl(var(--risk-medium))" fillOpacity={0.07} />

        <XAxis
          dataKey="t"
          type="number"
          scale="time"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(v) => formatDateShort(new Date(v).toISOString())}
          {...AXIS}
          tickLine={false}
          minTickGap={48}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v) => `${Math.round(v * 100)}%`}
          {...AXIS}
          tickLine={false}
          axisLine={false}
          width={52}
        />
        <Tooltip
          content={<ChartTooltip digits={0} isRisk />}
          cursor={{ stroke: "hsl(var(--muted-foreground))", strokeDasharray: "3 3" }}
        />
        <ReferenceLine
          y={bands.high}
          stroke="hsl(var(--risk-high))"
          strokeDasharray="4 4"
          label={{
            value: "alert threshold",
            position: "insideTopRight",
            fontSize: 10,
            fill: "hsl(var(--risk-high))",
          }}
        />
        {failures.map((t) => (
          <ReferenceLine
            key={t}
            x={t}
            stroke="hsl(var(--risk-critical))"
            strokeWidth={2}
            label={{
              value: "failure",
              position: "top",
              fontSize: 10,
              fill: "hsl(var(--risk-critical))",
            }}
          />
        ))}
        <Area
          type="monotone"
          dataKey="risk_score"
          stroke="hsl(var(--risk-critical))"
          strokeWidth={2}
          fill="url(#riskFill)"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/** Compact inline risk sparkline for the fleet table. */
export function RiskSparkline({ history }: { history: SensorPoint[] }) {
  const rows = toRows(history);
  return (
    <ResponsiveContainer width="100%" height={28}>
      <LineChart data={rows} margin={{ top: 2, right: 0, bottom: 2, left: 0 }}>
        <YAxis domain={[0, 1]} hide />
        <Line
          type="monotone"
          dataKey="risk_score"
          stroke="hsl(var(--risk-high))"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
