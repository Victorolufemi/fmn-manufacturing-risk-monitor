"use client";

import Link from "next/link";
import { ChevronRight } from "lucide-react";

import { RiskBadge } from "@/components/risk-badge";
import { TrendIndicator } from "@/components/trend-indicator";
import { ColdStartTag } from "@/components/cold-start-notice";
import { InfoTip } from "@/components/ui/tooltip";
import { formatHours, formatNumber, formatRiskScore } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { MachineSummary } from "@/types/api";

export function MachineTable({
  machines,
  asOf,
}: {
  machines: MachineSummary[];
  asOf: string | null;
}) {
  if (machines.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-10 text-center">
        <p className="font-medium">No machines match these filters</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Clear a filter or search for a different machine.
        </p>
      </div>
    );
  }

  const href = (id: string) =>
    asOf
      ? `/manufacturing/machine/${encodeURIComponent(id)}?as_of=${encodeURIComponent(asOf)}`
      : `/manufacturing/machine/${encodeURIComponent(id)}`;

  return (
    <div className="overflow-x-auto rounded-lg border bg-card">
      <table className="w-full min-w-[900px] text-sm">
        <thead>
          <tr className="border-b bg-muted/40 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <th className="px-4 py-3 font-semibold">Machine</th>
            <th className="px-4 py-3 font-semibold">Risk</th>
            <th className="px-4 py-3 font-semibold">
              Risk score
              <InfoTip label="About risk score">
                The model estimated probability that this machine has a failure within the
                prediction window. It is calibrated against historical outcomes, not a raw
                model output.
              </InfoTip>
            </th>
            <th className="px-4 py-3 font-semibold">Temperature</th>
            <th className="px-4 py-3 font-semibold">Vibration</th>
            <th className="px-4 py-3 font-semibold">
              Run hours
              <InfoTip label="About run hours">
                Hours the machine has run since its last maintenance or repair.
              </InfoTip>
            </th>
            <th className="px-4 py-3 font-semibold">
              Trend
              <InfoTip label="About trend">
                Direction of the risk score over the last 24 hours.
              </InfoTip>
            </th>
            <th className="px-4 py-3 font-semibold">Main driver</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {machines.map((m) => (
            <tr
              key={m.machine_id}
              className={cn(
                "border-b last:border-0 transition-colors hover:bg-accent/40",
                m.risk_level === "CRITICAL" && "bg-risk-critical/[0.035]",
              )}
            >
              <td className="px-4 py-3">
                <Link
                  href={href(m.machine_id)}
                  className="font-semibold text-primary hover:underline"
                >
                  {m.machine_id}
                </Link>
                <div className="mt-0.5 flex items-center gap-1.5 text-xs text-muted-foreground">
                  <span>{m.line}</span>
                  {m.cold_start.is_cold_start && (
                    <ColdStartTag hours={m.cold_start.history_hours} />
                  )}
                </div>
              </td>
              <td className="px-4 py-3">
                <RiskBadge level={m.risk_level} />
              </td>
              <td className="px-4 py-3 font-semibold tabular">
                {formatRiskScore(m.risk_score)}
              </td>
              <td className="px-4 py-3 tabular">
                {formatNumber(m.temperature_c, 1, "°C")}
              </td>
              <td className="px-4 py-3 tabular">
                {formatNumber(m.vibration_mm_s, 2, "mm/s")}
              </td>
              <td className="px-4 py-3 tabular">
                {formatHours(m.run_hours_since_maintenance)}
              </td>
              <td className="px-4 py-3">
                <TrendIndicator trend={m.trend} />
              </td>
              <td className="px-4 py-3 text-muted-foreground">
                {m.main_driver ?? <span className="text-muted-foreground/60">-</span>}
              </td>
              <td className="px-4 py-3 text-right">
                <Link
                  href={href(m.machine_id)}
                  aria-label={`Open ${m.machine_id}`}
                  className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
                >
                  <ChevronRight className="h-4 w-4" />
                </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
