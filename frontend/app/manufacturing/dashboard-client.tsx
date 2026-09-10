"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  AlertTriangle,
  CircleAlert,
  Factory,
  History,
  Sparkles,
  TrendingUp,
} from "lucide-react";

import { AsOfControl } from "@/components/as-of-control";
import { EMPTY_FILTERS, Filters, type FilterState } from "@/components/filters";
import { KpiCard } from "@/components/kpi-card";
import { MachineTable } from "@/components/machine-table";
import { PageHeader } from "@/components/page-header";
import { QaPanel } from "@/components/qa-panel";
import { RiskBadge } from "@/components/risk-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { formatRiskScore, formatTimestamp } from "@/lib/format";
import type { AppMetadata, Dashboard } from "@/types/api";

export function DashboardClient() {
  const params = useSearchParams();
  const asOf = params.get("as_of");

  const [data, setData] = useState<Dashboard | null>(null);
  const [meta, setMeta] = useState<AppMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setData(null);
    Promise.all([api.dashboard(asOf), api.metadata()])
      .then(([d, m]) => {
        if (cancelled) return;
        setData(d);
        setMeta(m);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "Could not load fleet data.");
      });
    return () => {
      cancelled = true;
    };
  }, [asOf]);

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.machines.filter((m) => {
      if (filters.riskLevel !== "all" && m.risk_level !== filters.riskLevel) return false;
      if (filters.trend !== "all" && m.trend.direction !== filters.trend) return false;
      if (filters.cohort === "new" && !m.cold_start.is_cold_start) return false;
      if (filters.cohort === "established" && m.cold_start.is_cold_start) return false;
      if (
        filters.search &&
        !m.machine_id.toLowerCase().includes(filters.search.trim().toLowerCase())
      )
        return false;
      return true;
    });
  }, [data, filters]);

  // Most recent moment the model raised an alert, so the reviewer can jump
  // straight to a period where the tool has something to show.
  const lastAlert = data?.recent_risk_events?.[0]?.ended_at ?? null;

  if (error) {
    return (
      <>
        <PageHeader
          title="Manufacturing Machine Risk Monitor"
          subtitle="Identify machines that may require attention before an unexpected failure."
        />
        <main className="container py-8">
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Cannot load fleet data</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </main>
      </>
    );
  }

  if (!data || !meta) {
    return (
      <>
        <PageHeader
          title="Manufacturing Machine Risk Monitor"
          subtitle="Identify machines that may require attention before an unexpected failure."
        />
        <main className="container space-y-6 py-8">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-28" />
            ))}
          </div>
          <Skeleton className="h-96" />
        </main>
      </>
    );
  }

  const c = data.counts;
  const needAttention = c.critical + c.high;

  return (
    <>
      <PageHeader
        title="Manufacturing Machine Risk Monitor"
        subtitle="Identify machines that may require attention before an unexpected failure."
        right={
          <AsOfControl
            asOf={data.as_of}
            dataRange={data.data_range}
            jumpTo={lastAlert}
            jumpLabel="Go to last alert"
            basePath="/manufacturing"
          />
        }
      />

      <main className="container space-y-6 py-8">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <KpiCard
            label="Machines monitored"
            value={c.total_machines}
            Icon={Factory}
            hint={`Across ${meta.lines.length} production lines`}
          />
          <KpiCard
            label="Critical or high risk"
            value={needAttention}
            Icon={CircleAlert}
            tone={needAttention > 0 ? "alert" : "good"}
            hint={
              needAttention > 0
                ? `${c.critical} critical, ${c.high} high`
                : "No machines above the alert threshold"
            }
            tooltip={`Machines the model estimates are at elevated risk of failing within ${data.prediction_horizon_hours} hours.`}
          />
          <KpiCard
            label="Trending upward"
            value={c.trending_up}
            Icon={TrendingUp}
            tone={c.trending_up > 0 ? "watch" : "neutral"}
            hint="Risk score has risen in the last 24 hours"
            tooltip="Machines whose risk score is higher now than it was 24 hours ago, whatever band they are in."
          />
          <KpiCard
            label="Newly commissioned"
            value={c.newly_commissioned}
            Icon={Sparkles}
            hint="Limited history - estimates less reliable"
            tooltip="Machines with under 14 days of recorded history. They have no failures of their own to learn from, so their scores rely on fleet-wide patterns."
          />
        </div>

        {needAttention === 0 && (
          <Alert variant="info">
            <CircleAlert aria-hidden />
            <AlertTitle>No machines are above the alert threshold right now</AlertTitle>
            <AlertDescription>
              This is a genuine all-clear at {formatTimestamp(data.as_of)}. Recent alerts the
              model raised are listed below, and you can move the review time in the header to
              inspect any earlier point.
            </AlertDescription>
          </Alert>
        )}

        <section className="space-y-4">
          <div>
            <h2 className="text-lg font-semibold">Machines requiring attention</h2>
            <p className="text-sm text-muted-foreground">
              Ranked by the estimated probability of failure within the next{" "}
              {data.prediction_horizon_hours} hours.
            </p>
          </div>

          <Filters
            value={filters}
            onChange={setFilters}
            resultCount={filtered.length}
            totalCount={data.machines.length}
          />

          <MachineTable machines={filtered} asOf={asOf} />
        </section>

        <div className="grid gap-6 lg:grid-cols-2">
          <QaPanel asOf={asOf} suggestions={meta.suggested_questions} />

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <History className="h-4 w-4 text-primary" aria-hidden />
                Recent alerts raised
              </CardTitle>
              <CardDescription>
                Periods in the last 14 days where a machine sat above the alert threshold, and
                what followed.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {data.recent_risk_events.length === 0 ? (
                <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
                  No alerts raised in the 14 days before this review time.
                </p>
              ) : (
                <ul className="space-y-3">
                  {data.recent_risk_events.map((e) => (
                    <li
                      key={`${e.machine_id}-${e.started_at}`}
                      className="rounded-md border p-3"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <Link
                          href={`/manufacturing/machine/${e.machine_id}?as_of=${encodeURIComponent(e.ended_at)}`}
                          className="font-semibold text-primary hover:underline"
                        >
                          {e.machine_id}
                        </Link>
                        <RiskBadge level={e.peak_risk_level} />
                      </div>
                      <p className="mt-1 text-sm text-muted-foreground">
                        Peaked at {formatRiskScore(e.peak_risk_score)} between{" "}
                        {formatTimestamp(e.started_at)} and {formatTimestamp(e.ended_at)}.
                      </p>
                      <p className="mt-1 text-sm">
                        {e.resulted_in_failure ? (
                          <span className="font-medium text-risk-critical">
                            A failure followed on {formatTimestamp(e.failure_at)}
                            {e.warning_hours != null &&
                              ` - ${Math.round(e.warning_hours)} hours after the first alert.`}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">
                            No failure recorded at or before this review time.
                          </span>
                        )}
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        </div>

        <p className="pb-4 text-xs text-muted-foreground">
          Risk bands: medium from {formatRiskScore(data.risk_bands.medium)}, high from{" "}
          {formatRiskScore(data.risk_bands.high)}, critical from{" "}
          {formatRiskScore(data.risk_bands.critical)}. Data extract covers{" "}
          {formatTimestamp(data.data_range.start)} to {formatTimestamp(data.data_range.end)}.
        </p>
      </main>
    </>
  );
}
