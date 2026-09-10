"use client";

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { AlertTriangle, Activity, Thermometer, Timer, Waves } from "lucide-react";

import { AsOfControl } from "@/components/as-of-control";
import { ColdStartNotice } from "@/components/cold-start-notice";
import { DriverList } from "@/components/driver-list";
import { ExplanationCard } from "@/components/explanation-card";
import { PageHeader } from "@/components/page-header";
import { QaPanel } from "@/components/qa-panel";
import { RiskBadge } from "@/components/risk-badge";
import { RiskChart, SensorChart } from "@/components/charts";
import { TrendIndicator } from "@/components/trend-indicator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { InfoTip } from "@/components/ui/tooltip";
import { api } from "@/lib/api";
import {
  formatHours,
  formatNumber,
  formatRiskScore,
  formatSigned,
  formatTimestamp,
} from "@/lib/format";
import type { AppMetadata, MachineDetail } from "@/types/api";

function SensorCard({
  title,
  Icon,
  value,
  unit,
  digits,
  average,
  change,
  baseline,
  deviation,
}: {
  title: string;
  Icon: typeof Thermometer;
  value: number | null;
  unit: string;
  digits: number;
  average: number | null;
  change: number | null;
  baseline: number | null;
  deviation: number | null;
}) {
  const elevated = deviation != null && deviation > 0;
  return (
    <Card className="p-5">
      <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
        <Icon className="h-4 w-4" aria-hidden />
        {title}
      </div>
      <p className="mt-1.5 text-3xl font-semibold tabular">
        {formatNumber(value, digits)}
        <span className="ml-1 text-base font-normal text-muted-foreground">{unit}</span>
      </p>
      <dl className="mt-3 space-y-1 text-sm">
        <div className="flex justify-between gap-3">
          <dt className="text-muted-foreground">Average, last 24h</dt>
          <dd className="tabular font-medium">{formatNumber(average, digits)}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-muted-foreground">Change over 24h</dt>
          <dd className="tabular font-medium">{formatSigned(change, digits)}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-muted-foreground">Normal for this machine</dt>
          <dd className="tabular font-medium">{formatNumber(baseline, digits)}</dd>
        </div>
        <div className="flex justify-between gap-3">
          <dt className="text-muted-foreground">Above its normal by</dt>
          <dd
            className={`tabular font-semibold ${elevated ? "text-risk-high" : "text-muted-foreground"}`}
          >
            {formatSigned(deviation, digits)}
          </dd>
        </div>
      </dl>
    </Card>
  );
}

export function MachineClient({ machineId }: { machineId: string }) {
  const params = useSearchParams();
  const asOf = params.get("as_of");

  const [data, setData] = useState<MachineDetail | null>(null);
  const [meta, setMeta] = useState<AppMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    setData(null);
    Promise.all([api.machine(machineId, asOf, 336), api.metadata()])
      .then(([d, m]) => {
        if (cancelled) return;
        setData(d);
        setMeta(m);
      })
      .catch((e) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Could not load this machine.");
      });
    return () => {
      cancelled = true;
    };
  }, [machineId, asOf]);

  const backHref = asOf
    ? `/manufacturing?as_of=${encodeURIComponent(asOf)}`
    : "/manufacturing";

  if (error) {
    return (
      <>
        <PageHeader title={machineId} backHref={backHref} backLabel="All machines" />
        <main className="container py-8">
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertTitle>Cannot load {machineId}</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </main>
      </>
    );
  }

  if (!data || !meta) {
    return (
      <>
        <PageHeader title={machineId} backHref={backHref} backLabel="All machines" />
        <main className="container space-y-6 py-8">
          <Skeleton className="h-32" />
          <div className="grid gap-4 md:grid-cols-3">
            <Skeleton className="h-52" />
            <Skeleton className="h-52" />
            <Skeleton className="h-52" />
          </div>
          <Skeleton className="h-72" />
        </main>
      </>
    );
  }

  return (
    <>
      <PageHeader
        title={data.machine_id}
        subtitle={`${data.line} - reading taken ${formatTimestamp(data.timestamp)}`}
        backHref={backHref}
        backLabel="All machines"
        right={
          <AsOfControl
            asOf={data.timestamp}
            dataRange={meta.data_range}
            basePath={`/manufacturing/machine/${encodeURIComponent(machineId)}`}
          />
        }
      />

      <main className="container space-y-6 py-8">
        <ColdStartNotice info={data.cold_start} />

        <Card>
          <CardContent className="flex flex-wrap items-center justify-between gap-6 p-6">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Current risk</p>
              <div className="mt-2 flex items-center gap-3">
                <RiskBadge level={data.risk_level} size="lg" />
                <span className="text-4xl font-semibold tabular">
                  {formatRiskScore(data.risk_score)}
                </span>
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">
                Prediction window
                <InfoTip label="About the prediction window">
                  The model estimates the chance of a failure occurring within this many hours
                  of the reading. It is not a countdown or a predicted failure time.
                </InfoTip>
              </p>
              <p className="mt-2 text-2xl font-semibold tabular">
                next {data.prediction_horizon_hours} hours
              </p>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Trend, last 24h</p>
              <div className="mt-2 flex items-baseline gap-2">
                <TrendIndicator trend={data.trend} className="text-lg" />
                <span className="tabular text-sm text-muted-foreground">
                  {formatRiskScore(data.trend.previous_score)} &rarr;{" "}
                  {formatRiskScore(data.risk_score)}
                </span>
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-muted-foreground">Run hours</p>
              <p className="mt-2 text-2xl font-semibold tabular">
                {formatHours(data.run_hours_since_maintenance)}
              </p>
              <p className="text-xs text-muted-foreground">since last maintenance</p>
            </div>
          </CardContent>
        </Card>

        <ExplanationCard machine={data} asOf={asOf} />

        <div className="grid gap-4 md:grid-cols-3">
          <SensorCard
            title="Temperature"
            Icon={Thermometer}
            value={data.temperature.current}
            unit="°C"
            digits={1}
            average={data.temperature.rolling_mean_24h}
            change={data.temperature.change_24h}
            baseline={data.temperature.baseline}
            deviation={data.temperature.deviation_from_baseline}
          />
          <SensorCard
            title="Vibration"
            Icon={Waves}
            value={data.vibration.current}
            unit="mm/s"
            digits={2}
            average={data.vibration.rolling_mean_24h}
            change={data.vibration.change_24h}
            baseline={data.vibration.baseline}
            deviation={data.vibration.deviation_from_baseline}
          />
          <Card className="p-5">
            <div className="flex items-center gap-2 text-sm font-medium text-muted-foreground">
              <Timer className="h-4 w-4" aria-hidden />
              Maintenance history
            </div>
            <p className="mt-1.5 text-3xl font-semibold tabular">
              {formatHours(data.run_hours_since_maintenance)}
            </p>
            <p className="text-sm text-muted-foreground">since last maintenance</p>
            <div className="mt-3 space-y-1.5 text-sm">
              {data.maintenance_history.length === 0 ? (
                <p className="text-muted-foreground">
                  No maintenance events recorded in the data up to this review time.
                </p>
              ) : (
                data.maintenance_history
                  .slice(-4)
                  .reverse()
                  .map((e) => (
                    <div key={e.timestamp} className="flex justify-between gap-3">
                      <span className="text-muted-foreground">
                        {e.type === "FAILURE_REPAIR" ? "Repair after failure" : "Planned work"}
                      </span>
                      <span className="tabular font-medium">
                        {formatTimestamp(e.timestamp)}
                      </span>
                    </div>
                  ))
              )}
            </div>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-4 w-4 text-primary" aria-hidden />
              Risk score over time
              <InfoTip label="About this chart">
                Every point is a historical model prediction from the same trained pipeline,
                produced without using data from after that point. Shaded bands show the
                medium, high and critical thresholds.
              </InfoTip>
            </CardTitle>
            <CardDescription>
              Whether this machine is becoming more concerning, not just where it stands now.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <RiskChart history={data.history} bands={meta.risk_bands} />
          </CardContent>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Temperature over time</CardTitle>
              <CardDescription>
                Dashed line is this machine&apos;s own normal operating temperature.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SensorChart
                history={data.history}
                dataKey="temperature_c"
                color="hsl(var(--risk-high))"
                unit="°C"
                digits={1}
                baseline={data.temperature.baseline}
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Vibration over time</CardTitle>
              <CardDescription>
                Dashed line is this machine&apos;s own normal vibration level.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <SensorChart
                history={data.history}
                dataKey="vibration_mm_s"
                color="hsl(var(--primary))"
                unit="mm/s"
                digits={2}
                baseline={data.vibration.baseline}
              />
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Run hours since maintenance</CardTitle>
            <CardDescription>
              Resets to zero at each maintenance event or repair.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <SensorChart
              history={data.history}
              dataKey="run_hours_since_maintenance"
              color="hsl(var(--muted-foreground))"
              unit="h"
              digits={0}
              height={170}
            />
          </CardContent>
        </Card>

        <div className="grid gap-6 lg:grid-cols-2">
          <DriverList drivers={data.drivers} />
          <QaPanel
            asOf={asOf}
            suggestions={[
              `Why is ${data.machine_id} at risk?`,
              `What changed on ${data.machine_id}?`,
              "Which machines need attention right now?",
            ]}
          />
        </div>

        <p className="pb-4 text-xs text-muted-foreground">
          {data.failure_history.length > 0
            ? `Recorded failures for this machine up to this review time: ${data.failure_history
                .map((t) => formatTimestamp(t))
                .join("; ")}.`
            : "No failures recorded for this machine up to this review time."}{" "}
          {data.score_is_out_of_sample
            ? "This risk score was produced by a model trained only on data from before this period."
            : "This period falls inside the model's earliest training window, so its score is shown for context rather than as an out-of-sample prediction."}
        </p>
      </main>
    </>
  );
}
