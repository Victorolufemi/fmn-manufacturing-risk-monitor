import { formatHours, formatNumber, formatRiskScore } from "@/lib/format";
import type { MachineDetail } from "@/types/api";

function Row({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: string;
  emphasis?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-dashed py-1.5 last:border-0">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className={`tabular text-sm ${emphasis ? "font-semibold" : "font-medium"}`}>
        {value}
      </span>
    </div>
  );
}

/**
 * The exact numbers the backend put in front of the model. Rendering these
 * makes the AI explanation auditable: a supervisor can check every figure it
 * quotes without leaving the page.
 */
export function EvidencePanel({ machine }: { machine: MachineDetail }) {
  const t = machine.temperature;
  const v = machine.vibration;

  return (
    <div className="rounded-md border">
      <div className="border-b bg-muted/40 px-4 py-2">
        <p className="text-sm font-semibold">Evidence used</p>
        <p className="text-xs text-muted-foreground">
          Readings taken from {machine.machine_id} at this review time.
        </p>
      </div>
      <div className="grid gap-x-8 gap-y-0 p-4 sm:grid-cols-2">
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Temperature
          </p>
          <Row label="Current" value={formatNumber(t.current, 1, "°C")} emphasis />
          <Row label="Recent average (24h)" value={formatNumber(t.rolling_mean_24h, 1, "°C")} />
          <Row label="Change over 24h" value={formatNumber(t.change_24h, 1, "°C")} />
          <Row label="Normal for this machine" value={formatNumber(t.baseline, 1, "°C")} />
          <Row
            label="Above its normal by"
            value={formatNumber(t.deviation_from_baseline, 1, "°C")}
          />
        </div>
        <div>
          <p className="mb-1 mt-4 text-xs font-semibold uppercase tracking-wide text-muted-foreground sm:mt-0">
            Vibration
          </p>
          <Row label="Current" value={formatNumber(v.current, 2, "mm/s")} emphasis />
          <Row label="Recent average (24h)" value={formatNumber(v.rolling_mean_24h, 2, "mm/s")} />
          <Row label="Change over 24h" value={formatNumber(v.change_24h, 2, "mm/s")} />
          <Row label="Normal for this machine" value={formatNumber(v.baseline, 2, "mm/s")} />
          <Row
            label="Above its normal by"
            value={formatNumber(v.deviation_from_baseline, 2, "mm/s")}
          />
        </div>
        <div className="mt-4 sm:col-span-2">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Maintenance and risk
          </p>
          <Row
            label="Run hours since maintenance"
            value={formatHours(machine.run_hours_since_maintenance)}
          />
          <Row label="Risk score" value={formatRiskScore(machine.risk_score)} emphasis />
          <Row
            label="Prediction window"
            value={`next ${machine.prediction_horizon_hours} hours`}
          />
          <Row
            label="Hours of history for this machine"
            value={formatHours(machine.cold_start.history_hours)}
          />
        </div>
      </div>
    </div>
  );
}
