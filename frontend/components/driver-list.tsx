import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { InfoTip } from "@/components/ui/tooltip";
import { formatNumber, formatRiskScore } from "@/lib/format";
import type { RiskDriver } from "@/types/api";

/**
 * Risk drivers come from a counterfactual run of the fitted model: each feature
 * family is replaced with this machine's own normal values and the model is
 * re-scored. The drop is that family's contribution, which is why we can say
 * "at normal vibration this machine would score X" rather than just naming a
 * feature.
 */
export function DriverList({ drivers }: { drivers: RiskDriver[] }) {
  if (drivers.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          What is driving this risk
          <InfoTip label="About risk drivers">
            Each driver is measured by re-running the model with that group of readings set
            back to this machine&apos;s own normal values. The bar shows how much of the
            current risk score disappears when it does.
          </InfoTip>
        </CardTitle>
        <CardDescription>Ranked by how much each one is raising the score.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {drivers.map((d) => (
          <div key={d.group}>
            <div className="flex items-baseline justify-between gap-4">
              <p className="font-medium">{d.label}</p>
              <p className="tabular text-sm font-semibold">{Math.round(d.importance * 100)}%</p>
            </div>
            <p className="mt-0.5 text-sm text-muted-foreground">{d.description}</p>

            <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-risk-high"
                style={{ width: `${Math.max(2, Math.round(d.importance * 100))}%` }}
              />
            </div>

            {d.evidence.length > 0 && (
              <dl className="mt-2.5 grid gap-x-6 gap-y-1 sm:grid-cols-2">
                {d.evidence.map((e) => (
                  <div key={e.feature} className="flex items-baseline justify-between gap-3">
                    <dt className="text-xs text-muted-foreground">{e.label}</dt>
                    <dd className="tabular text-xs">
                      <span className="font-semibold">{formatNumber(e.value, 2)}</span>
                      <span className="text-muted-foreground">
                        {" "}
                        vs {formatNumber(e.typical_value, 2)} normal
                      </span>
                    </dd>
                  </div>
                ))}
              </dl>
            )}

            <p className="mt-2 text-xs text-muted-foreground">
              If this were back to normal, the model would score this machine at{" "}
              <span className="font-semibold text-foreground">
                {formatRiskScore(d.risk_if_normal)}
              </span>
              .
            </p>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
