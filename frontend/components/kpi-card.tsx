import type { LucideIcon } from "lucide-react";

import { Card } from "@/components/ui/card";
import { InfoTip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export function KpiCard({
  label,
  value,
  hint,
  tooltip,
  Icon,
  tone = "neutral",
}: {
  label: string;
  value: string | number;
  hint?: string;
  tooltip?: string;
  Icon: LucideIcon;
  tone?: "neutral" | "alert" | "watch" | "good";
}) {
  const tones = {
    neutral: "text-foreground",
    alert: "text-risk-critical",
    watch: "text-risk-high",
    good: "text-risk-low",
  };
  return (
    <Card className="p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-muted-foreground">
            {label}
            {tooltip && <InfoTip label={`About ${label}`}>{tooltip}</InfoTip>}
          </p>
          <p className={cn("mt-1.5 text-3xl font-semibold tabular", tones[tone])}>{value}</p>
          {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
        </div>
        <div className="rounded-md bg-muted p-2 text-muted-foreground">
          <Icon className="h-5 w-5" aria-hidden />
        </div>
      </div>
    </Card>
  );
}
