import { Minus, TrendingDown, TrendingUp } from "lucide-react";

import { cn } from "@/lib/utils";
import { TREND_COPY } from "@/lib/format";
import type { RiskTrend } from "@/types/api";

export function TrendIndicator({
  trend,
  className,
  showLabel = true,
}: {
  trend: RiskTrend;
  className?: string;
  showLabel?: boolean;
}) {
  const config = {
    RISING: { Icon: TrendingUp, tone: "text-risk-high", label: "Rising" },
    FALLING: { Icon: TrendingDown, tone: "text-risk-low", label: "Falling" },
    STABLE: { Icon: Minus, tone: "text-muted-foreground", label: "Stable" },
  }[trend.direction];

  const { Icon, tone, label } = config;
  return (
    <span
      title={TREND_COPY[trend.direction]}
      className={cn("inline-flex items-center gap-1.5 text-sm font-medium", tone, className)}
    >
      <Icon className="h-4 w-4" aria-hidden />
      {showLabel && <span>{label}</span>}
    </span>
  );
}
