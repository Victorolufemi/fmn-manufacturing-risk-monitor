import { AlertTriangle, CheckCircle2, CircleAlert, Eye } from "lucide-react";

import { cn } from "@/lib/utils";
import { RISK_COPY } from "@/lib/format";
import type { RiskLevel } from "@/types/api";

const STYLES: Record<RiskLevel, string> = {
  CRITICAL: "bg-risk-critical text-white border-risk-critical",
  HIGH: "bg-risk-high text-white border-risk-high",
  MEDIUM: "bg-risk-medium/12 text-risk-medium border-risk-medium/40",
  LOW: "bg-risk-low/10 text-risk-low border-risk-low/30",
};

const ICONS: Record<RiskLevel, typeof AlertTriangle> = {
  CRITICAL: CircleAlert,
  HIGH: AlertTriangle,
  MEDIUM: Eye,
  LOW: CheckCircle2,
};

export function RiskBadge({
  level,
  className,
  size = "default",
}: {
  level: RiskLevel;
  className?: string;
  size?: "default" | "lg";
}) {
  const Icon = ICONS[level];
  return (
    <span
      title={RISK_COPY[level].meaning}
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border font-semibold uppercase tracking-wide",
        size === "lg" ? "px-3 py-1 text-sm" : "px-2 py-0.5 text-[11px]",
        STYLES[level],
        className,
      )}
    >
      <Icon className={size === "lg" ? "h-4 w-4" : "h-3 w-3"} aria-hidden />
      {RISK_COPY[level].label}
    </span>
  );
}
