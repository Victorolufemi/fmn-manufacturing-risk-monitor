import { Info } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import type { ColdStartInfo } from "@/types/api";

/**
 * Shown wherever a limited-history machine is presented. The wording states
 * only what is actually measured - hours of history and which baseline was
 * used - and does not invent a confidence percentage.
 */
export function ColdStartNotice({ info }: { info: ColdStartInfo }) {
  if (!info.message) return null;
  return (
    <Alert variant="warning">
      <Info aria-hidden />
      <AlertTitle>
        {info.is_cold_start ? "Limited history" : "Borrowed baseline"}
      </AlertTitle>
      <AlertDescription>{info.message}</AlertDescription>
    </Alert>
  );
}

export function ColdStartTag({ hours }: { hours: number }) {
  return (
    <span
      title={`Only ${hours} hours of history. Risk estimates are less reliable than for established machines.`}
      className="inline-flex items-center rounded border border-amber-300 bg-amber-50 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-800"
    >
      New
    </span>
  );
}
