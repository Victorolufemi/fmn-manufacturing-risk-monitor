"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { Clock, RotateCcw, Zap } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { InfoTip } from "@/components/ui/tooltip";
import { formatTimestamp, toLocalInputValue } from "@/lib/format";

/**
 * The dataset is a fixed historical extract, so "now" has to be explicit.
 * Everything on screen is filtered to at-or-before this instant - the app never
 * shows a user information from after the moment they are reviewing.
 */
export function AsOfControl({
  asOf,
  dataRange,
  jumpTo,
  jumpLabel,
  basePath,
}: {
  asOf: string;
  dataRange: { start: string; end: string };
  jumpTo?: string | null;
  jumpLabel?: string;
  basePath: string;
}) {
  const router = useRouter();
  const params = useSearchParams();
  const [value, setValue] = useState(toLocalInputValue(asOf));

  function navigate(next: string | null) {
    const q = new URLSearchParams(params.toString());
    if (next) q.set("as_of", next);
    else q.delete("as_of");
    const qs = q.toString();
    router.push(qs ? `${basePath}?${qs}` : basePath);
  }

  const isLatest = !params.get("as_of");

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-lg border bg-card px-3 py-2">
      <div className="flex items-center gap-2 text-sm">
        <Clock className="h-4 w-4 text-muted-foreground" aria-hidden />
        <span className="font-medium">Reviewing as of</span>
        <InfoTip label="About the review time">
          This tool reads from a fixed plant data extract covering{" "}
          {formatTimestamp(dataRange.start)} to {formatTimestamp(dataRange.end)}. Move this
          control to review the fleet as it stood at any earlier point; nothing after the
          selected time is used.
        </InfoTip>
      </div>

      <Input
        type="datetime-local"
        value={value}
        min={toLocalInputValue(dataRange.start)}
        max={toLocalInputValue(dataRange.end)}
        onChange={(e) => setValue(e.target.value)}
        onBlur={() => value && navigate(value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && value) navigate(value);
        }}
        aria-label="Review time"
        className="h-8 w-auto min-w-[13rem] text-sm"
      />

      {!isLatest && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => {
            setValue(toLocalInputValue(dataRange.end));
            navigate(null);
          }}
        >
          <RotateCcw className="h-3.5 w-3.5" aria-hidden />
          Latest
        </Button>
      )}

      {jumpTo && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => {
            const v = toLocalInputValue(jumpTo);
            setValue(v);
            navigate(v);
          }}
        >
          <Zap className="h-3.5 w-3.5" aria-hidden />
          {jumpLabel ?? "Go to last alert"}
        </Button>
      )}
    </div>
  );
}
