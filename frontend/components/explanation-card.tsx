"use client";

import { useState } from "react";
import { AlertTriangle, Loader2, Sparkles, Wrench } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import { formatTimestamp } from "@/lib/format";
import type { Explanation, MachineDetail } from "@/types/api";

import { EvidencePanel } from "@/components/evidence-panel";

/**
 * The explanation is generated on demand by the backend, which calls Claude at
 * runtime with a structured evidence object. Nothing here is templated - the
 * button triggers a real request, and the evidence used is rendered underneath
 * so the plant team can check every number the AI quotes.
 */
export function ExplanationCard({
  machine,
  asOf,
}: {
  machine: MachineDetail;
  asOf: string | null;
}) {
  const [state, setState] = useState<"idle" | "loading" | "done" | "error">("idle");
  const [result, setResult] = useState<Explanation | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function generate() {
    setState("loading");
    setError(null);
    try {
      const data = await api.explanation(machine.machine_id, asOf);
      setResult(data);
      setState("done");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not generate an explanation.");
      setState("error");
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" aria-hidden />
              Why this machine is flagged
            </CardTitle>
            <CardDescription>
              A plain-English summary written from this machine&apos;s actual sensor evidence.
            </CardDescription>
          </div>
          <Button onClick={generate} disabled={state === "loading"} size="sm">
            {state === "loading" ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                Analysing sensor evidence...
              </>
            ) : state === "done" ? (
              "Regenerate"
            ) : (
              "Generate explanation"
            )}
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {state === "idle" && (
          <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
            Generate an explanation to see what the model is reacting to on {machine.machine_id},
            together with the readings it used.
          </p>
        )}

        {state === "loading" && (
          <div className="space-y-3">
            <div className="h-4 w-2/3 animate-pulse rounded bg-muted" />
            <div className="h-4 w-full animate-pulse rounded bg-muted" />
            <div className="h-4 w-5/6 animate-pulse rounded bg-muted" />
            <div className="h-4 w-1/2 animate-pulse rounded bg-muted" />
          </div>
        )}

        {state === "error" && (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {state === "done" && result && (
          <div className="space-y-4">
            {result.warning && (
              <Alert variant="warning">
                <AlertTriangle aria-hidden />
                <AlertDescription>{result.warning}</AlertDescription>
              </Alert>
            )}

            <div>
              <p className="text-base font-semibold leading-snug">{result.headline}</p>
              <p className="mt-2 leading-relaxed text-foreground/90">{result.explanation}</p>
            </div>

            <div className="rounded-md border bg-muted/40 p-4">
              <p className="flex items-center gap-2 text-sm font-semibold">
                <Wrench className="h-4 w-4 text-primary" aria-hidden />
                Suggested next step
              </p>
              <p className="mt-1.5 text-sm leading-relaxed">{result.recommended_action}</p>
            </div>

            {result.confidence_note && (
              <p className="text-sm italic text-muted-foreground">{result.confidence_note}</p>
            )}

            <EvidencePanel machine={machine} />

            <p className="border-t pt-3 text-xs text-muted-foreground">
              {result.source === "llm"
                ? `Generated ${formatTimestamp(result.generated_at)} by ${result.model}, from the evidence shown above.`
                : `Assembled ${formatTimestamp(result.generated_at)} directly from the evidence shown above (AI service unavailable).`}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
