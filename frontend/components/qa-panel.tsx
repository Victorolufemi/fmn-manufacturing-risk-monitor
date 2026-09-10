"use client";

import { useState } from "react";
import { AlertTriangle, Loader2, MessageSquareText, Send } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import type { QAResponse } from "@/types/api";

/** Renders the small subset of markdown the answers actually use: bullet lists. */
function AnswerText({ text }: { text: string }) {
  const lines = text.split("\n").filter((l) => l.trim().length > 0);
  const blocks: { type: "p" | "ul"; items: string[] }[] = [];

  for (const line of lines) {
    const isBullet = /^\s*[-*]\s+/.test(line);
    const content = isBullet ? line.replace(/^\s*[-*]\s+/, "") : line;
    const last = blocks[blocks.length - 1];
    if (isBullet && last?.type === "ul") last.items.push(content);
    else blocks.push({ type: isBullet ? "ul" : "p", items: [content] });
  }

  return (
    <div className="space-y-2 leading-relaxed">
      {blocks.map((b, i) =>
        b.type === "ul" ? (
          <ul key={i} className="ml-5 list-disc space-y-1">
            {b.items.map((it, j) => (
              <li key={j}>{it.replace(/\*\*/g, "")}</li>
            ))}
          </ul>
        ) : (
          <p key={i}>{b.items[0].replace(/\*\*/g, "")}</p>
        ),
      )}
    </div>
  );
}

export function QaPanel({
  asOf,
  suggestions,
  defaultQuestion,
}: {
  asOf: string | null;
  suggestions: string[];
  defaultQuestion?: string;
}) {
  const [question, setQuestion] = useState(defaultQuestion ?? "");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<QAResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function ask(q: string) {
    const trimmed = q.trim();
    if (!trimmed || busy) return;
    setQuestion(trimmed);
    setBusy(true);
    setError(null);
    try {
      setAnswer(await api.ask(trimmed, asOf));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not answer that question.");
      setAnswer(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <MessageSquareText className="h-4 w-4 text-primary" aria-hidden />
          Ask about the fleet
        </CardTitle>
        <CardDescription>
          Answers are drawn only from the current machine data. The assistant is not able to
          use anything outside it.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void ask(question);
          }}
          className="flex gap-2"
        >
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. Which machines need attention right now?"
            aria-label="Question about the machine fleet"
            maxLength={500}
          />
          <Button type="submit" disabled={busy || question.trim().length < 2}>
            {busy ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
            ) : (
              <Send className="h-4 w-4" aria-hidden />
            )}
            <span className="sr-only sm:not-sr-only">Ask</span>
          </Button>
        </form>

        <div className="flex flex-wrap gap-2">
          {suggestions.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => void ask(s)}
              disabled={busy}
              className="rounded-full border bg-card px-3 py-1 text-xs text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-50"
            >
              {s}
            </button>
          ))}
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertTriangle aria-hidden />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {busy && (
          <div className="space-y-2 rounded-md border p-4">
            <div className="h-4 w-1/2 animate-pulse rounded bg-muted" />
            <div className="h-4 w-full animate-pulse rounded bg-muted" />
            <div className="h-4 w-4/5 animate-pulse rounded bg-muted" />
          </div>
        )}

        {!busy && answer && (
          <div className="rounded-md border bg-muted/30 p-4">
            <p className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
              {answer.question}
            </p>
            <AnswerText text={answer.answer} />
            {answer.warning && (
              <p className="mt-3 rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                {answer.warning}
              </p>
            )}
            <p className="mt-3 border-t pt-2 text-xs text-muted-foreground">
              {answer.source === "llm"
                ? `Answered by ${answer.model} from retrieved machine data.`
                : "Assembled directly from retrieved machine data (AI service unavailable)."}
              {answer.machines_referenced.length > 0 &&
                ` Machines referenced: ${answer.machines_referenced.join(", ")}.`}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
