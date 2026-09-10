import Link from "next/link";
import { Gauge } from "lucide-react";

export function PageHeader({
  title,
  subtitle,
  right,
  backHref,
  backLabel,
}: {
  title: string;
  subtitle?: string;
  right?: React.ReactNode;
  backHref?: string;
  backLabel?: string;
}) {
  return (
    <header className="border-b bg-card">
      <div className="container py-5">
        {backHref && (
          <Link
            href={backHref}
            className="mb-3 inline-block text-sm text-primary hover:underline"
          >
            &larr; {backLabel ?? "Back"}
          </Link>
        )}
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 rounded-md bg-primary/10 p-2 text-primary">
              <Gauge className="h-5 w-5" aria-hidden />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
              {subtitle && <p className="mt-1 text-muted-foreground">{subtitle}</p>}
            </div>
          </div>
          {right}
        </div>
      </div>
    </header>
  );
}
