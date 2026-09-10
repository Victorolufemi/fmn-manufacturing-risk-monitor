import { Suspense } from "react";

import { Skeleton } from "@/components/ui/skeleton";

import { DashboardClient } from "./dashboard-client";

export default function ManufacturingPage() {
  return (
    <Suspense
      fallback={
        <main className="container space-y-6 py-10">
          <Skeleton className="h-24" />
          <Skeleton className="h-96" />
        </main>
      }
    >
      <DashboardClient />
    </Suspense>
  );
}
