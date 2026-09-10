import { Suspense } from "react";

import { Skeleton } from "@/components/ui/skeleton";

import { MachineClient } from "./machine-client";

export default async function MachinePage({
  params,
}: {
  params: Promise<{ machineId: string }>;
}) {
  const { machineId } = await params;
  return (
    <Suspense
      fallback={
        <main className="container space-y-6 py-10">
          <Skeleton className="h-24" />
          <Skeleton className="h-96" />
        </main>
      }
    >
      <MachineClient machineId={decodeURIComponent(machineId)} />
    </Suspense>
  );
}
