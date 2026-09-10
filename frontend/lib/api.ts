/**
 * Browser-side client for the FastAPI backend.
 *
 * The backend URL comes from NEXT_PUBLIC_API_URL so the same build runs against
 * a local server and against Render. No production URL is hardcoded, and no
 * secret ever reaches this file - every model, data and LLM call is made
 * server-side by the backend.
 */
import type {
  AppMetadata,
  Dashboard,
  Explanation,
  MachineDetail,
  MachineSummary,
  ModelInfo,
  QAResponse,
} from "@/types/api";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      `Cannot reach the risk service at ${API_BASE}. Check that the backend is running and that NEXT_PUBLIC_API_URL is correct.`,
      0,
    );
  }

  if (!res.ok) {
    let detail = `Request failed with status ${res.status}`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* keep the generic message */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

function withAsOf(path: string, asOf?: string | null, extra?: Record<string, string>) {
  const params = new URLSearchParams(extra ?? {});
  if (asOf) params.set("as_of", asOf);
  const qs = params.toString();
  return qs ? `${path}?${qs}` : path;
}

export const api = {
  metadata: () => request<AppMetadata>("/api/metadata"),

  modelInfo: () => request<ModelInfo>("/api/model-info"),

  dashboard: (asOf?: string | null) =>
    request<Dashboard>(withAsOf("/api/dashboard", asOf)),

  machines: (asOf?: string | null, filters?: Record<string, string>) =>
    request<MachineSummary[]>(withAsOf("/api/machines", asOf, filters)),

  machine: (machineId: string, asOf?: string | null, historyHours = 336) =>
    request<MachineDetail>(
      withAsOf(`/api/machines/${encodeURIComponent(machineId)}`, asOf, {
        history_hours: String(historyHours),
      }),
    ),

  explanation: (machineId: string, asOf?: string | null) =>
    request<Explanation>(
      withAsOf(`/api/machines/${encodeURIComponent(machineId)}/explanation`, asOf),
    ),

  ask: (question: string, asOf?: string | null) =>
    request<QAResponse>("/api/qa", {
      method: "POST",
      body: JSON.stringify({ question, as_of: asOf ?? null }),
    }),
};
