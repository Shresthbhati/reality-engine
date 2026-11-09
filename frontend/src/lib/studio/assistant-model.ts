export interface ToolSchema {
  name: string;
  label?: string;
  description: string;
  mutating: boolean;
  parameters: Record<string, unknown>;
}

export interface CallResult {
  tool: string;
  result: unknown;
  error?: string;
}

export interface PlanResult {
  plan_id: string;
  operation: Record<string, unknown>;
  base_version_id: string | null;
  preview: unknown;
  warnings: string[];
  requires_confirmation: boolean;
  expires_at: string;
}

export interface ConfirmResult {
  plan_id: string;
  status: string;
  version_id: string | null;
  entity_ids_added: string[];
  preview: unknown;
}

export interface AskResult {
  text?: string;
  answer?: string;
  [key: string]: unknown;
}

export interface CapturePlanResult {
  [key: string]: unknown;
}

const baseUrl = "/api";

export async function fetchToolSchemas(): Promise<ToolSchema[]> {
  const res = await fetch(`${baseUrl}/assistant/tools`, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`Failed to load tools: ${res.status}`);
  const data = await res.json();
  return data.tools as ToolSchema[];
}

export async function callAssistantTool(
  worldId: string,
  tool: string,
  args: Record<string, unknown>
): Promise<CallResult> {
  const res = await fetch(`${baseUrl}/worlds/${worldId}/assistant/call`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ tool, args }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Failed to call ${tool}: ${res.status}`);
  }
  return res.json() as Promise<CallResult>;
}

export async function confirmAssistantPlan(
  worldId: string,
  planId: string
): Promise<ConfirmResult> {
  const res = await fetch(
    `${baseUrl}/worlds/${worldId}/assistant/confirm`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ plan_id: planId }),
    }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Failed to confirm plan: ${res.status}`);
  }
  return res.json() as Promise<ConfirmResult>;
}

export async function askAssistant(
  worldId: string,
  query: string
): Promise<AskResult> {
  const res = await fetch(`${baseUrl}/worlds/${worldId}/assistant/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Failed to ask: ${res.status}`);
  }
  return res.json() as Promise<AskResult>;
}

export async function planCapture(
  worldId: string,
  params?: { max_recommendations?: number; camera_height_m?: number; camera_fov_deg?: number }
): Promise<CapturePlanResult> {
  const res = await fetch(`${baseUrl}/worlds/${worldId}/assistant/capture-plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(params ?? {}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Failed to plan capture: ${res.status}`);
  }
  return res.json() as Promise<CapturePlanResult>;
}
