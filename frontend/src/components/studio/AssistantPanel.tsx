"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  callAssistantTool,
  confirmAssistantPlan,
  fetchToolSchemas,
  planCapture,
  askAssistant,
  CallResult,
  PlanResult,
  ConfirmResult,
  ToolSchema,
} from "@/lib/studio/assistant-model";

export interface AssistantMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  result?: CallResult;
  plan?: PlanResult;
  confirm?: ConfirmResult;
}

export default function AssistantPanel({
  worldId,
  onClose,
}: {
  worldId: string;
  onClose: () => void;
}) {
  const [schemas, setSchemas] = useState<ToolSchema[]>([]);
  const [loadingSchemas, setLoadingSchemas] = useState(false);
  const [messages, setMessages] = useState<AssistantMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [selectedTool, setSelectedTool] = useState<string | null>(null);
  const [toolArgs, setToolArgs] = useState<Record<string, unknown>>({});

  useEffect(() => {
    setLoadingSchemas(true);
    fetchToolSchemas()
      .then(setSchemas)
      .catch(() => setSchemas([]))
      .finally(() => setLoadingSchemas(false));
  }, [worldId]);

  const addMessage = (msg: Omit<AssistantMessage, "id" | "timestamp">) => {
    const id = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { ...msg, id, timestamp: new Date().toISOString() },
    ]);
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;

    setInput("");
    setSending(true);

    addMessage({ role: "user", content: text });

    try {
      const res = await askAssistant(worldId, text);
      addMessage({
        role: "assistant",
        content: res.text || res.answer || "No response",
      });
    } catch (err) {
      addMessage({
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      setSending(false);
    }
  };

  const handleToolCall = async (schema: ToolSchema) => {
    setSelectedTool(schema.name);
    // For now, run a generic ask that mentions the tool
    try {
      const res = await askAssistant(
        worldId,
        `Run the ${schema.name} tool. ${schema.description}`
      );
      addMessage({
        role: "assistant",
        content: res.text || res.answer || "Tool executed",
      });
    } catch (err) {
      addMessage({
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      setSelectedTool(null);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0f172a] text-slate-100">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-slate-500 text-sm">
            No conversation yet. Ask about your world or run a tool.
          </div>
        ) : (
          messages.map((m) => (
            <div
              key={m.id}
              className={`p-3 rounded-lg max-w-[90%] ${
                m.role === "user"
                  ? "bg-blue-600/30 text-blue-100 ml-auto border border-blue-500/30"
                  : "bg-slate-700/50 text-slate-200 border border-slate-700/50"
              }`}
            >
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-bold uppercase opacity-60">
                  {m.role}
                </span>
                <span className="text-[10px] text-slate-500">{m.timestamp.slice(11, 16)}</span>
              </div>
              <div className="text-sm whitespace-pre-wrap">{m.content}</div>
              {m.result && (
                <div className="mt-2 p-2 bg-slate-800 rounded text-xs text-slate-300">
                  Tool result: {JSON.stringify(m.result).slice(0, 200)}...
                </div>
              )}
              {m.plan && (
                <div className="mt-2 p-2 bg-amber-900/30 border border-amber-900/50 rounded text-xs text-amber-200">
                  Plan: {typeof m.plan.preview === "object" && m.plan.preview !== null && "warnings" in m.plan.preview ? (m.plan.preview as { warnings: string[] }).warnings.join(", ") : "Plan ready"}
                </div>
              )}
            </div>
          ))
        )}
      </div>
      <div className="border-t border-slate-700 p-3">
        {loadingSchemas ? (
          <div className="text-center text-slate-500 text-xs">Loading tools...</div>
        ) : (
          <>
            <div className="flex gap-2 mb-2">
              {schemas.map((s) => (
                <button
                  key={s.name}
                  onClick={() => handleToolCall(s)}
                  disabled={selectedTool === s.name}
                  className={`flex-1 flex flex-col items-center gap-1 px-2 py-2 rounded bg-slate-700 text-slate-200 hover:bg-slate-600 ${
                    selectedTool === s.name ? "bg-blue-600 text-blue-50" : ""
                  }`}
                >
                  <div className="w-4 h-4 mb-1" />
                  <span className="text-[10px] truncate">{s.name}</span>
                </button>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder="Ask about your world..."
                className="flex-1 px-3 py-2 rounded bg-slate-800 text-slate-100 border border-slate-600 focus:outline-none focus:border-blue-500"
                disabled={sending}
              />
              <button
                onClick={handleSend}
                disabled={sending || !input.trim()}
                className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-500 disabled:opacity-50"
              >
                Ask
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
