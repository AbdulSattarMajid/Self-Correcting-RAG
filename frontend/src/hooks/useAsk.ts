import { useCallback, useRef, useState } from "react";
import { ApiError, streamAsk } from "../api/client";
import type { Mode, PipelineNode, PipelineProgress } from "../types";

type AskResult =
  | { status: "complete"; answer: Awaited<ReturnType<typeof streamAsk>> }
  | { status: "error"; message: string; code?: number }
  | { status: "cancelled" };

const initialProgress: PipelineProgress = {
  completed: [],
  attempt: 0,
  webSearch: false,
  activeNode: null,
};

function isPipelineNode(value: unknown): value is PipelineNode {
  if (!value || typeof value !== "object" || !("node" in value)) return false;
  return ["retrieve", "grade_documents", "generate", "grade_generation", "web_search"].includes(
    String(value.node),
  );
}

export function useAsk() {
  const controllerRef = useRef<AbortController | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<PipelineProgress>(initialProgress);

  const cancel = useCallback(() => controllerRef.current?.abort(), []);

  const ask = useCallback(async (
    question: string,
    mode: Mode,
    sessionId: string | null,
  ): Promise<AskResult> => {
    if (controllerRef.current) return { status: "error", message: "A question is already running." };

    const controller = new AbortController();
    controllerRef.current = controller;
    setRunning(true);
    setProgress({ ...initialProgress });

    try {
      const answer = await streamAsk(question, mode, sessionId, controller.signal, (message) => {
        if (message.event !== "node") return;
        try {
          const node = JSON.parse(message.data) as unknown;
          if (!isPipelineNode(node)) return;
          setProgress((current) => {
            const completed = current.completed.includes(node.node)
              ? current.completed
              : [...current.completed, node.node];
            return {
              completed,
              attempt: node.node === "generate" ? node.attempt : current.attempt,
              webSearch: current.webSearch || node.node === "web_search",
              activeNode: node.node,
            };
          });
        } catch {
          // Ignore malformed progress events; the final event still determines the answer.
        }
      });
      return { status: "complete", answer };
    } catch (error) {
      if (controller.signal.aborted) return { status: "cancelled" };
      return {
        status: "error",
        message: error instanceof ApiError
          ? error.message
          : "Something went wrong while asking the backend.",
        code: error instanceof ApiError ? error.status : undefined,
      };
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null;
      setRunning(false);
    }
  }, []);

  return { ask, cancel, running, progress };
}
