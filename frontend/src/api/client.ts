import type { FinalAnswer, Mode, SessionDetails, UploadFileStatus } from "../types";
import { parseSSEStream, type SSEMessage } from "./sse";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export class ApiError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError("Could not reach the backend. Check that the API is running and try again.");
  }

  if (!response.ok) {
    let message = `Backend request failed (${response.status}).`;
    try {
      const body = await response.json() as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the HTTP status message when the backend doesn't return JSON.
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export function getHealth(): Promise<{ status: string }> {
  return requestJson("/health");
}

export function createSession(): Promise<{ session_id: string }> {
  return requestJson("/sessions", { method: "POST" });
}

export function getSession(sessionId: string): Promise<SessionDetails> {
  return requestJson(`/sessions/${encodeURIComponent(sessionId)}`);
}

export function deleteSession(sessionId: string): Promise<{ deleted: boolean }> {
  return requestJson(`/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
}

export async function uploadFiles(
  sessionId: string,
  files: File[],
): Promise<{ files: UploadFileStatus[]; total_files: number }> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file, file.name));
  return requestJson(`/sessions/${encodeURIComponent(sessionId)}/files`, {
    method: "POST",
    body,
  });
}

export async function streamAsk(
  question: string,
  mode: Mode,
  sessionId: string | null,
  signal: AbortSignal,
  onMessage: (message: SSEMessage) => void,
): Promise<FinalAnswer> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode, session_id: sessionId }),
      signal,
    });
  } catch (error) {
    if (signal.aborted) throw error;
    throw new ApiError("Could not reach the backend. Check the API and CORS settings, then try again.");
  }

  if (!response.ok) {
    let message = `Backend request failed (${response.status}).`;
    try {
      const body = await response.json() as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the HTTP status message when the backend doesn't return JSON.
    }
    throw new ApiError(message, response.status);
  }
  if (!response.body) throw new ApiError("The backend did not provide a response stream.");

  let finalAnswer: FinalAnswer | null = null;
  for await (const message of parseSSEStream(response.body, signal)) {
    onMessage(message);
    if (message.event === "error") {
      let error = "The backend could not complete this question.";
      try {
        const data = JSON.parse(message.data) as { message?: string };
        if (data.message) error = data.message;
      } catch {
        // Retain a clear fallback if an error event is malformed.
      }
      throw new ApiError(error);
    }
    if (message.event === "final") {
      try {
        finalAnswer = JSON.parse(message.data) as FinalAnswer;
      } catch {
        throw new ApiError("The backend returned an unreadable final answer.");
      }
    }
  }

  if (!finalAnswer) throw new ApiError("The answer stream ended before a final answer arrived.");
  return finalAnswer;
}
