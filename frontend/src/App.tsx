import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, createSession, deleteSession, getHealth, getSession, uploadFiles } from "./api/client";
import { ChatComposer } from "./components/ChatComposer";
import { Conversation } from "./components/Conversation";
import { Header } from "./components/Header";
import { UploadPanel } from "./components/UploadPanel";
import { useAsk } from "./hooks/useAsk";
import type { ConversationItem, Mode, UploadFileStatus } from "./types";

const examples = [
  "What is the Saga pattern?",
  "How does event-driven architecture work?",
  "When should a system use microservices?",
];

type Health = "checking" | "online" | "offline";
type Theme = "light" | "dark";
type LocalUploadStatus = UploadFileStatus & { pending?: boolean };

function newId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getStoredSession(): string | null {
  try {
    return window.sessionStorage.getItem("crag-session-id");
  } catch {
    return null;
  }
}

function setStoredSession(value: string | null) {
  try {
    if (value) window.sessionStorage.setItem("crag-session-id", value);
    else window.sessionStorage.removeItem("crag-session-id");
  } catch {
    // The active session remains usable when browser storage is unavailable.
  }
}

export default function App() {
  const [mode, setMode] = useState<Mode>("library");
  const [health, setHealth] = useState<Health>("checking");
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const stored = window.localStorage.getItem("crag-theme");
      if (stored === "dark" || stored === "light") return stored;
    } catch {
      // Fall back to the operating system theme.
    }
    return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  const [sessionId, setSessionId] = useState<string | null>(getStoredSession);
  const skipSessionRestoreRef = useRef(false);
  const [uploadedFiles, setUploadedFiles] = useState<LocalUploadStatus[]>([]);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);
  const [messages, setMessages] = useState<ConversationItem[]>([]);
  const { ask, cancel, running, progress } = useAsk();

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      window.localStorage.setItem("crag-theme", theme);
    } catch {
      // Theme selection still applies until the page reloads.
    }
  }, [theme]);

  useEffect(() => {
    let mounted = true;
    const check = () => {
      void getHealth().then(
        () => mounted && setHealth("online"),
        () => mounted && setHealth("offline"),
      );
    };
    check();
    const interval = window.setInterval(check, 25_000);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    if (skipSessionRestoreRef.current) {
      skipSessionRestoreRef.current = false;
      return;
    }
    let mounted = true;
    void getSession(sessionId).then((session) => {
      if (!mounted) return;
      setUploadedFiles(session.files.map((file) => ({ ...file, ok: true })));
      setUploadNotice(null);
    }).catch((error: unknown) => {
      if (!mounted) return;
      if (error instanceof ApiError && error.status === 404) {
        setSessionId(null);
        setStoredSession(null);
        setUploadedFiles([]);
        setUploadNotice("This upload session is no longer available. Re-upload your PDFs to continue.");
      } else {
        setUploadNotice("Could not restore your PDF session. Check the backend connection and try again.");
      }
    });
    return () => {
      mounted = false;
    };
  }, [sessionId]);

  const successfulFiles = useMemo(() => uploadedFiles.filter((file) => file.ok), [uploadedFiles]);

  const handleUpload = useCallback(async (valid: File[], invalid: UploadFileStatus[]) => {
    setUploadNotice(null);
    if (invalid.length) setUploadedFiles((current) => [...current.filter((file) => !file.pending), ...invalid]);
    if (!valid.length) return;

    const pending = valid.map((file) => ({ name: file.name, pages: 0, ok: false, pending: true }));
    setUploadedFiles((current) => [...current.filter((file) => !file.pending), ...pending]);
    setUploadBusy(true);
    let activeSessionId = sessionId;

    try {
      if (!activeSessionId) {
        const created = await createSession();
        activeSessionId = created.session_id;
        skipSessionRestoreRef.current = true;
        setSessionId(activeSessionId);
        setStoredSession(activeSessionId);
      }
      const result = await uploadFiles(activeSessionId, valid);
      setUploadedFiles((current) => [
        ...current.filter((file) => !file.pending),
        ...result.files,
      ]);
      if (result.files.some((file) => !file.ok)) {
        setUploadNotice("Some files could not be added. Check the file status for details.");
      }
    } catch (error) {
      if (error instanceof ApiError && error.status === 404 && activeSessionId) {
        setSessionId(null);
        setStoredSession(null);
        setUploadedFiles((current) => current.filter((file) => !file.ok && !file.pending));
        setUploadNotice("The upload session expired or the backend restarted. Re-upload your PDFs.");
      } else {
        const message = error instanceof Error ? error.message : "The files could not be uploaded.";
        setUploadedFiles((current) => [
          ...current.filter((file) => !file.pending),
          ...valid.map((file) => ({ name: file.name, pages: 0, ok: false, error: message })),
        ]);
        setUploadNotice(message);
      }
    } finally {
      setUploadBusy(false);
    }
  }, [sessionId]);

  const handleRemoveAll = useCallback(async () => {
    setUploadNotice(null);
    if (sessionId) {
      try {
        await deleteSession(sessionId);
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 404)) {
          setUploadNotice(error instanceof Error ? error.message : "Could not remove the PDF session.");
          return;
        }
      }
    }
    setSessionId(null);
    setStoredSession(null);
    setUploadedFiles([]);
  }, [sessionId]);

  async function handleQuestion(question: string) {
    const assistantId = newId();
    setMessages((current) => [
      ...current,
      { id: newId(), role: "user", content: question },
      { id: assistantId, role: "assistant", content: "", status: "pending" },
    ]);

    const result = await ask(question, mode, sessionId);
    const expiredUploadSession = result.status === "error" && result.code === 404 && mode === "uploads";
    if (expiredUploadSession) {
      setSessionId(null);
      setStoredSession(null);
      setUploadedFiles([]);
      setUploadNotice("This upload session is no longer available. Re-upload your PDFs to continue.");
    }
    setMessages((current) => current.map((message) => {
      if (message.id !== assistantId) return message;
      if (result.status === "complete") {
        return {
          ...message,
          content: result.answer.answer,
          status: "complete",
          grounded: result.answer.grounded,
          critic_checks: result.answer.critic_checks,
          sources: result.answer.sources,
        };
      }
      if (result.status === "cancelled") {
        return { ...message, status: "cancelled", content: "" };
      }
      if (expiredUploadSession) {
        return {
          ...message,
          status: "error",
          content: "This upload session is no longer available. Re-upload your PDFs and try again.",
        };
      }
      return { ...message, status: "error", content: result.message };
    }));
  }

  async function copyAnswer(text: string) {
    await navigator.clipboard.writeText(text);
  }

  function clearChat() {
    if (running) cancel();
    setMessages([]);
  }

  const uploadsReady = successfulFiles.length > 0;
  return (
    <div className="app-shell">
      <Header
        mode={mode}
        onModeChange={setMode}
        health={health}
        theme={theme}
        onThemeToggle={() => setTheme((current) => current === "light" ? "dark" : "light")}
      />

      <main className="workspace">
        <aside className="sidebar">
          <div className="workspace-intro">
            <p className="eyebrow">Research workspace</p>
            <h1>Ask a question.<br />Follow the evidence.</h1>
            <p>
              {mode === "library"
                ? "Search the architecture library and use web sources when more context is needed."
                : "Ask questions grounded only in PDFs indexed for this session."}
            </p>
          </div>
          {mode === "uploads" ? (
            <UploadPanel
              files={uploadedFiles}
              busy={uploadBusy}
              notice={uploadNotice}
              onUpload={(files, invalid) => void handleUpload(files, invalid)}
              onRemoveAll={() => void handleRemoveAll()}
            />
          ) : (
            <section className="library-panel">
              <p className="eyebrow">Knowledge source</p>
              <h2>Architecture library</h2>
              <p>Hybrid retrieval, relevance grading, and answer verification work together to ground each response.</p>
              <div className="feature-list">
                <span><b>01</b> Retrieve relevant passages</span>
                <span><b>02</b> Grade evidence</span>
                <span><b>03</b> Verify the answer</span>
              </div>
            </section>
          )}
          <div className="sidebar-bottom">
            <span className="privacy-mark" aria-hidden="true">◇</span>
            <span>{mode === "uploads" ? "Session-only PDFs · not added to the library" : "Sources shown with every answer"}</span>
          </div>
        </aside>

        <section className="chat-pane" aria-label="Research chat">
          <div className="chat-toolbar">
            <div>
              <p className="eyebrow">{mode === "library" ? "Architecture library" : "My PDFs"}</p>
              <h2>{mode === "library" ? "Research assistant" : "Ask your documents"}</h2>
            </div>
            <button type="button" className="text-button clear-button" onClick={clearChat} disabled={messages.length === 0}>
              Clear chat
            </button>
          </div>

          {health === "offline" && (
            <div className="backend-warning" role="alert">
              The backend is unavailable. Check that FastAPI is running at the configured API URL.
            </div>
          )}

          {messages.length > 0 ? (
            <Conversation messages={messages} progress={progress} running={running} onCopy={copyAnswer} />
          ) : (
            <div className="empty-state">
              <span className="empty-mark" aria-hidden="true">✦</span>
              <h3>What would you like to understand?</h3>
              <p>Try a question below, or ask your own.</p>
              <div className="example-list">
                {examples.map((example) => (
                  <button
                    type="button"
                    key={example}
                    disabled={mode === "uploads" && !uploadsReady}
                    onClick={() => void handleQuestion(example)}
                  >
                    <span>{example}</span><span aria-hidden="true">↗</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="composer-dock">
            {mode === "uploads" && !uploadsReady && (
              <p className="composer-notice">Upload at least one PDF to ask a question.</p>
            )}
            <ChatComposer
              disabled={health !== "online" || (mode === "uploads" && !uploadsReady)}
              running={running}
              placeholder={mode === "uploads" ? "Ask a question about your PDFs…" : "Ask about software architecture…"}
              onSubmit={(question) => void handleQuestion(question)}
              onCancel={cancel}
            />
          </div>
        </section>
      </main>
    </div>
  );
}
