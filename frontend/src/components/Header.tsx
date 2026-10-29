import type { Mode } from "../types";

type Props = {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
  health: "checking" | "online" | "offline";
  theme: "light" | "dark";
  onThemeToggle: () => void;
};

export function Header({ mode, onModeChange, health, theme, onThemeToggle }: Props) {
  return (
    <header className="topbar">
      <a className="brand" href="#" aria-label="Knowledge Studio home">
        <span className="brand-mark" aria-hidden="true">✦</span>
        <span>
          <strong>KNOWLEDGE STUDIO</strong>
          <small>Self-correcting RAG</small>
        </span>
      </a>

      <div className="header-controls">
        <div className="health-status" role="status" aria-live="polite" aria-label={`Backend status: ${health}`}>
          <span className={`health-dot health-${health}`} aria-hidden="true" />
          <span>{health === "online" ? "Backend online" : health === "offline" ? "Backend unavailable" : "Checking backend"}</span>
        </div>
        <div className="mode-switch" role="group" aria-label="Knowledge source">
          <button
            type="button"
            className={mode === "library" ? "selected" : ""}
            aria-pressed={mode === "library"}
            onClick={() => onModeChange("library")}
          >
            Library
          </button>
          <button
            type="button"
            className={mode === "uploads" ? "selected" : ""}
            aria-pressed={mode === "uploads"}
            onClick={() => onModeChange("uploads")}
          >
            My PDFs
          </button>
        </div>
        <button
          className="icon-button theme-toggle"
          type="button"
          onClick={onThemeToggle}
          aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
        >
          {theme === "light" ? "☾" : "☀"}
        </button>
      </div>
    </header>
  );
}
