import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Source } from "../types";
import { remarkCitationChips } from "./citations";

type Props = {
  answer: string;
  grounded: "yes" | "no";
  criticChecks: number;
  sources: Source[];
};

export const markdownComponents: Components = {
  a({ href, children, title }) {
    const citationPrefix = "https://citation.local/";
    if (href?.startsWith(citationPrefix)) {
      const label = decodeURIComponent(href.slice(citationPrefix.length));
      const separator = label.match(/,\s*pages?\s+/i);
      const file = separator ? label.slice(0, separator.index) : label;
      const shortName = file.length > 24 ? `${file.slice(0, 12)}…${file.slice(-8)}` : file;
      const suffix = separator ? label.slice(separator.index) : "";
      return (
        <span className="citation-chip" title={label}>
          <span>{shortName}</span>{suffix && <small>{suffix}</small>}
        </span>
      );
    }
    return <a href={href} title={title} target="_blank" rel="noreferrer">{children}</a>;
  },
};

export function AnswerCard({ answer, grounded, criticChecks, sources }: Props) {
  const notFound = answer.toLowerCase().includes("no relevant passage was found");
  return (
    <div className="answer-card">
      {notFound ? (
        <div className="calm-info" role="status">{answer}</div>
      ) : (
        <>
          <div className="answer-meta">
            <span className={`grounding-badge ${grounded === "yes" ? "grounded" : "ungrounded"}`}>
              <span aria-hidden="true">{grounded === "yes" ? "✓" : "!"}</span>
              {grounded === "yes" ? "Grounded" : "Could not fully verify"}
            </span>
            <span className="critic-count">critic checks: {criticChecks}</span>
          </div>
          <div className="markdown-answer" aria-live="polite">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkCitationChips]}
              components={markdownComponents}
            >
              {answer}
            </ReactMarkdown>
          </div>
        </>
      )}
      {sources.length > 0 && (
        <details className="sources">
          <summary>Sources / retrieved context <span>({sources.length})</span></summary>
          <ul>
            {sources.map((source, index) => (
              <li key={`${source.type}-${index}`}>
                <span className={`source-type source-${source.type}`}>{source.type}</span>
                <p>{source.text}</p>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
