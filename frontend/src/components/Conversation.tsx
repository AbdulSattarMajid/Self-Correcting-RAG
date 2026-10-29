import { useState } from "react";
import type { ConversationItem, PipelineProgress } from "../types";
import { AnswerCard } from "./AnswerCard";
import { Progress } from "./Progress";

type Props = {
  messages: ConversationItem[];
  progress: PipelineProgress;
  running: boolean;
  onCopy: (text: string) => void;
};

export function Conversation({ messages, progress, running, onCopy }: Props) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [copyErrorId, setCopyErrorId] = useState<string | null>(null);

  async function copyAnswer(message: ConversationItem) {
    if (!message.content || message.status !== "complete") return;
    try {
      await onCopy(message.content);
      setCopyErrorId(null);
      setCopiedId(message.id);
      window.setTimeout(() => setCopiedId((current) => current === message.id ? null : current), 1600);
    } catch {
      setCopyErrorId(message.id);
    }
  }

  return (
    <div className="conversation" aria-live="polite">
      {messages.map((message) => (
        <article className={`message message-${message.role}`} key={message.id}>
          <div className="message-avatar" aria-hidden="true">{message.role === "user" ? "You" : "✦"}</div>
          <div className="message-body">
            <div className="message-label">{message.role === "user" ? "You" : "Knowledge Studio"}</div>
            {message.role === "user" ? (
              <p className="user-question">{message.content}</p>
            ) : message.status === "pending" ? (
              <Progress progress={progress} />
            ) : message.status === "cancelled" ? (
              <div className="cancelled-message" role="status">This question was cancelled.</div>
            ) : message.status === "error" ? (
              <div className="error-card" role="alert">
                <strong>We couldn’t complete that request</strong>
                <p>{message.content}</p>
              </div>
            ) : (
              <>
                <AnswerCard
                  answer={message.content}
                  grounded={message.grounded ?? "no"}
                  criticChecks={message.critic_checks ?? 0}
                  sources={message.sources ?? []}
                />
                <button className="copy-button" type="button" onClick={() => void copyAnswer(message)}>
                  {copyErrorId === message.id ? "Clipboard access denied" : copiedId === message.id ? "Copied" : "Copy answer"}
                </button>
              </>
            )}
          </div>
          {running && message.status === "pending" && <span className="visually-hidden">Answer is being generated</span>}
        </article>
      ))}
    </div>
  );
}
