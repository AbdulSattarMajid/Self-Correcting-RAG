import { useRef, useState, type KeyboardEvent } from "react";

type Props = {
  disabled: boolean;
  running: boolean;
  placeholder: string;
  onSubmit: (question: string) => void;
  onCancel: () => void;
};

export function ChatComposer({ disabled, running, placeholder, onSubmit, onCancel }: Props) {
  const [question, setQuestion] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function submit() {
    const value = question.trim();
    if (!value || disabled || running) return;
    onSubmit(value);
    setQuestion("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <div className="composer-wrap">
      <label className="visually-hidden" htmlFor="question-input">Your question</label>
      <div className="composer">
        <textarea
          id="question-input"
          ref={textareaRef}
          value={question}
          onChange={(event) => {
            setQuestion(event.currentTarget.value);
            event.currentTarget.style.height = "auto";
            event.currentTarget.style.height = `${Math.min(event.currentTarget.scrollHeight, 180)}px`;
          }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          disabled={disabled || running}
        />
        {running ? (
          <button className="cancel-button" type="button" onClick={onCancel}>Cancel</button>
        ) : (
          <button className="send-button" type="button" onClick={submit} disabled={disabled || !question.trim()}>
            <span>Ask</span><span aria-hidden="true">↑</span>
          </button>
        )}
      </div>
      <div className="composer-hint">
        <span>Enter to send · Shift+Enter for a new line</span>
        <span>Each question is answered independently</span>
      </div>
    </div>
  );
}
