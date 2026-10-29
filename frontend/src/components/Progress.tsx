import type { PipelineProgress } from "../types";

const steps = [
  { node: "retrieve", label: "Retrieve" },
  { node: "grade_documents", label: "Grade" },
  { node: "generate", label: "Generate" },
  { node: "grade_generation", label: "Critic" },
];

type Props = {
  progress: PipelineProgress;
};

export function Progress({ progress }: Props) {
  return (
    <div className="progress-card" aria-live="polite" aria-label="Answer pipeline progress">
      <div className="progress-heading">
        <span className="spinner" aria-hidden="true" />
        <strong>
          {progress.attempt > 1
            ? `Regenerating (attempt ${progress.attempt})`
            : progress.webSearch
              ? "Searching the web"
              : "Following the evidence"}
        </strong>
      </div>
      <ol className="pipeline-steps">
        {steps.map((step) => {
          const complete = progress.completed.includes(step.node);
          const active = progress.activeNode === step.node;
          return (
            <li className={complete ? "step-complete" : active ? "step-active" : ""} key={step.node}>
              <span aria-hidden="true">{complete ? "✓" : active ? "•" : "○"}</span>
              {step.label}
            </li>
          );
        })}
      </ol>
      {progress.webSearch && <p className="web-search-hint">Searching the web for supporting context</p>}
    </div>
  );
}
