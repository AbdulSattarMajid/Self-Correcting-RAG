export type Mode = "library" | "uploads";

export type Source = {
  type: "library" | "web" | "upload";
  text: string;
};

export type FinalAnswer = {
  question: string;
  answer: string;
  grounded: "yes" | "no";
  critic_checks: number;
  sources: Source[];
};

export type PipelineNode =
  | { node: "retrieve"; documents: number }
  | { node: "grade_documents"; kept: number; web_search_needed: boolean }
  | { node: "generate"; attempt: number }
  | { node: "grade_generation"; grounded: "yes" | "no" }
  | { node: "web_search" };

export type PipelineProgress = {
  completed: string[];
  attempt: number;
  webSearch: boolean;
  activeNode: string | null;
};

export type UploadFileStatus = {
  name: string;
  pages: number;
  ok: boolean;
  error?: string;
};

export type SessionDetails = {
  session_id: string;
  files: Array<{ name: string; pages: number }>;
  total_files: number;
};

export type ConversationItem = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status?: "pending" | "complete" | "error" | "cancelled";
  grounded?: "yes" | "no";
  critic_checks?: number;
  sources?: Source[];
};
