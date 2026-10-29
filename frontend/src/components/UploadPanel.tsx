import { useRef, useState } from "react";
import type { UploadFileStatus } from "../types";

const MAX_FILES = 3;
const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_PAGES = 20;

type Props = {
  files: Array<UploadFileStatus & { pending?: boolean }>;
  busy: boolean;
  notice: string | null;
  onUpload: (files: File[], invalid: UploadFileStatus[]) => void;
  onRemoveAll: () => void;
};

export function UploadPanel({ files, busy, notice, onUpload, onRemoveAll }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const acceptedCount = files.filter((file) => file.ok || file.pending).length;

  async function handleSelection(selected: FileList | File[]) {
    const queue = Array.from(selected);
    const valid: File[] = [];
    const invalid: UploadFileStatus[] = [];
    let remaining = Math.max(0, MAX_FILES - acceptedCount);

    for (const file of queue) {
      let error: string | undefined;
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        error = "File must have a .pdf extension.";
      } else if (file.type && file.type !== "application/pdf") {
        error = "Choose a PDF file.";
      } else if (file.size > MAX_FILE_BYTES) {
        error = "File exceeds the 10 MB limit.";
      } else if (remaining <= 0) {
        error = "You can upload at most 3 PDFs.";
      } else {
        const signature = new Uint8Array(await file.slice(0, 4).arrayBuffer());
        if (signature[0] !== 0x25 || signature[1] !== 0x50 || signature[2] !== 0x44 || signature[3] !== 0x46) {
          error = "File does not have a valid PDF signature.";
        }
      }

      if (error) {
        invalid.push({ name: file.name, pages: 0, ok: false, error });
      } else {
        valid.push(file);
        remaining -= 1;
      }
    }

    onUpload(valid, invalid);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <section className="upload-panel" aria-labelledby="upload-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Private session</p>
          <h2 id="upload-heading">Your PDFs</h2>
        </div>
        {files.length > 0 && (
          <button className="text-button" type="button" onClick={onRemoveAll} disabled={busy}>
            Remove all
          </button>
        )}
      </div>
      <p className="upload-description">
        Answers in this mode use only these files. Up to 3 PDFs, 20 pages and 10 MB each.
      </p>
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept=".pdf,application/pdf"
        multiple
        aria-label="Choose PDF files"
        disabled={busy}
        onChange={(event) => {
          if (event.currentTarget.files) void handleSelection(event.currentTarget.files);
        }}
      />
      <button
        type="button"
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          if (event.dataTransfer.files.length) void handleSelection(event.dataTransfer.files);
        }}
        disabled={busy}
        aria-label="Drop PDF files here or choose files"
      >
        <span className="drop-icon" aria-hidden="true">↑</span>
        <strong>{busy ? "Indexing your files…" : "Drop PDFs here"}</strong>
        <span>or choose from your device</span>
      </button>
      {notice && <p className="inline-notice" role="status">{notice}</p>}
      {files.length > 0 && (
        <ul className="file-list" aria-label="Uploaded files">
          {files.map((file, index) => (
            <li className={`file-item ${file.ok ? "file-ok" : file.pending ? "file-pending" : "file-error"}`} key={`${file.name}-${index}`}>
              <span className="file-icon" aria-hidden="true">PDF</span>
              <span className="file-details">
                <strong title={file.name}>{file.name}</strong>
                <small>
                  {file.pending
                    ? "Reading and indexing…"
                    : file.ok
                      ? `${file.pages} ${file.pages === 1 ? "page" : "pages"}`
                      : file.error}
                </small>
              </span>
              <span className="file-state" aria-label={file.ok ? "Ready" : file.pending ? "In progress" : "Upload failed"}>
                {file.ok ? "✓" : file.pending ? "…" : "!"}
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="upload-footnote">
        The API also enforces a {MAX_PAGES}-page per-file limit. Session files are cleared when the backend restarts or the session expires.
      </p>
    </section>
  );
}
