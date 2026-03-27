"use client";

import { useMemo, useState } from "react";

type AnalyzeResponse = {
  stats: {
    files: number;
    chunks: number;
  };
  explanation: string;
  summary: string;
  answer: string;
};

type ParsedSection = {
  title: string;
  bullets: string[];
  paragraphs: string[];
};

function parseStructuredText(text: string): ParsedSection[] {
  const lines = (text || "").split(/\r?\n/).map((line) => line.trim());
  const sections: ParsedSection[] = [{ title: "Overview", bullets: [], paragraphs: [] }];

  let current = sections[0];

  for (const rawLine of lines) {
    if (!rawLine) {
      continue;
    }

    const headingHashMatch = rawLine.match(/^#{1,6}\s+(.+)$/);
    const headingBoldMatch = rawLine.match(/^\*\*(.+?)\*\*:?$/);
    const headingColonMatch = rawLine.match(/^([A-Z][A-Za-z0-9\s\-\/()]{2,80}):$/);
    const bulletMatch = rawLine.match(/^[-*•]\s+(.+)$/) || rawLine.match(/^\d+\.\s+(.+)$/);

    if (headingHashMatch || headingBoldMatch || headingColonMatch) {
      const title = (headingHashMatch?.[1] || headingBoldMatch?.[1] || headingColonMatch?.[1] || "Section").trim();
      current = { title, bullets: [], paragraphs: [] };
      sections.push(current);
      continue;
    }

    if (bulletMatch) {
      current.bullets.push(bulletMatch[1].trim());
      continue;
    }

    current.paragraphs.push(rawLine);
  }

  return sections.filter((section) => section.bullets.length > 0 || section.paragraphs.length > 0);
}

function OutputPanel({ title, text }: { title: string; text: string }) {
  const sections = useMemo(() => parseStructuredText(text), [text]);

  return (
    <section className="card result-card">
      <h2>{title}</h2>
      {sections.length === 0 && <p>No content generated.</p>}
      {sections.map((section, idx) => (
        <article key={`${section.title}-${idx}`} className="result-section">
          <h3>{section.title}</h3>
          {section.paragraphs.map((paragraph, pIdx) => (
            <p key={`p-${pIdx}`}>{paragraph}</p>
          ))}
          {section.bullets.length > 0 && (
            <ul>
              {section.bullets.map((bullet, bIdx) => (
                <li key={`b-${bIdx}`}>{bullet}</li>
              ))}
            </ul>
          )}
        </article>
      ))}
    </section>
  );
}

export default function HomePage() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState("");

  const fileCount = useMemo(() => files?.length ?? 0, [files]);

  async function handleAnalyze() {
    setError("");
    setResult(null);

    if (!files || files.length === 0) {
      setError("Upload at least one legal document.");
      return;
    }

    const formData = new FormData();
    for (const file of Array.from(files)) {
      formData.append("files", file);
    }
    formData.append("query", query);

    setLoading(true);
    const phases = [
      "Extracting text from documents...",
      "Building RAG chunks and embeddings...",
      "Generating explanation and summary...",
      "Answering your query..."
    ];
    let idx = 0;
    setStatus(phases[idx]);
    const timer = setInterval(() => {
      idx = Math.min(idx + 1, phases.length - 1);
      setStatus(phases[idx]);
    }, 1200);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || "Analysis failed.");
      }
      setResult(payload as AnalyzeResponse);
      setStatus("Completed.");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      setStatus("");
    } finally {
      clearInterval(timer);
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>Legal Document Analyzer</h1>
      <p>Gemini-only RAG analysis for explanation, summary, and question answering.</p>

      <section className="card">
        <label htmlFor="files">Upload legal files</label>
        <input id="files" type="file" multiple onChange={(e) => setFiles(e.target.files)} />
        <p>{fileCount} file(s) selected</p>

        <label htmlFor="query">Specific query (optional)</label>
        <textarea
          id="query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Example: What are the termination conditions and notice period?"
        />

        <button onClick={handleAnalyze} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Documents"}
        </button>

        {status && <p className="status">{status}</p>}
        {error && <p style={{ color: "#9d1d1d", fontWeight: 700 }}>{error}</p>}
      </section>

      {result && (
        <div className="grid">
          <OutputPanel title="Explanation" text={result.explanation} />
          <OutputPanel title="Summary" text={result.summary} />
          <OutputPanel title="Answer" text={result.answer || "No query provided."} />
          <section className="card result-card">
            <h2>Stats</h2>
            <div className="stats-grid">
              <div className="stat-chip">
                <span>Files</span>
                <strong>{result.stats.files}</strong>
              </div>
              <div className="stat-chip">
                <span>Chunks</span>
                <strong>{result.stats.chunks}</strong>
              </div>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
