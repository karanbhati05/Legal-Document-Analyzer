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
          <section className="card">
            <h2>Explanation</h2>
            <pre>{result.explanation}</pre>
          </section>
          <section className="card">
            <h2>Summary</h2>
            <pre>{result.summary}</pre>
          </section>
          <section className="card">
            <h2>Answer</h2>
            <pre>{result.answer || "No query provided."}</pre>
          </section>
          <section className="card">
            <h2>Stats</h2>
            <pre>{JSON.stringify(result.stats, null, 2)}</pre>
          </section>
        </div>
      )}
    </main>
  );
}
