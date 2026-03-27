import { GoogleGenerativeAI } from "@google/generative-ai";
import { NextResponse } from "next/server";
import mammoth from "mammoth";
import pdfParse from "pdf-parse";

export const runtime = "nodejs";
export const maxDuration = 60;

type Chunk = {
  content: string;
  source: string;
};

function envNumber(name: string, fallback: number): number {
  const parsed = Number(process.env[name]);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function normalizeEmbeddingModel(model: string | undefined): string {
  const raw = (model || "gemini-embedding-001").trim().replace(/^['"]|['"]$/g, "");
  const compact = raw.replace(/^models\/models\//, "models/");
  if (compact.startsWith("models/")) {
    return compact;
  }
  const finalId = compact.includes("/") ? compact.split("/").pop() || "gemini-embedding-001" : compact;
  return `models/${finalId}`;
}

function chunkText(input: string, chunkSize: number, overlap: number): string[] {
  const text = input.replace(/\r/g, "").trim();
  if (!text) return [];

  const chunks: string[] = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const piece = text.slice(start, end).trim();
    if (piece) chunks.push(piece);
    if (end >= text.length) break;
    start = Math.max(0, end - overlap);
  }
  return chunks;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i += 1) {
    dot += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

async function extractText(file: File): Promise<string> {
  const bytes = Buffer.from(await file.arrayBuffer());
  const lower = file.name.toLowerCase();

  if (lower.endsWith(".pdf")) {
    const parsed = await pdfParse(bytes);
    return parsed.text || "";
  }

  if (lower.endsWith(".docx")) {
    const parsed = await mammoth.extractRawText({ buffer: bytes });
    return parsed.value || "";
  }

  return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
}

function buildContext(chunks: Chunk[], topK: number): string {
  return chunks
    .slice(0, topK)
    .map((c, idx) => `[Chunk ${idx + 1} | Source: ${c.source}]\n${c.content}`)
    .join("\n\n");
}

async function generateText(genAI: GoogleGenerativeAI, modelName: string, prompt: string): Promise<string> {
  const model = genAI.getGenerativeModel({ model: modelName });
  const result = await model.generateContent(prompt);
  return result.response.text();
}

export async function POST(request: Request) {
  try {
    const apiKey = process.env.GOOGLE_API_KEY;
    if (!apiKey) {
      return NextResponse.json({ error: "Missing GOOGLE_API_KEY" }, { status: 500 });
    }

    const modelName = process.env.GEMINI_MODEL || "gemini-2.5-flash";
    const embeddingModel = normalizeEmbeddingModel(process.env.GEMINI_EMBEDDING_MODEL || "gemini-embedding-001");
    const topK = envNumber("TOP_K", 3);
    const chunkSize = envNumber("CHUNK_SIZE", 900);
    const chunkOverlap = envNumber("CHUNK_OVERLAP", 120);

    const formData = await request.formData();
    const rawFiles = formData.getAll("files");
    const files = rawFiles.filter((item): item is File => item instanceof File);
    const query = String(formData.get("query") || "").trim();

    if (files.length === 0) {
      return NextResponse.json({ error: "No files uploaded." }, { status: 400 });
    }

    const genAI = new GoogleGenerativeAI(apiKey);
    const embedding = genAI.getGenerativeModel({ model: embeddingModel });

    const allChunks: Chunk[] = [];
    for (const file of files) {
      const text = await extractText(file);
      const chunks = chunkText(text, chunkSize, chunkOverlap);
      for (const chunk of chunks) {
        allChunks.push({ content: chunk, source: file.name });
      }
    }

    if (allChunks.length === 0) {
      return NextResponse.json(
        { error: "No readable text found in uploaded files. Try text-based PDF/DOCX/TXT files." },
        { status: 400 }
      );
    }

    const chunkVectors: number[][] = [];
    for (const chunk of allChunks) {
      const embedded = await embedding.embedContent(chunk.content);
      chunkVectors.push(embedded.embedding.values || []);
    }

    async function retrieve(queryText: string): Promise<Chunk[]> {
      const q = await embedding.embedContent(queryText);
      const qVec = q.embedding.values || [];
      const scored = allChunks.map((chunk, i) => ({
        chunk,
        score: cosineSimilarity(qVec, chunkVectors[i])
      }));
      scored.sort((a, b) => b.score - a.score);
      return scored.slice(0, topK).map((s) => s.chunk);
    }

    const explanationContext = buildContext(await retrieve("Explain this legal document in depth"), topK);
    const summaryContext = buildContext(await retrieve("Summarize key legal points"), topK);
    const answerContext = buildContext(await retrieve(query || "What are key obligations and risks?"), topK);

    const explanationPrompt = `You are an expert legal document analyst. Provide a clear explanation in plain language.\n\nSections:\n- What this document is about\n- Parties and roles\n- Key rights and obligations\n- Important dates and terms\n- Risks and liabilities\n\nContext:\n${explanationContext}`;

    const summaryPrompt = `Summarize this legal document in concise bullet points (max 10 bullets). Include obligations, timelines, payment terms, termination, and dispute terms where present.\n\nContext:\n${summaryContext}`;

    const answerPrompt = `Answer the user question using only the provided context. If not found, say what is missing.\n\nQuestion:\n${query || "No query provided"}\n\nContext:\n${answerContext}`;

    const [explanation, summary, answer] = await Promise.all([
      generateText(genAI, modelName, explanationPrompt),
      generateText(genAI, modelName, summaryPrompt),
      query ? generateText(genAI, modelName, answerPrompt) : Promise.resolve("")
    ]);

    return NextResponse.json({
      stats: {
        files: files.length,
        chunks: allChunks.length
      },
      explanation,
      summary,
      answer
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected server error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
