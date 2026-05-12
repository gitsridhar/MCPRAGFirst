from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fpdf import FPDF
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader


@dataclass
class RetrievalResult:
    context: str
    matches: list[dict[str, Any]]


class RAGEngine:
    def __init__(
        self,
        llm_client,
        pinecone_api_key: str,
        index_name: str = "mcp-rag-index",
        namespace: str = "pdf-docs",
        region: str = "us-east-1",
        cloud: str = "aws",
    ):
        self.llm_client = llm_client
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.region = region
        self.cloud = cloud
        self.index = None
        self._is_index_ready = False

    def create_sample_pdf(self, pdf_path: str) -> str:
        path = Path(pdf_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=12)
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        lines = [
            "Model Context Protocol (MCP) enables clients to discover and call tools from servers.",
            "Retrieval-Augmented Generation (RAG) combines vector search and large language models.",
            "A robust RAG pipeline includes document ingestion, chunking, embeddings, and retrieval.",
            "Pinecone is a managed vector database that supports fast semantic search.",
            "Ollama can expose local LLM models through an OpenAI-compatible API endpoint.",
            "The MCP client can request context from a server tool and compose the final answer with an LLM.",
        ]
        for line in lines:
            pdf.multi_cell(0, 8, line)
            pdf.ln(1)
        pdf.output(str(path))
        return str(path)

    def _read_pdf_text(self, pdf_path: str) -> str:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts).strip()

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        step = max(1, chunk_size - overlap)
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start += step
        return chunks

    def _ensure_index(self, dimension: int) -> None:
        if self._is_index_ready:
            return

        index_list = self.pc.list_indexes()
        if hasattr(index_list, "names"):
            existing = list(index_list.names())
        else:
            existing = [idx.get("name") for idx in index_list if isinstance(idx, dict)]

        if self.index_name not in existing:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
            )
        self.index = self.pc.Index(self.index_name)
        self._is_index_ready = True

    def ingest_pdf(self, pdf_path: str) -> dict[str, Any]:
        text = self._read_pdf_text(pdf_path)
        chunks = self._chunk_text(text)
        if not chunks:
            return {"indexed_chunks": 0, "message": "No text found in PDF."}

        embeddings = [self.llm_client.get_embedding(chunk) for chunk in chunks]
        self._ensure_index(dimension=len(embeddings[0]))

        vectors = []
        source_prefix = hashlib.sha1(pdf_path.encode("utf-8")).hexdigest()[:12]
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            vectors.append(
                {
                    "id": f"{source_prefix}-chunk-{i}",
                    "values": embedding,
                    "metadata": {"text": chunk, "source": pdf_path},
                }
            )

        self.index.upsert(vectors=vectors, namespace=self.namespace)
        return {
            "indexed_chunks": len(vectors),
            "index_name": self.index_name,
            "namespace": self.namespace,
            "source": pdf_path,
        }

    def retrieve(self, query: str, top_k: int = 4) -> RetrievalResult:
        if not self._is_index_ready:
            raise RuntimeError("Index is not initialized. Ingest a PDF first.")

        query_vector = self.llm_client.get_embedding(query)
        result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            namespace=self.namespace,
            include_metadata=True,
        )

        if isinstance(result, dict):
            matches = result.get("matches", [])
        else:
            matches = getattr(result, "matches", [])

        contexts = []
        for m in matches:
            if isinstance(m, dict):
                contexts.append(m.get("metadata", {}).get("text", ""))
                continue
            metadata = getattr(m, "metadata", None) or {}
            contexts.append(metadata.get("text", ""))

        context = "\n\n".join([c for c in contexts if c])
        return RetrievalResult(context=context, matches=matches)
