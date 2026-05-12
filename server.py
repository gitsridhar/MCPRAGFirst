import os
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from llm_client import CustomLLMClient
from rag_engine import RAGEngine


mcp = FastMCP("PDF-RAG-MCP-Server")

llm_client = CustomLLMClient(
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b"),
    embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
)

rag = RAGEngine(
    llm_client=llm_client,
    pinecone_api_key=os.environ["PINECONE_API_KEY"],
    index_name=os.getenv("PINECONE_INDEX_NAME", "mcp-rag-index"),
    namespace=os.getenv("PINECONE_NAMESPACE", "pdf-docs"),
    region=os.getenv("PINECONE_REGION", "us-east-1"),
    cloud=os.getenv("PINECONE_CLOUD", "aws"),
)

ingestion_report = {}

# Mode 1 (default): create and ingest a sample PDF during startup.
startup_mode = os.getenv("RAG_STARTUP_MODE", "sample").strip().lower()
if startup_mode == "sample":
    pdf_path = os.getenv("RAG_PDF_PATH", "data/mcp_rag_reference.pdf")
    created_pdf = rag.create_sample_pdf(pdf_path)
    ingestion_report = rag.ingest_pdf(created_pdf)
else:
    ingestion_report = {
        "indexed_chunks": 0,
        "message": "Startup ingestion disabled. Use ingest_pdf_document tool.",
    }


@mcp.tool()
def retrieve_context_from_pdf(query: str, top_k: int = 4) -> dict:
    """Custom tool: retrieve relevant context chunks from Pinecone for a user query."""
    retrieval = rag.retrieve(query=query, top_k=top_k)
    return {
        "query": query,
        "context": retrieval.context,
        "match_count": len(retrieval.matches),
        "index": rag.index_name,
        "namespace": rag.namespace,
    }


@mcp.tool()
def ingest_pdf_document(pdf_path: str) -> dict:
    """Mode 2: ingest a user-supplied PDF path on demand."""
    path = Path(pdf_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        return {"status": "error", "message": f"PDF file not found: {path}"}
    if path.suffix.lower() != ".pdf":
        return {"status": "error", "message": f"Not a PDF file: {path}"}

    report = rag.ingest_pdf(str(path))
    return {"status": "ok", "mode": "on-demand", "ingestion": report}


@mcp.tool()
def server_health() -> dict:
    """Predefined utility tool: basic server status and ingestion metadata."""
    return {
        "status": "ok",
        "server": "PDF-RAG-MCP-Server",
        "time_utc": datetime.utcnow().isoformat(),
        "ingestion": ingestion_report,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
