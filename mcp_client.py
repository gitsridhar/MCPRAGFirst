import argparse
import asyncio
import json
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm_client import CustomLLMClient


def _extract_tool_payload(result: Any) -> dict[str, Any]:
    """Normalize different MCP tool result shapes into a dictionary payload."""
    if hasattr(result, "structuredContent") and result.structuredContent:
        payload = result.structuredContent
        if isinstance(payload, dict):
            return payload

    if hasattr(result, "content") and result.content:
        first = result.content[0]
        text = getattr(first, "text", "")
        if isinstance(text, str) and text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"text": text}

    if isinstance(result, dict):
        return result

    return {"raw": str(result)}


async def run_client(query: str, top_k: int, pdf_path: str | None) -> None:
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
        env=os.environ.copy(),
    )

    llm_client = CustomLLMClient(
        api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b"),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            health_result = await session.call_tool("server_health", {})
            health = _extract_tool_payload(health_result)

            ingest_response = None
            if pdf_path:
                ingest_result = await session.call_tool(
                    "ingest_pdf_document", {"pdf_path": pdf_path}
                )
                ingest_response = _extract_tool_payload(ingest_result)

            retrieve_result = await session.call_tool(
                "retrieve_context_from_pdf", {"query": query, "top_k": top_k}
            )
            retrieved = _extract_tool_payload(retrieve_result)
            context = retrieved.get("context", "")

            final_answer = llm_client.generate_response(prompt=query, context=context)

            print("=== MCP SERVER HEALTH ===")
            print(json.dumps(health, indent=2))
            if ingest_response is not None:
                print("\n=== ON-DEMAND INGESTION ===")
                print(json.dumps(ingest_response, indent=2))
            print("\n=== RETRIEVED CONTEXT ===")
            print(context or "No context retrieved.")
            print("\n=== FINAL ANSWER (OLLAMA) ===")
            print(final_answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Custom MCP client over STDIO with Ollama LLM")
    parser.add_argument("query", type=str, help="User query")
    parser.add_argument("--top-k", type=int, default=4, help="Top K chunks to retrieve")
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="If set, ingest this PDF on demand before retrieval (mode 2).",
    )
    args = parser.parse_args()

    asyncio.run(run_client(query=args.query, top_k=args.top_k, pdf_path=args.pdf_path))


if __name__ == "__main__":
    main()
