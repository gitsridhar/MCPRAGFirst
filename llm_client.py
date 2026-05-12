from openai import OpenAI


class CustomLLMClient:
    def __init__(
        self,
        api_key: str = "ollama",
        base_url: str = "http://localhost:11434/v1",
        chat_model: str = "llama3.1:8b",
        embedding_model: str = "nomic-embed-text",
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def generate_response(self, prompt: str, context: str) -> str:
        system_prompt = (
            "Answer using only the supplied context when possible. "
            "If context is missing details, state uncertainty briefly.\n\n"
            f"Context:\n{context}"
        )
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding
