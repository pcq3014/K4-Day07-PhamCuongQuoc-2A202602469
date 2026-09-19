from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    EMPTY_CONTEXT_ANSWER = (
        "Không tìm thấy tài liệu liên quan trong cơ sở tri thức, "
        "nên tôi không trả lời câu hỏi này."
    )

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            # Empty store or no candidates: say so instead of burning an LLM call.
            return self.EMPTY_CONTEXT_ANSWER

        prompt = self._build_prompt(question, results)
        return self.llm_fn(prompt)

    def _build_prompt(self, question: str, results: list[dict]) -> str:
        """Number each chunk so the answer can cite [1]/[2]/[3] back to a source."""
        blocks = []
        for position, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}
            source = (
                metadata.get("source_url")
                or metadata.get("source")
                or metadata.get("doc_id")
                or result.get("id")
                or "unknown"
            )
            blocks.append(f"[{position}] (nguồn: {source})\n{result['content']}")
        context = "\n\n".join(blocks)

        return (
            "Bạn trả lời câu hỏi CHỈ dựa trên các đoạn ngữ cảnh được đánh số bên dưới.\n"
            "Quy tắc:\n"
            "- Không dùng kiến thức ngoài ngữ cảnh, không suy đoán quy định.\n"
            "- Mỗi ý trong câu trả lời phải kèm số trích dẫn tương ứng, ví dụ [1].\n"
            "- Nếu ngữ cảnh không chứa đủ thông tin, trả lời đúng một câu: "
            "'Ngữ cảnh được cung cấp không trả lời được câu hỏi này.'\n\n"
            f"=== NGỮ CẢNH ===\n{context}\n\n"
            f"=== CÂU HỎI ===\n{question}\n\n"
            "=== TRẢ LỜI ===\n"
        )
