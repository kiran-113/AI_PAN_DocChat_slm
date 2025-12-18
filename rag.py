import logging
import chromadb
import tiktoken
import time
import os
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from llm_inference import LLMInference
nltk.download('punkt_tab')



class RAGSystem:
    def __init__(
        self,
        collection_name: str,
        db_path: str = "PDF_ChromaDB",
        n_results: int = 5
    ):
        self.collection_name = collection_name
        self.db_path = db_path
        self.n_results = n_results

        if not self.collection_name:
            raise ValueError("'collection_name' parameter is required.")

        self.llm_inference = LLMInference()
        self.logger = self._setup_logging()

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------
    def _format_time(self, response_time: float) -> str:
        minutes = response_time // 60
        seconds = response_time % 60
        return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"

    def _get_tokens_count(self, text: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))

    # --------------------------------------------------
    # Embeddings & Retrieval
    # --------------------------------------------------
    def _generate_embeddings(self, text: str):
        return self.llm_inference._generate_embeddings(
            input_text=text,
            model_name="nomic-embed-text:latest"
        )

    def _retrieve(self, user_text: str, n_results: int = 10):
        embedding = self._generate_embeddings(user_text)

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "embeddings"]
        )

        if not results["documents"]:
            return [], []

        chunks = results["documents"][0]
        embeddings = results["embeddings"][0]

        return chunks, embeddings

    # --------------------------------------------------
    # Re-ranking (BM25 + Semantic)
    # --------------------------------------------------
    def _rerank_docs(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        query: str,
        top_k: int = 5
    ):
        # BM25
        tokenized_chunks = [
            word_tokenize(chunk.lower()) for chunk in chunks
        ]
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(
            word_tokenize(query.lower())
        )

        # Semantic similarity
        query_embedding = np.array(self._generate_embeddings(query))
        chunk_embeddings = np.array(embeddings)

        dot_product = np.dot(chunk_embeddings, query_embedding)
        query_norm = np.linalg.norm(query_embedding)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

        semantic_scores = dot_product / (
            chunk_norms * query_norm + 1e-10
        )

        # Normalization
        bm25_norm = (bm25_scores - np.min(bm25_scores)) / (
            np.max(bm25_scores) - np.min(bm25_scores) + 1e-5
        )
        sem_norm = (semantic_scores - np.min(semantic_scores)) / (
            np.max(semantic_scores) - np.min(semantic_scores) + 1e-5
        )

        combined_scores = 0.5 * bm25_norm + 0.5 * sem_norm

        ranked_indices = np.argsort(combined_scores)[::-1]
        return [chunks[i] for i in ranked_indices[:top_k]]

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------
    def _get_prompt(self, query: str, context: str) -> str:
        return f"""
You are an AI assistant specialized in answering questions based **only** on the provided context.
The context is part of a PDF document.
Sections are separated by `########`.

### Context:
'''
{context}
'''

### Question:
"{query}"

### Instructions:
- Answer using only the provided context
- Do not summarize
- Be concise and direct

### Answer:
"""

    # --------------------------------------------------
    # Ollama Response
    # --------------------------------------------------
    def generate_response(self, query: str, ollama_model: str):
        if not ollama_model:
            return "Error: Choose an Ollama LLM"

        self.logger.info(
            f"--> Generate Response Using Ollama LLM : {ollama_model}"
        )

        chunks, embeddings = self._retrieve(query, n_results=20)

        if not chunks:
            return "No relevant information found. Database may be empty."

        reranked_docs = self._rerank_docs(
            chunks, embeddings, query, self.n_results
        )

        context = "\n\n########\n\n".join(reranked_docs)
        prompt = self._get_prompt(query, context)

        input_tokens = self._get_tokens_count(prompt)
        start_time = time.time()

        response = self.llm_inference.generate_text(
            prompt=prompt,
            model_name=ollama_model,
            llm_provider="Ollama"
        )

        output_tokens = self._get_tokens_count(response)
        response_time = time.time() - start_time

        self.logger.info(f"-> LLM Response : {response}")
        self.logger.info(
            f"-> Input tokens : {input_tokens} | "
            f"Output tokens : {output_tokens} | "
            f"Time : {self._format_time(response_time)}"
        )

        return (
            response,
            self._format_time(response_time),
            self.n_results,
            input_tokens,
            output_tokens,
        )

    # --------------------------------------------------
    # Delete Collection
    # --------------------------------------------------
    def delete_collection(self):
        self.client.delete_collection(self.collection_name)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    rag_system = RAGSystem(
        collection_name="pdf_content",
        db_path="PDF_ChromaDB",
        n_results=5
    )

    print(
        rag_system.generate_response(
            "What is the name of the book?",
            ollama_model="deepseek-r1:1.5b"
        )
    )
