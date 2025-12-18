import ollama


class LLMInference:

    # --------------------------------------------------
    # Utils
    # --------------------------------------------------
    def _format_time(self, response_time):
        minutes = response_time // 60
        seconds = response_time % 60
        return f"{int(minutes)}m {int(seconds)}s" if minutes else f"Time: {int(seconds)}s"

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------
    def _generate_embeddings(self, input_text: str, model_name: str):
        return ollama.embeddings(
            model=model_name,
            prompt=input_text
        ).get("embedding", [])

    # --------------------------------------------------
    # Ollama Text Generation
    # --------------------------------------------------
    def generate_text_ollama(self, prompt: str, model_name: str):

        if not prompt or not model_name:
            return {"error": "Both 'prompt' and 'model_name' are required"}

        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.get("message", {}).get("content", "")
        except Exception as e:
            return {
                "error": "Failed to generate response using Ollama",
                "details": str(e)
            }

    # --------------------------------------------------
    # Unified Interface
    # --------------------------------------------------
    def generate_text(self, prompt: str, model_name: str, llm_provider: str):

        if llm_provider == "Ollama":
            return self.generate_text_ollama(prompt, model_name)

        raise ValueError(
            "Unsupported LLM provider. Only 'Ollama' is available."
        )


# --------------------------------------------------
# Main (Test)
# --------------------------------------------------
if __name__ == "__main__":
    llm = LLMInference()

    prompt = "What is the capital of France?"
    response = llm.generate_text(
        prompt=prompt,
        model_name="deepseek-r1:1.5b",
        llm_provider="Ollama"
    )

    print(response)
