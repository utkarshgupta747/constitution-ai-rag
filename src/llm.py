import os

import requests
from dotenv import load_dotenv


load_dotenv()


class OpenRouterProvider:

    def __init__(
        self,
        model: str = "openrouter/free",
    ):
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not configured."
            )

        self.model = model

        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:

        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
        }

        response = requests.post(
            self.url,
            headers=headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]


if __name__ == "__main__":

    provider = OpenRouterProvider()

    system_prompt = """
You are a factual assistant answering questions about
the Constitution of India.

Answer clearly and concisely.
Do not invent constitutional provisions.
"""

    prompt = """
What is Article 21 of the Constitution of India?
"""

    answer = provider.generate(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    print("\nLLM Response:\n")
    print(answer)