import requests

from contextwalker.config import OLLAMA_GENERATE_URL, OLLAMA_MODEL


def ollama_generate(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data["response"].strip()

