from typing import Optional, Union

import requests

from contextwalker.config import OLLAMA_GENERATE_URL, OLLAMA_MODEL


def ollama_generate(
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    think: Optional[Union[bool, str]] = "low",
    max_attempts: int = 3,
) -> str:
    """Generate a complete, non-empty response from Ollama.

    ``gpt-oss`` uses reasoning tokens before emitting its final response. A
    small ``num_predict`` limit can therefore produce an empty response or a
    response cut off with ``done_reason=length``. Use low reasoning for this
    retrieval-enrichment task and retry with a larger output allowance when
    the model exhausts the current limit.
    """

    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_problem = "Ollama did not return a response"

    for attempt in range(1, max_attempts + 1):
        token_limit = max_tokens * (2 ** (attempt - 1))
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": token_limit,
            },
        }
        if think is not None:
            payload["think"] = think

        try:
            response = requests.post(
                OLLAMA_GENERATE_URL,
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            generated_text = str(data.get("response") or "").strip()
            done_reason = str(data.get("done_reason") or "").lower()

            if generated_text and done_reason != "length":
                return generated_text

            if done_reason == "length":
                last_problem = (
                    "generation reached its token limit before completing "
                    "the final response"
                )
            else:
                last_problem = "generation returned an empty final response"
        except (requests.RequestException, TypeError, ValueError) as error:
            last_problem = str(error)

        if attempt < max_attempts:
            print(
                "[WARN] Ollama generation attempt "
                f"{attempt}/{max_attempts} failed: {last_problem}. "
                f"Retrying with {token_limit * 2} tokens."
            )

    raise RuntimeError(
        f"Ollama generation failed after {max_attempts} attempts: "
        f"{last_problem}"
    )
