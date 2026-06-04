"""OpenRouter LLM client — OpenAI-compatible API."""

from __future__ import annotations

import time
from dataclasses import dataclass

from openai import OpenAI, RateLimitError, APIStatusError

from src.config import Settings, settings


@dataclass
class LLMResponse:
    text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMClient:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=cfg.openrouter_api_key,
        )
        self.model = cfg.generation_model
        self.fallback_model = cfg.fallback_model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        start = time.time()
        model_used = self.model

        try:
            response = self._call(model_used, system_prompt, user_prompt, temperature, max_tokens)
        except (RateLimitError, APIStatusError):
            model_used = self.fallback_model
            response = self._call(model_used, system_prompt, user_prompt, temperature, max_tokens)

        elapsed_ms = (time.time() - start) * 1000
        usage = response.usage

        return LLMResponse(
            text=response.choices[0].message.content,
            model_used=model_used,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed_ms,
        )

    def _call(self, model, system_prompt, user_prompt, temperature, max_tokens):
        return self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )


# --- Runnable standalone ---

if __name__ == "__main__":
    llm = LLMClient()
    resp = llm.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="What is the FCA Handbook? Answer in 2 sentences.",
    )
    print(f"Model: {resp.model_used}")
    print(f"Tokens: {resp.prompt_tokens} in / {resp.completion_tokens} out")
    print(f"Latency: {resp.latency_ms:.0f}ms")
    print(f"\n{resp.text}")
