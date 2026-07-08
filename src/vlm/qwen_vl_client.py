# DMS reproduction project — local Qwen2.5-VL-7B-Instruct client.
#
# Implements AndroidWorld's `infer.LlmWrapper` / `infer.MultimodalLlmWrapper`
# interfaces (see android_world/agents/infer.py) against a local vLLM
# OpenAI-compatible server, so it is a drop-in replacement for
# `infer.Gpt4Wrapper` / `infer.GeminiGcpWrapper` inside M3A / T3A / our own
# Planner-Actor agents.

from __future__ import annotations

import base64
import dataclasses
import time
from typing import Any, Optional

import numpy as np
import requests
from android_world.agents import infer


@dataclasses.dataclass
class QwenVLConfig:
  """Connection + generation settings for the local vLLM Qwen2.5-VL server."""

  base_url: str = "http://localhost:8000/v1"
  model: str = "Qwen2.5-VL-7B-Instruct"
  api_key: str = "EMPTY"  # vLLM does not check this unless --api-key is set.
  temperature: float = 0.0
  top_p: float = 0.9
  max_tokens: int = 1000
  max_retry: int = 3
  retry_waiting_seconds: float = 5.0
  request_timeout: float = 120.0


class QwenVLWrapper(infer.LlmWrapper, infer.MultimodalLlmWrapper):
  """Wraps a local vLLM-served Qwen2.5-VL-7B-Instruct as an AndroidWorld LLM.

  Mirrors the retry / (text, is_safe, raw_response) contract of
  `infer.Gpt4Wrapper` so it can be swapped in directly for M3A/T3A agents.
  """

  def __init__(self, config: QwenVLConfig | None = None, **overrides: Any):
    self.config = config or QwenVLConfig()
    for key, value in overrides.items():
      setattr(self.config, key, value)
    self._endpoint = f"{self.config.base_url.rstrip('/')}/chat/completions"

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(infer.array_to_jpeg_bytes(image)).decode("utf-8")

  def predict(self, text_prompt: str) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    for image in images:
      content.append({
          "type": "image_url",
          "image_url": {
              "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
          },
      })

    payload = {
        "model": self.config.model,
        "temperature": self.config.temperature,
        "top_p": self.config.top_p,
        "max_tokens": self.config.max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.config.api_key}",
    }

    counter = self.config.max_retry
    wait_seconds = self.config.retry_waiting_seconds
    last_error = None
    while counter > 0:
      try:
        response = requests.post(
            self._endpoint,
            headers=headers,
            json=payload,
            timeout=self.config.request_timeout,
        )
        if response.ok:
          body = response.json()
          if "choices" in body and body["choices"]:
            text = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            return text, True, {"response": body, "usage": usage}
        last_error = f"HTTP {response.status_code}: {response.text[:500]}"
      except Exception as e:  # pylint: disable=broad-exception-caught
        last_error = str(e)

      counter -= 1
      if counter > 0:
        print(f"[QwenVLWrapper] call failed ({last_error}); retrying in "
              f"{wait_seconds}s...")
        time.sleep(wait_seconds)
        wait_seconds *= 2

    print(f"[QwenVLWrapper] giving up after retries: {last_error}")
    return infer.ERROR_CALLING_LLM, None, None
