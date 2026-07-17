"""Minimal Gemini REST client (no SDK dependency).

Used for: identifying unknown games from screenshots, inferring semantic
actions for discovered keybinds, and generating gesture mapping suggestions.
Every caller must tolerate None (offline heuristics take over)."""
from __future__ import annotations

import base64
import json
import re
import threading

import requests

API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 fallback_models: list[str] | None = None, timeout: float = 25.0):
        self.api_key = api_key
        self.model = model
        self.fallback_models = fallback_models or []
        self.timeout = timeout
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._working_model: str | None = None

    def _call(self, model: str, parts: list, json_mode: bool) -> str | None:
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 2048},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        try:
            resp = requests.post(
                f"{API_BASE}/{model}:generateContent",
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=body, timeout=self.timeout)
        except requests.RequestException as e:
            self.last_error = f"network: {e}"
            return None
        if resp.status_code == 429:
            self.last_error = (f"{model}: rate limit / quota exceeded — "
                               "offline heuristics remain active; retries automatically later")
            return None
        if resp.status_code != 200:
            self.last_error = f"{model}: HTTP {resp.status_code}: {resp.text[:300]}"
            return None
        try:
            data = resp.json()
            texts = [p.get("text", "") for c in data.get("candidates", [])
                     for p in c.get("content", {}).get("parts", [])]
            out = "".join(texts).strip()
            if out:
                self.last_error = None
                return out
            self.last_error = f"{model}: empty response"
        except (ValueError, KeyError) as e:
            self.last_error = f"{model}: bad response: {e}"
        return None

    def generate(self, prompt: str, image_jpeg: bytes | None = None,
                 json_mode: bool = True) -> str | None:
        parts: list = [{"text": prompt}]
        if image_jpeg:
            parts.append({"inline_data": {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_jpeg).decode("ascii")}})
        with self._lock:
            models = ([self._working_model] if self._working_model else []) + \
                     [self.model] + self.fallback_models
        seen = set()
        for m in models:
            if m in seen:
                continue
            seen.add(m)
            out = self._call(m, parts, json_mode)
            if out is not None:
                with self._lock:
                    self._working_model = m
                return out
        return None

    def generate_json(self, prompt: str, image_jpeg: bytes | None = None) -> dict | None:
        text = self.generate(prompt, image_jpeg, json_mode=True)
        if text is None:
            return None
        return extract_json(text)

    def ping(self) -> bool:
        """Cheap availability check; caches the first working model."""
        out = self.generate('Reply with exactly this JSON: {"ok": true}', json_mode=True)
        parsed = extract_json(out) if out else None
        return bool(parsed and parsed.get("ok") is True)


def extract_json(text: str) -> dict | None:
    """Parse JSON, tolerating markdown fences and leading prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    for candidate in (text, text[text.find("{"): text.rfind("}") + 1]):
        if not candidate:
            continue
        try:
            out = json.loads(candidate)
            return out if isinstance(out, dict) else None
        except json.JSONDecodeError:
            continue
    return None
