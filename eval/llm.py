"""OpenAI-compatible chat client (stdlib only) with RPM pacing, 429 backoff and token accounting.

Never logs or returns the API key. `transport`, `clock`, `sleep` and `rng` are injectable
so the retry/pacing logic is unit-tested without network or real sleeping.
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MIN_RECOMMENDED_MAX_TOKENS = 2048  # thinking tokens can count inside max_tokens
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}

Transport = Callable[[str, Dict[str, str], bytes, float], Tuple[int, Dict[str, str], bytes]]


class LLMError(RuntimeError):
    """Non-retryable failure, or retries exhausted. Nothing is checkpointed for the run."""


class TransportError(RuntimeError):
    """Network-level failure (DNS, reset, timeout); retried like a 5xx."""


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    latency_s: float
    attempts: int
    model_returned: Optional[str]
    finish_reason: Optional[str]


class RateLimiter:
    """Minimum spacing of 60/rpm seconds between request starts."""

    def __init__(self, rpm: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        self.interval = 60.0 / rpm
        self._clock, self._sleep = clock, sleep
        self._next_ok: Optional[float] = None

    def wait(self) -> None:
        now = self._clock()
        if self._next_ok is not None and now < self._next_ok:
            self._sleep(self._next_ok - now)
            now = self._next_ok
        self._next_ok = now + self.interval


def urllib_transport(url: str, headers: Dict[str, str], body: bytes, timeout: float):
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise TransportError(str(exc)) from exc


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        rpm: float = 13,
        send_api_seed: bool = False,
        max_retries: int = 8,
        base_backoff: float = 2.0,
        max_backoff: float = 120.0,
        timeout: float = 120.0,
        transport: Transport = urllib_transport,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self._key = api_key
        self.temperature, self.max_tokens, self.send_api_seed = temperature, max_tokens, send_api_seed
        self.max_retries, self.base_backoff, self.max_backoff = max_retries, base_backoff, max_backoff
        self.timeout = timeout
        self._transport, self._clock, self._sleep = transport, clock, sleep
        self._rng = rng or random.Random()
        self.limiter = RateLimiter(rpm, clock=clock, sleep=sleep)
        self.total_requests = 0  # every HTTP attempt counts against a daily quota

    def _backoff(self, attempt: int, headers: Dict[str, str]) -> float:
        cap = min(self.max_backoff, self.base_backoff * (2 ** attempt))
        delay = cap / 2 + self._rng.uniform(0, cap / 2)  # "equal jitter": never ~0
        retry_after = {k.lower(): v for k, v in headers.items()}.get("retry-after")
        if retry_after:
            try:
                delay = max(delay, min(float(retry_after), self.max_backoff))
            except ValueError:
                pass
        return delay

    def complete(self, messages: List[Dict[str, str]], seed: Optional[int] = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Off by default: Google's OpenAI-compat endpoint rejects the field with HTTP 400.
        if self.send_api_seed and seed is not None:
            payload["seed"] = seed
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}

        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            self.total_requests += 1
            started = self._clock()
            try:
                status, resp_headers, raw = self._transport(self.url, headers, body, self.timeout)
            except TransportError as exc:
                status, resp_headers, raw = -1, {}, str(exc).encode("utf-8", "replace")
            latency = self._clock() - started

            if status == 200:
                try:
                    return self._parse(raw, latency, attempt + 1)
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    last = f"unparseable 200 response: {exc}"
                    status = -2  # treat as retryable
            else:
                last = f"HTTP {status}: {raw[:300].decode('utf-8', 'replace')}"
            if status != -2 and status != -1 and status not in RETRYABLE_STATUS:
                raise LLMError(last)
            if attempt < self.max_retries:
                self._sleep(self._backoff(attempt, resp_headers))
        raise LLMError(f"gave up after {self.max_retries + 1} attempts; last error: {last}")

    @staticmethod
    def _parse(raw: bytes, latency: float, attempts: int) -> LLMResponse:
        data = json.loads(raw.decode("utf-8"))
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        content = (choice.get("message") or {}).get("content")
        return LLMResponse(
            text=content or "",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_s=latency,
            attempts=attempts,
            model_returned=data.get("model"),
            finish_reason=choice.get("finish_reason"),
        )
