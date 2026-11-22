import json

import pytest

from eval.llm import LLMError, OpenAICompatClient, RateLimiter, TransportError


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def ok_body(text="hi", pt=11, ct=7):
    return json.dumps({"model": "m-1", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": pt, "completion_tokens": ct}}).encode()


def make(responses, **kw):
    clock = Clock()
    seen = []

    def transport(url, headers, body, timeout):
        seen.append((url, headers, json.loads(body)))
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    import random
    client = OpenAICompatClient("https://x.test/v1/", "m-1", "SECRET-KEY", transport=transport, clock=clock.now,
                                sleep=clock.sleep, rng=random.Random(0), rpm=60, **kw)
    return client, clock, seen


MSG = [{"role": "user", "content": "q"}]


def test_success_parses_text_usage_and_builds_request():
    client, _, seen = make([(200, {}, ok_body("answer", 5, 9))])
    r = client.complete(MSG, seed=3)
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.attempts, r.model_returned) == ("answer", 5, 9, 1, "m-1")
    url, headers, payload = seen[0]
    assert url == "https://x.test/v1/chat/completions"
    assert headers["Authorization"] == "Bearer SECRET-KEY"
    assert payload["temperature"] == 0.0 and payload["model"] == "m-1"
    assert "seed" not in payload  # default: Google compat endpoint 400s on unknown field "seed"
    assert set(payload) == {"model", "messages", "temperature", "max_tokens"}


def test_seed_is_omitted_by_default_even_when_the_caller_passes_one():
    client, _, seen = make([(200, {}, ok_body())])
    assert client.send_api_seed is False
    client.complete(MSG, seed=3)
    assert "seed" not in seen[0][2]


def test_seed_is_sent_only_with_explicit_opt_in():
    client, _, seen = make([(200, {}, ok_body()), (200, {}, ok_body())], send_api_seed=True)
    client.complete(MSG, seed=3)
    client.complete(MSG)  # no seed given -> still nothing to send
    assert seen[0][2]["seed"] == 3 and "seed" not in seen[1][2]


def test_default_max_tokens_leaves_room_for_thinking_tokens():
    from eval.llm import MIN_RECOMMENDED_MAX_TOKENS
    client, _, seen = make([(200, {}, ok_body())])
    client.complete(MSG)
    assert client.max_tokens >= 4096 >= MIN_RECOMMENDED_MAX_TOKENS
    assert seen[0][2]["max_tokens"] == client.max_tokens


def test_429_backoff_grows_and_eventually_succeeds():
    client, clock, _ = make([(429, {}, b"slow down"), (429, {}, b"slow down"), (200, {}, ok_body())])
    r = client.complete(MSG)
    assert r.attempts == 3 and client.total_requests == 3
    backoffs = [s for s in clock.sleeps if s > 1.5]  # pacing sleeps are <= 1s at rpm=60
    assert len(backoffs) == 2 and backoffs[1] > backoffs[0] * 0.9
    assert 1.0 <= backoffs[0] <= 2.0 and 2.0 <= backoffs[1] <= 4.0  # equal jitter within [cap/2, cap]


def test_retry_after_header_is_honoured_up_to_the_cap():
    client, clock, _ = make([(429, {"Retry-After": "30"}, b""), (200, {}, ok_body())])
    client.complete(MSG)
    assert max(clock.sleeps) >= 30


def test_non_retryable_status_fails_fast_without_leaking_the_key():
    client, _, _ = make([(400, {}, b"bad request")])
    with pytest.raises(LLMError) as exc:
        client.complete(MSG)
    assert client.total_requests == 1 and "SECRET-KEY" not in str(exc.value)


def test_retries_exhausted_raises():
    client, _, _ = make([(503, {}, b"")] * 3, max_retries=2)
    with pytest.raises(LLMError, match="gave up after 3 attempts"):
        client.complete(MSG)
    assert client.total_requests == 3


def test_transport_errors_and_garbage_200_are_retried():
    client, _, _ = make([TransportError("reset"), (200, {}, b"not json"), (200, {}, ok_body("fine"))])
    assert client.complete(MSG).text == "fine" and client.total_requests == 3


def test_null_content_becomes_empty_text():
    body = json.dumps({"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}).encode()
    client, _, _ = make([(200, {}, body)])
    r = client.complete(MSG)
    assert r.text == "" and r.finish_reason == "content_filter" and r.prompt_tokens is None


def test_rate_limiter_spaces_request_starts():
    clock = Clock()
    limiter = RateLimiter(13, clock=clock.now, sleep=clock.sleep)
    for _ in range(4):
        limiter.wait()
    assert clock.t == pytest.approx(3 * 60 / 13)
    assert clock.sleeps == pytest.approx([60 / 13] * 3)
