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


# ---- Step 8c-3: transient 5xx / network faults must not abort a long run -------------------------

@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_every_transient_status_is_retried_then_succeeds(status):
    client, clock, _ = make([(status, {}, b"transient"), (200, {}, ok_body("done"))])
    r = client.complete(MSG)
    assert r.text == "done" and r.attempts == 2 and client.total_requests == 2
    assert any(s > 1.5 for s in clock.sleeps)  # a backoff sleep (not only pacing) happened


def test_503_retry_after_header_is_honoured():
    client, clock, _ = make([(503, {"Retry-After": "20"}, b""), (200, {}, ok_body())])
    client.complete(MSG)
    assert max(clock.sleeps) >= 20


def test_a_burst_of_503s_spread_over_a_pilot_does_not_abort_it():
    # 15 calls, each preceded by one 503: 30 HTTP attempts, all 15 calls succeed.
    responses = []
    for i in range(15):
        responses += [(503, {}, b"overloaded"), (200, {}, ok_body(f"r{i}"))]
    client, _, _ = make(responses)
    texts = [client.complete(MSG).text for _ in range(15)]
    assert texts == [f"r{i}" for i in range(15)] and client.total_requests == 30


def test_backoff_is_exponential_with_jitter_and_capped():
    client, clock, _ = make([(503, {}, b"")] * 7 + [(200, {}, ok_body())], max_retries=8, max_backoff=20.0)
    client.complete(MSG)
    backoffs = [s for s in clock.sleeps if s > 1.0]
    assert len(backoffs) == 7
    caps = [min(20.0, 2.0 * 2 ** n) for n in range(7)]
    assert all(c / 2 <= b <= c for b, c in zip(backoffs, caps))


def test_max_retries_is_configurable_and_exhaustion_is_a_clean_error():
    client, _, _ = make([(503, {}, b"")] * 2, max_retries=1)
    with pytest.raises(LLMError, match="gave up after 2 attempts"):
        client.complete(MSG)


# A real local HTTP server exercises the urllib transport itself (no mocks).
import http.server  # noqa: E402
import threading  # noqa: E402


class _Script(http.server.BaseHTTPRequestHandler):
    script = []
    hits = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        type(self).hits += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        kind = type(self).script.pop(0)
        if kind == "close":
            self.connection.close()
            return
        if kind == "truncate":
            self.send_response(200)
            self.send_header("Content-Length", "500")
            self.end_headers()
            self.wfile.write(b'{"choices":')
            self.wfile.flush()
            self.connection.close()
            return
        status = int(kind)
        body = ok_body("served") if status == 200 else b"unavailable"
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 503:
            self.send_header("Retry-After", "0")
        self.end_headers()
        self.wfile.write(body)


def _serve(script):
    _Script.script, _Script.hits = list(script), 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Script)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = OpenAICompatClient(f"http://127.0.0.1:{server.server_address[1]}/v1", "m-1", "k", rpm=6000,
                                base_backoff=0.01, max_backoff=0.05, timeout=5)
    return server, client


def test_real_http_503s_then_success():
    server, client = _serve(["503", "503", "200"])
    try:
        assert client.complete(MSG).text == "served" and _Script.hits == 3
    finally:
        server.shutdown()


def test_real_http_dropped_connection_and_truncated_body_are_retried():
    server, client = _serve(["close", "truncate", "200"])
    try:
        assert client.complete(MSG).text == "served" and _Script.hits == 3
    finally:
        server.shutdown()


def test_real_http_non_retryable_status_fails_fast():
    server, client = _serve(["400"])
    try:
        with pytest.raises(LLMError):
            client.complete(MSG)
        assert _Script.hits == 1
    finally:
        server.shutdown()
