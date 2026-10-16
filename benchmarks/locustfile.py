"""
Locust load test for the vLLM OpenAI-compatible /v1/chat/completions endpoint.

Measures, per request, via streaming SSE chunks:
  - TTFT (time-to-first-token): time from request start to the first streamed token.
  - ITL (inter-token latency): average gap between subsequent streamed tokens.
  - Total tokens generated and end-to-end latency.

Target host defaults to the local vLLM server (see serving/docker-compose.yml).
Point it at LM Studio instead (per this project's local-only constraint) with:
    locust -f locustfile.py --host http://localhost:1234

Run headless, e.g. 10 concurrent users for 60s:
    locust -f locustfile.py --host http://localhost:8000 \
        --headless -u 10 -r 10 -t 60s --csv=results/run_10users
"""
import json
import os
import time

from locust import HttpUser, between, events, task

MODEL_NAME = os.environ.get("BENCH_MODEL_NAME", "tool-sql-llama3.1-8b")

PROMPTS = [
    {
        "system": "You are a function-calling assistant. Respond ONLY with a JSON object "
                  "matching {\"tool\": <name>, \"arguments\": {...}}. Available tools: "
                  "get_weather(location, unit), search_flights(origin, destination, date), "
                  "create_calendar_event(title, start, end).",
        "user": "What's the weather in Portland, Oregon in Celsius?",
    },
    {
        "system": "You are a function-calling assistant. Respond ONLY with a JSON object "
                  "matching {\"tool\": <name>, \"arguments\": {...}}. Available tools: "
                  "get_weather(location, unit), search_flights(origin, destination, date), "
                  "create_calendar_event(title, start, end).",
        "user": "Book a flight from LAX to SEA on 2026-12-01.",
    },
    {
        "system": "You are a text-to-SQL assistant for schema: employees(id, name, department, "
                  "salary, hire_date), departments(id, name, budget).",
        "user": "Show the 5 most recently hired employees.",
    },
    {
        "system": "You are a text-to-SQL assistant for schema: employees(id, name, department, "
                  "salary, hire_date), departments(id, name, budget).",
        "user": "What is the total salary cost per department?",
    },
]


class ChatCompletionUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def chat_completion_stream(self):
        prompt = PROMPTS[int(time.time() * 1000) % len(PROMPTS)]
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": True,
        }

        start = time.perf_counter()
        first_token_time = None
        token_timestamps = []
        total_chars = 0
        response_name = "/v1/chat/completions [stream]"

        try:
            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                stream=True,
                name=response_name,
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not decoded.startswith("data:"):
                        continue
                    data = decoded[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    token_timestamps.append(now)
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {}).get("content", "")
                        total_chars += len(delta)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

                end = time.perf_counter()
                total_latency_ms = (end - start) * 1000

                if first_token_time is not None:
                    ttft_ms = (first_token_time - start) * 1000
                    if len(token_timestamps) > 1:
                        gaps = [
                            (token_timestamps[i] - token_timestamps[i - 1]) * 1000
                            for i in range(1, len(token_timestamps))
                        ]
                        itl_ms = sum(gaps) / len(gaps)
                    else:
                        itl_ms = 0.0

                    events.request.fire(
                        request_type="TTFT",
                        name=response_name,
                        response_time=ttft_ms,
                        response_length=0,
                        exception=None,
                    )
                    events.request.fire(
                        request_type="ITL",
                        name=response_name,
                        response_time=itl_ms,
                        response_length=total_chars,
                        exception=None,
                    )
                    resp.success()
                else:
                    resp.failure("No tokens streamed before connection closed")

                _ = total_latency_ms  # captured implicitly by locust's default HTTP timer too
        except Exception as exc:  # noqa: BLE001 - surface any transport error to Locust
            events.request.fire(
                request_type="POST",
                name=response_name,
                response_time=(time.perf_counter() - start) * 1000,
                response_length=0,
                exception=exc,
            )
