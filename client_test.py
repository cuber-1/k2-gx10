#!/usr/bin/env python3
import json
import os
import sys
import urllib.error
import urllib.request


BASE_URL = os.environ.get("K2_BASE_URL", "http://127.0.0.1:30000")
BUDGET = int(os.environ.get("K2_TEST_REASONING_BUDGET", "32"))
MAX_TOKENS = int(os.environ.get("K2_TEST_MAX_TOKENS", "96"))
PROMPT = os.environ.get("K2_TEST_PROMPT", "What is 2+2? Reply with only the answer.")


def post(path, body):
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=MAX_TOKENS * 2 + 60) as response:
        return json.load(response)


def main():
    body = {
        "model": "K2 Think V2",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "seed": 42,
        "reasoning_format": "deepseek",
        "reasoning_budget_tokens": BUDGET,
        "reasoning_budget_message": "\nThe reasoning budget is exhausted. Give only the requested final answer.\n",
    }

    rendered = post("/apply-template", body)
    result = post("/v1/chat/completions", body)
    choice = result["choices"][0]
    message = choice["message"]

    print("Rendered prompt:")
    print(repr(rendered["prompt"]))
    print("\nResponse:")
    print(json.dumps({
        "finish_reason": choice["finish_reason"],
        "message": message,
        "usage": result.get("usage"),
        "timings": result.get("timings"),
    }, indent=2))

    if choice["finish_reason"] != "stop":
        raise AssertionError(f"expected finish_reason=stop, got {choice['finish_reason']!r}")
    if not (message.get("content") or "").strip():
        raise AssertionError("expected a visible final answer in message.content")

    print("\nPASS: visible content and finish_reason=stop")


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, KeyError, ValueError, AssertionError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
