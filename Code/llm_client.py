"""
OpenRouter LLM client — free-tier models only.

All LLM calls in the system go through chat(). The model is validated
at import time: any model not ending in ':free' raises ValueError so
paid models can never be accidentally used.

Rate-limit handling: free-tier models return 429s on cold starts and
burst traffic. chat() retries up to 3 times with exponential backoff.
"""
import httpx
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import OPEN_ROUTER_API_KEY

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Model selection (must end in :free) ───────────────────────────────────
# meta-llama/llama-3.3-70b-instruct:free  — best reasoning on free tier
# deepseek/deepseek-r1:free               — alternative reasoning model
FREE_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

_RETRY_DELAYS = (5, 15, 30)   # seconds between attempts on 429


def _validate_free(model: str):
    if not model.endswith(":free"):
        raise ValueError(
            f"[llm_client] Model '{model}' is not a free-tier model. "
            f"Only models ending in ':free' are permitted."
        )


_validate_free(FREE_MODEL)   # fail fast at import if someone changes the constant


# ── Public interface ───────────────────────────────────────────────────────

def chat(messages: list, max_tokens: int = 500, model: str = FREE_MODEL) -> str:
    """
    Send a chat request to OpenRouter and return the assistant message text.
    Retries up to 3 times on 429 (rate-limit) with exponential backoff.

    Raises:
        ValueError:        if a non-free model is requested
        httpx.HTTPError:   on non-429 HTTP failure after all retries
    """
    _validate_free(model)

    headers = {
        "Authorization": f"Bearer {OPEN_ROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ai-trading-project",
        "X-Title": "AI Quant Trader",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    last_exc = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            r = httpx.post(OPENROUTER_URL, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 and delay is not None:
                print(f"  [LLM] 429 rate-limit — retrying in {delay}s (attempt {attempt}/3)...")
                time.sleep(delay)
                last_exc = exc
            else:
                raise
        except Exception as exc:
            last_exc = exc
            if delay is None:
                raise

    raise last_exc  # exhausted retries


def chat_json(messages: list, max_tokens: int = 500, model: str = FREE_MODEL) -> dict:
    """
    Like chat() but parses and returns a JSON dict.
    Strips markdown fences if the model wraps output in ```json ... ```.
    """
    raw = chat(messages, max_tokens=max_tokens, model=model)
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) >= 2 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
