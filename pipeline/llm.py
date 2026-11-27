"""LLM 客户端（OpenAI 兼容）。带重试、并发安全、reasoning 字段兼容。"""
import json
import time
import urllib.error
import urllib.request

from . import config


def chat(messages, temperature=0.7, max_tokens=2600, retries=3, timeout=180):
    payload = {
        "model": config.MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            config.API_BASE + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + config.API_KEY,
                     "Content-Type": "application/json",
                     "User-Agent": config.UA},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read().decode())
            ch = body["choices"][0]["message"]
            content = (ch.get("content") or "").strip()
            if not content:
                raise ValueError("empty content (likely all tokens went to reasoning)")
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed: {last_err}")


def parse_json(text):
    """从可能带 ```json 包裹或前后废话的输出里抠 JSON。"""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no json object found")
    return json.loads(t[start:end + 1])
