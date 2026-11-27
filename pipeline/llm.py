"""LLM 客户端（OpenAI 兼容）。用 curl 子进程，带硬超时 + 磁盘缓存 + 重试。
缓存键 = payload 的 md5，换 prompt 即换缓存，重复实验不重复等待。"""
import hashlib
import json
import subprocess
import time

from . import config

CACHE_DIR = config.ROOT / ".cache" / "llm"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key(payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def chat(messages, temperature=0.7, max_tokens=2600, retries=3, max_time=120):
    payload = {
        "model": config.MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    key = _key(payload)
    cache = CACHE_DIR / f"{key}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")

    body_file = CACHE_DIR / f"{key}.req.json"
    body_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    last_err = None
    for attempt in range(retries):
        r = subprocess.run(
            ["curl.exe", "-sS", "--max-time", str(max_time),
             "-H", "Authorization: Bearer " + config.API_KEY,
             "-H", "Content-Type: application/json",
             "-d", "@" + str(body_file),
             config.API_BASE + "/v1/chat/completions"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        try:
            obj = json.loads(r.stdout)
            if "error" in obj:
                raise RuntimeError(str(obj["error"])[:200])
            content = (obj["choices"][0]["message"].get("content") or "").strip()
            if not content:
                raise ValueError("empty content (likely all tokens went to reasoning)")
            cache.write_text(content, encoding="utf-8")
            return content
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 + attempt * 2)
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
