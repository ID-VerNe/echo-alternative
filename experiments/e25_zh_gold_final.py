"""E25: 人工校验后定稿中文金标准（农场4）。

校验方式：逐条读 E24b 的 59 个候选段（55 个主候选 + 4 个补充候选），
剔除句序错位（中文多/少一句）的 10 条，保留 49 条。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DROP = {
    "farm4-c01-0022",  # 英文首句是残句，中文并进了上一句
    "farm4-c01-0031",  # 中文从上一句开始
    "farm4-c02-0002",  # 中文从上一句开始
    "farm4-c05-0013",  # 中文少了半句
    "farm4-c05-0038",  # 中文从上一句开始
    "farm4-c08-0015",  # 中文从上一句开始
    "farm4-c14-0040",  # 中文首句多一句「不，这主意很糟糕」
    "farm4-c15-0044",  # 中文首句被截断
    "farm4-c17-0056",  # 中文从上一句开始
    "farm4-c19-0007",  # 中文从上一句开始
}

CJK = re.compile(r"[\u4e00-\u9fff]")


def tidy(s: str) -> str:
    s = re.sub(r"#\s*[^\n]{0,40}$", "", s)
    s = re.sub(r"\s*#\s*", "", s)
    s = s.replace("\\mathcal{L}", "£").replace("\\%", "%").replace("\\times", "×")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def main() -> None:
    chunks = [json.loads(l) for l in (ROOT / "artifacts/zh_gold_chunks_farm4.jsonl").read_text(encoding="utf-8").splitlines()]
    extra = [json.loads(l) for l in (ROOT / "artifacts/zh_gold_extra_farm4.jsonl").read_text(encoding="utf-8").splitlines()]
    seen = {c["id"] for c in chunks}
    pool = chunks + [c for c in extra if c["id"] not in seen]

    kept = []
    for c in pool:
        if c["id"] in DROP:
            continue
        c["en"], c["zh"] = tidy(c["en"]), tidy(c["zh"])
        kept.append(c)
    kept.sort(key=lambda c: (c["ch"], c["id"]))

    out = ROOT / "artifacts/zh_gold_farm4.jsonl"
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in kept), encoding="utf-8")
    n_sent = sum(c["n_sent"] for c in kept)
    zh_chars = sum(len(CJK.findall(c["zh"])) for c in kept)
    en_words = sum(len(c["en"].split()) for c in kept)
    print(f"pool={len(pool)} dropped={len(DROP)} kept={len(kept)}")
    print(f"sent_pairs={n_sent} en_words={en_words} zh_cjk={zh_chars} zh/en={zh_chars/max(1,en_words):.2f}")
    print("chapters covered:", len({c['ch'] for c in kept}), "/ 23")
    print("written", out)


if __name__ == "__main__":
    main()
