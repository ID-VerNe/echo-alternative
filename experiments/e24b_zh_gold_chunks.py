"""E24b: 把章节锚定后的句对合并成"段级"金标准候选。

规则：
- 只用同一章内、句序连续的句对；
- 每对长度比 [1.25, 2.05]，两侧都有最低长度；
- 两侧都带锚点时（数字/拉丁串）必须至少共享一个；
- 合并 4–6 句，英文 55–140 词，避免过短或过长。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CJK = re.compile(r"[\u4e00-\u9fff]")


def cjk_len(s: str) -> int:
    return len(CJK.findall(s))


def words(s: str) -> int:
    return len(s.split())


def ok_pair(r: dict) -> bool:
    if not (1.25 <= r["ratio"] <= 2.05):
        return False
    if words(r["en"]) < 6 or cjk_len(r["zh"]) < 12:
        return False
    a, b = set(r["anchors_en"]), set(r["anchors_zh"])
    if a and b and not (a & b):
        return False
    if not r["en"].rstrip().endswith((".", "!", "?", '"', "’")):
        return False
    return True


def main() -> None:
    book = "farm4"
    rows = [json.loads(l) for l in (ROOT / "artifacts" / f"zh_pairs_{book}.jsonl").read_text(encoding="utf-8").splitlines()]
    rows.sort(key=lambda r: (r["ch"], r["ei"][0]))
    chunks = []
    by_ch: dict[int, list[dict]] = {}
    for r in rows:
        by_ch.setdefault(r["ch"], []).append(r)

    for ch, seq in by_ch.items():
        i = 0
        while i < len(seq):
            take = []
            while i < len(seq) and len(take) < 6:
                r = seq[i]
                if not ok_pair(r):
                    break
                # 句序必须连续（后一对的 en 起点 = 前一对的 en 终点）
                if take and r["ei"][0] != take[-1]["ei"][1]:
                    break
                if take and r["zi"][0] != take[-1]["zi"][1]:
                    break
                if take and words(" ".join(x["en"] for x in take + [r])) > 140:
                    break
                take.append(r)
                i += 1
            if len(take) >= 4 and 55 <= words(" ".join(x["en"] for x in take)) <= 140:
                chunks.append(
                    {
                        "book": book,
                        "id": f"{book}-c{ch:02d}-{take[0]['ei'][0]:04d}",
                        "ch": ch,
                        "ch_title": take[0]["ch_title"],
                        "n_sent": len(take),
                        "ratio": round(cjk_len("".join(x["zh"] for x in take)) / max(1, words(" ".join(x["en"] for x in take))), 2),
                        "en": " ".join(x["en"] for x in take),
                        "zh": "".join(x["zh"] for x in take),
                    }
                )
            if not take:
                i += 1
    out = ROOT / "artifacts" / f"zh_gold_chunks_{book}.jsonl"
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    print(f"chunks={len(chunks)}  avg_sent={sum(c['n_sent'] for c in chunks)/max(1,len(chunks)):.1f}")
    print("written", out)
    for c in chunks[:2]:
        print("=" * 80)
        print("EN:", c["en"][:260])
        print("ZH:", c["zh"][:220])


if __name__ == "__main__":
    main()
