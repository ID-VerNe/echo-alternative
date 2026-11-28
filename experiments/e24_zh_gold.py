"""E24: 从 E23 的句对里筛出高置信区段，组成可直接当参照的"段级"金标准。

筛选逻辑（保守优先，宁可少要，不要脏）：
1. 长度比落在 [1.25, 2.05]；
2. 两侧都有足够长度（英文 ≥8 词、中文 ≥15 汉字），排除被 OCR 切断的碎片；
3. 两侧都有锚点（数字/拉丁串）时必须至少共享一个；只有一侧有锚点则丢弃；
4. 只保留连续 ≥3 个合格句对组成的区段（漂移通常成片出现，孤立的合格对不可信）。
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
    ratio = r["ratio"]
    if not (1.25 <= ratio <= 2.05):
        return False
    if words(r["en"]) < 8 or cjk_len(r["zh"]) < 15:
        return False
    a, b = set(r["anchors_en"]), set(r["anchors_zh"])
    if a and b:
        return bool(a & b)
    if a or b:
        return False
    return True


def main() -> None:
    book = "farm4"
    rows = [json.loads(l) for l in (ROOT / "artifacts" / f"zh_pairs_{book}.jsonl").read_text(encoding="utf-8").splitlines()]
    flags = [ok_pair(r) for r in rows]

    golds, i = [], 0
    while i < len(rows):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(rows) and flags[j]:
            j += 1
        run = rows[i:j]
        # 连续区段内在句序上也必须连续
        if len(run) >= 3:
            k = 0
            while k + 3 <= len(run):
                take = run[k : k + 4]
                if words(" ".join(x["en"] for x in take)) <= 150:
                    golds.append(
                        {
                            "book": book,
                            "ei": [take[0]["ei"][0], take[-1]["ei"][1]],
                            "zi": [take[0]["zi"][0], take[-1]["zi"][1]],
                            "en": " ".join(x["en"] for x in take),
                            "zh": "".join(x["zh"] for x in take),
                            "ratio": round(cjk_len("".join(x["zh"] for x in take)) / max(1, words(" ".join(x["en"] for x in take))), 2),
                            "n_sent": len(take),
                        }
                    )
                k += 4
        i = j

    out = ROOT / "artifacts" / f"zh_gold_{book}.jsonl"
    out.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in golds), encoding="utf-8")
    print(f"ok_pairs={sum(flags)}/{len(rows)} chunks={len(golds)}")
    print("written", out)
    for g in golds[:3]:
        print("=" * 80)
        print(g["en"][:260])
        print(g["zh"][:220])


if __name__ == "__main__":
    main()
