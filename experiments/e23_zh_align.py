"""E23: 英中句级对齐，构建可用的平行语料（以农场4为试点）。

思路：两端先清洗成句流，再用单调 DP 对齐；代价由「汉字/英文词长度比」与
「数字、拉丁字母串等跨语言锚点是否一致」共同决定。不做章节切分，靠锚点
自纠偏。
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "图书语料库"

BOOKS = {
    "farm4": {
        "en": CORPUS / "markdown/克拉克森的农场4：自食其果/克拉克森的农场4：自食其果.md",
        "zh": CORPUS / "中文译本/克拉克森的农场4：自食其果 三校样/克拉克森的农场4：自食其果 三校样.md",
        "en_start": "The contents of this book first appeared",
        "en_end": "World According To CLARKSON",
        "zh_start": "\n现在是养鸵鸟的时代\n",
    },
}

CJK = re.compile(r"[\u4e00-\u9fff]")


def _strip_noise(t: str) -> str:
    t = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", t)
    t = re.sub(r"<sup>\d+</sup>", "", t)
    t = re.sub(r"\\\((.*?)\\\)", r"\1", t)
    return t


def clean_en(t: str, start: str, end: str) -> str:
    t = _strip_noise(t)
    lines = []
    for raw in t.split("\n"):
        l = raw.strip().lstrip("#").strip()
        if not l:
            continue
        if re.fullmatch(r"(1\.\s*)+1?\.?", l):
            continue
        if re.fullmatch(r"\d{1,4}", l):
            continue
        lines.append(l)
    body = "\n".join(lines)
    i = body.find(start)
    body = body[body.find("\n", i) + 1 :] if i >= 0 else body
    j = body.find(end)
    if j > 0:
        body = body[:j]
    return body


def clean_zh(t: str, start: str) -> str:
    t = _strip_noise(t)
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    body = "\n".join(lines)
    i = body.find(start)
    body = body[i + len(start.strip()) :] if i >= 0 else body
    # 去掉书末的版权/广告页
    for mark in ("图书在版编目", "版权所有", "封底"):
        pass
    return body


def split_en(t: str) -> list[str]:
    t = re.sub(r"-\s*\n", "", t).replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[“‘\"A-Z0-9£$])", t)
    out = []
    for s in parts:
        s = s.strip().strip("#").strip()
        if len(s.split()) >= 3:
            out.append(s)
    return out


def split_zh(t: str) -> list[str]:
    t = t.replace("\n", "")
    parts = re.split(r"(?<=[。！？…”』」])", t)
    out = []
    for s in parts:
        s = s.strip()
        if len(CJK.findall(s)) >= 4:
            out.append(s)
    return out


NUM = re.compile(r"\d+(?:[.,]\d+)?%?")
LAT = re.compile(r"[A-Za-z][A-Za-z'&-]{2,}")


def anchors(s: str) -> set[str]:
    a = set(NUM.findall(s))
    a |= {w.lower() for w in LAT.findall(s)}
    return a


def cjk_len(s: str) -> int:
    return len(CJK.findall(s))


def words(s: str) -> int:
    return len(s.split())


EXPECT = 1.55  # 汉字/英文词


def pair_cost(en: str, zh: str) -> float:
    ratio = cjk_len(zh) / max(1, words(en))
    cost = abs(math.log(max(ratio, 0.2) / EXPECT))
    ea, za = anchors(en), anchors(zh)
    if ea and za:
        inter = ea & za
        if inter:
            cost -= 0.25
        else:
            cost += 0.55
    return cost


def align(es: list[str], zs: list[str]) -> list[tuple[int, int, int, int]]:
    """单调 DP，允许 1-1 / 1-2 / 2-1 / 1-0 / 0-1。返回 (i0,i1,j0,j1)。"""
    n, m = len(es), len(zs)
    INF = float("inf")
    d = [[INF] * (m + 1) for _ in range(n + 1)]
    bt: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    d[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            cur = d[i][j]
            if cur == INF:
                continue
            cand = []
            if i < n and j < m:
                cand.append((1, 1, pair_cost(es[i], zs[j])))
            if i + 1 < n and j < m:
                cand.append((2, 1, pair_cost(es[i] + " " + es[i + 1], zs[j]) + 0.15))
            if i < n and j + 1 < m:
                cand.append((1, 2, pair_cost(es[i], zs[j] + zs[j + 1]) + 0.15))
            if i < n:
                cand.append((1, 0, 1.4))
            if j < m:
                cand.append((0, 1, 1.4))
            for di, dj, c in cand:
                ni, nj = i + di, j + dj
                nd = cur + c
                if nd < d[ni][nj]:
                    d[ni][nj] = nd
                    bt[ni][nj] = (i, j)
    # 回溯
    out = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = bt[i][j]  # type: ignore[misc]
        out.append((pi, i, pj, j))
        i, j = pi, pj
    return out[::-1]


def load(book: str):
    cfg = BOOKS[book]
    en = split_en(clean_en(cfg["en"].read_text(encoding="utf-8"), cfg["en_start"], cfg["en_end"]))
    zh = split_zh(clean_zh(cfg["zh"].read_text(encoding="utf-8"), cfg["zh_start"]))
    return en, zh


def main() -> None:
    book = sys.argv[1] if len(sys.argv) > 1 else "farm4"
    es, zs = load(book)
    print(f"{book}: en_sents={len(es)} zh_sents={len(zs)}")
    segs = align(es, zs)
    rows = []
    for i0, i1, j0, j1 in segs:
        if i1 - i0 == 0 or j1 - j0 == 0:
            continue
        rows.append(
            {
                "book": book,
                "ei": [i0, i1],
                "zi": [j0, j1],
                "en": " ".join(es[i0:i1]),
                "zh": "".join(zs[j0:j1]),
                "ratio": round(cjk_len("".join(zs[j0:j1])) / max(1, words(" ".join(es[i0:i1]))), 2),
                "anchors_en": sorted(anchors(" ".join(es[i0:i1]))),
                "anchors_zh": sorted(anchors("".join(zs[j0:j1]))),
            }
        )
    out = ROOT / "artifacts" / f"zh_pairs_{book}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    # 质量概览
    ok = [r for r in rows if r["anchors_en"] and r["anchors_zh"] and set(r["anchors_en"]) & set(r["anchors_zh"])]
    both = [r for r in rows if r["anchors_en"] and r["anchors_zh"]]
    print(f"pairs={len(rows)} with_anchor_both={len(both)} matched={len(ok)}")
    ratios = sorted(r["ratio"] for r in rows)
    print("ratio median", ratios[len(ratios) // 2], "p10", ratios[len(ratios) // 10], "p90", ratios[len(ratios) * 9 // 10])
    print("written", out)


if __name__ == "__main__":
    main()
