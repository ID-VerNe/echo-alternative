"""E23b: 章节锚定的英中句级对齐（农场4）。

先用 23 个章节标题把两边切成 23 段，再在每段内部跑单调 DP，
这样对齐误差不会被累积放大到整本书。
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "图书语料库"

CJK = re.compile(r"[\u4e00-\u9fff]")
SUP = "¹²³⁴⁵⁶⁷⁸⁹⁰"

CHAPTERS = [
    ("It's Ostrich Time", "现在是养鸵鸟的时代"),
    ("Chipping Tikka Masala", "乡村小镇的印度咖喱梦"),
    ("Global Warming's Day Off", "全球变暖请假了"),
    ("Blessed are the Brickmakers", "做砖的人有福了"),
    ("Dead Lorry, Yellow Lorry", "废卡车，黄卡车"),
    ("Decent, Hard-working Farmers in the Community", "社区中正派又勤劳的农民"),
    ("A Home-grown Christmas", "从农场到餐桌的圣诞物语"),
    ("The End is Nigh!", "末日将至！"),
    ("Das Traktor", "开上拖拉机揭竿起义吧"),
    ("In My Back Yard", "我家后院"),
    ("Love is . . . Mucking Out Pigs", "爱就是……清理猪圈"),
    ("A Fate Worse Than Golf?", "还能有比高尔夫更糟糕的命运？"),
    ("A Load of Old Rubbish", "一堆破烂儿"),
    ("My Cows Don't Want Your Cuddles", "我的牛不想要你的拥抱"),
    ("Metal Box Nation", "集装箱之国"),
    ("A Cotswold Co-op", "科茨沃尔德合作社"),
    ("Chadlington Thunderbirds are Go!", "查德林顿雷鸟小队，出发！"),
    ("Howzat!", "这招怎么样！"),
    ("Not So Top Soil . . .", "治标不治本"),
    ("Brand Ambassador't", "向大牌说不"),
    ("Don't be a Canute", "别学克努特大帝"),
    ("What's Brown and Sticky?", "什么东西黄黄的、黏黏的？"),
    ("The Best Pub . . . in the World", "世界上最棒的酒吧"),
]

BOOKS = {
    "farm4": {
        "en": CORPUS / "markdown/克拉克森的农场4：自食其果/克拉克森的农场4：自食其果.md",
        "zh": CORPUS / "中文译本/克拉克森的农场4：自食其果 三校样/克拉克森的农场4：自食其果 三校样.md",
        "en_start": "The contents of this book first appeared",
        "en_end": "World According To CLARKSON",
        "zh_start": "\n现在是养鸵鸟的时代\n",
    },
}


def norm(s: str) -> str:
    s = s.replace("\u00ad", "")
    s = re.sub(r"-\s+", "-", s)      # Hard- working -> Hard-working
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def strip_noise(t: str) -> str:
    t = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", t)
    t = re.sub(r"<sup>\d+</sup>", "", t)
    t = re.sub(r"\\\((.*?)\\\)", r"\1", t)
    return t


def body_en(t: str, start: str, end: str) -> str:
    t = strip_noise(t)
    lines = []
    for raw in t.split("\n"):
        l = raw.strip().lstrip("#").strip()
        if not l or re.fullmatch(r"(1\.\s*)+1?\.?", l) or re.fullmatch(r"\d{1,4}", l):
            continue
        lines.append(l)
    body = "\n".join(lines)
    i = body.find(start)
    body = body[body.find("\n", i) + 1:] if i >= 0 else body
    j = body.find(end)
    return body[:j] if j > 0 else body


def body_zh(t: str, start: str) -> str:
    t = strip_noise(t)
    lines = [l.strip() for l in t.split("\n") if l.strip()]
    # 译者脚注行（"1 原文为 …"）会被切句器当成正文，先剔除
    lines = [l for l in lines if not re.match(r"^\d{1,3}\s*[^\d\s]", l)]
    body = "\n".join(lines)
    body = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", body)
    hits = list(_title_pattern(start.strip()).finditer(body))
    if not hits:
        return body
    # 第一次出现在目录里，取最后一次（正文标题）
    cut = hits[-1].end()
    return body[cut:]


def split_en(t: str) -> list[str]:
    t = re.sub(r"-\s*\n", "", t).replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[“‘\"A-Z0-9£$])", t)
    return [s.strip().strip("#").strip() for s in parts if len(s.split()) >= 3]


def split_zh(t: str) -> list[str]:
    t = t.replace("\n", "")
    parts = re.split(r"(?<=[。！？…”』」])", t)
    return [s.strip() for s in parts if len(CJK.findall(s)) >= 4]


NUM = re.compile(r"\d+(?:[.,]\d+)?%?")
LAT = re.compile(r"[A-Za-z][A-Za-z'&-]{2,}")


def anchors(s: str) -> set[str]:
    return set(NUM.findall(s)) | {w.lower() for w in LAT.findall(s)}


def cjk_len(s: str) -> int:
    return len(CJK.findall(s))


EXPECT = 1.55


def pair_cost(en: str, zh: str) -> float:
    ratio = cjk_len(zh) / max(1, len(en.split()))
    cost = abs(math.log(max(ratio, 0.2) / EXPECT))
    ea, za = anchors(en), anchors(zh)
    if ea and za:
        cost += -0.25 if ea & za else 0.55
    return cost


merge_pen = 0.05
skip_pen = 2.4


def align(es: list[str], zs: list[str]):
    n, m = len(es), len(zs)
    INF = float("inf")
    d = [[INF] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
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
                cand.append((2, 1, pair_cost(es[i] + " " + es[i + 1], zs[j]) + merge_pen))
            if i < n and j + 1 < m:
                cand.append((1, 2, pair_cost(es[i], zs[j] + zs[j + 1]) + merge_pen))
            if i < n:
                cand.append((1, 0, skip_pen))
            if j < m:
                cand.append((0, 1, skip_pen))
            for di, dj, c in cand:
                ni, nj = i + di, j + dj
                if cur + c < d[ni][nj]:
                    d[ni][nj] = cur + c
                    bt[ni][nj] = (i, j)
    segs, i, j = [], n, m
    while (i, j) != (0, 0):
        pi, pj = bt[i][j]
        segs.append((pi, i, pj, j))
        i, j = pi, pj
    return segs[::-1]


def _title_pattern(needle: str) -> re.Pattern:
    """标题定位：容忍 OCR 的换行、连字符、空格、省略号差异，但保留原文索引。"""
    tokens = [t for t in re.split(r"[\s\-–—.…]+", needle) if t]
    body = r"[\s\-–—.…]*".join(re.escape(t) for t in tokens)
    return re.compile(body, re.IGNORECASE)


def locate(hay: str, needle: str, after: int) -> int:
    m = _title_pattern(needle).search(hay, after)
    return m.start() if m else -1


def chapter_spans(text_en: str, text_zh: str):
    """返回每章正文的起始位置（标题之后），避免标题被粘进首句。"""
    en_spans, zh_spans = [], []
    pos_en = pos_zh = 0
    for k, (en_t, zh_t) in enumerate(CHAPTERS):
        m_en = _title_pattern(en_t).search(text_en, pos_en)
        en_spans.append(m_en.end() if m_en else -1)
        if k == 0:
            zh_spans.append(0)
            continue
        m_zh = _title_pattern(zh_t.strip(SUP)).search(text_zh, pos_zh)
        zh_spans.append(m_zh.end() if m_zh else -1)
        if m_zh:
            pos_zh = m_zh.end()
        if m_en:
            pos_en = m_en.end()
    return en_spans, zh_spans


def main() -> None:
    book = sys.argv[1] if len(sys.argv) > 1 else "farm4"
    cfg = BOOKS[book]
    en_raw = body_en(cfg["en"].read_text(encoding="utf-8"), cfg["en_start"], cfg["en_end"])
    zh_raw = body_zh(cfg["zh"].read_text(encoding="utf-8"), cfg["zh_start"])
    en_pos, zh_pos = chapter_spans(en_raw, zh_raw)
    print("en chapter hits:", sum(1 for x in en_pos if x >= 0), "/", len(CHAPTERS))
    print("zh chapter hits:", sum(1 for x in zh_pos if x >= 0), "/", len(CHAPTERS))

    # 用可用锚点切段；缺失的位置用相邻已定位章节的比例插值
    def fill(pos: list[int], total: int) -> list[int]:
        known = [(k, p) for k, p in enumerate(pos) if p >= 0]
        out = pos[:]
        for k, p in enumerate(out):
            if p >= 0:
                continue
            prev = max([x for x in known if x[0] < k], default=None, key=lambda t: t[0])
            nxt = min([x for x in known if x[0] > k], default=None, key=lambda t: t[0])
            if prev and nxt:
                ratio = (k - prev[0]) / (nxt[0] - prev[0])
                out[k] = int(prev[1] + ratio * (nxt[1] - prev[1]))
            elif prev:
                out[k] = prev[1]
            elif nxt:
                out[k] = 0
        out.append(total)
        return out

    en_cut = fill(en_pos, len(en_raw))
    zh_cut = fill(zh_pos, len(zh_raw))
    en_cut[0] = max(0, en_cut[0])
    zh_cut[0] = max(0, zh_cut[0])

    rows = []
    for k in range(len(CHAPTERS)):
        a0, a1 = en_cut[k], en_cut[k + 1]
        b0, b1 = zh_cut[k], zh_cut[k + 1]
        if a0 >= a1 or b0 >= b1:
            continue
        es = split_en(en_raw[a0:a1])
        zs = split_zh(zh_raw[b0:b1])
        for i0, i1, j0, j1 in align(es, zs):
            if i1 - i0 == 0 or j1 - j0 == 0:
                continue
            en_s, zh_s = " ".join(es[i0:i1]), "".join(zs[j0:j1])
            rows.append(
                {
                    "book": book,
                    "ch": k,
                    "ch_title": CHAPTERS[k][1],
                    "ei": [i0, i1],
                    "zi": [j0, j1],
                    "en": en_s,
                    "zh": zh_s,
                    "ratio": round(cjk_len(zh_s) / max(1, len(en_s.split())), 2),
                    "anchors_en": sorted(anchors(en_s)),
                    "anchors_zh": sorted(anchors(zh_s)),
                }
            )
    out = ROOT / "artifacts" / f"zh_pairs_{book}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print(f"pairs={len(rows)} chapters={len(CHAPTERS)} written={out}")


if __name__ == "__main__":
    main()
