"""锚点库：按风格强度挑选、按表达功能打标、按 intent 检索。
不按知识点建索引 —— 索引键是"这段话在干什么"（反驳/说理/叙事/自嘲）。"""
import json
import random
import re
from collections import Counter

from . import textutil

# ----------------- 风格强度 -----------------
HEDGE = ["actually", "frankly", "honestly", "literally", "apparently", "of course",
         "obviously", "clearly", "naturally", "admittedly"]
HYPE = ["most", "worst", "best", "ever", "never", "nothing", "everything", "entire",
        "absolutely", "completely", "utterly", "enormous"]
INSULT = ["idiot", "stupid", "moron", "rubbish", "awful", "appalling", " dreadful",
          "useless", "nonsense", "mad", "lunatic", "hell", "damn", "bloody"]
SELF = ["i'm", "i am", "my own", "myself", "i have no idea", "i don't know",
        "fool", "idiot", "incompetent", "useless"]

RANT = ["?"] + INSULT
ARG = ["because", "so ", "which means", "the reason", "in fact", "actually",
       "of course", "why", "therefore", "otherwise", "if you"]
ANEC = ["last week", "last month", "last year", "years ago", "when i was", "the other day",
        "then", "after", "before", "once", "one day", "i remember", "i once"]


def _count(text, words):
    t = text.lower()
    return sum(t.count(w) for w in words)


def intensity(p):
    t = p["text"].lower()
    n = p["wc"] or 1
    s = 0.0
    s += p["punct"].get("?", 0) / max(p["n_sent"], 1) * 2.0
    s += _count(t, HEDGE) / n * 1000 * 0.6
    s += _count(t, HYPE) / n * 1000 * 0.8
    s += _count(t, INSULT) / n * 1000 * 1.2
    s += p["hist"][0] / max(p["n_sent"], 1) * 1.5          # 短句占比
    s += _count(t, SELF) / n * 1000 * 0.5
    s += 1.0 if '"' in p["text"] or "\u201c" in p["text"] else 0.0
    return round(s, 4)


# ----------------- 表达功能 -----------------
def intent_scores(text):
    t = text.lower()
    sents = textutil.split_sentences(text)
    ns = len(sents) or 1
    fw = {"q": sum(1 for s in sents if s.rstrip('"\u201d').endswith("?")) / ns,
          "rant": _count(t, INSULT) / ns,
          "arg": _count(t, ARG) / ns,
          "anec": _count(t, ANEC) / ns,
          "self": _count(t, SELF) / ns,
          "first": _count(t, [" i ", "i ", "my ", " me "]) / ns}
    fw["rant"] += fw["q"] * 0.8
    fw["self"] += fw["first"] * 0.3
    return fw


INTENTS = ["rant", "argument", "anecdote", "selfmock"]


def classify_intent(text):
    f = intent_scores(text)
    scores = {"rant": f["rant"] + f["q"], "argument": f["arg"],
              "anecdote": f["anec"] + f["first"] * 0.4,
              "selfmock": f["self"] + f["first"] * 0.5}
    return max(scores, key=scores.get)


# ----------------- 选锚点 -----------------
def build_anchors(paras, target=640, seed=7):
    rnd = random.Random(seed)
    for p in paras:
        p["intensity"] = intensity(p)
        p["intent"] = classify_intent(p["text"])
    pool = [p for p in paras if 100 <= p["wc"] <= 380]
    pool.sort(key=lambda p: -p["intensity"])
    top = pool[: int(len(pool) * 0.35)]
    by_intent = {}
    for p in top:
        by_intent.setdefault(p["intent"], []).append(p)
    out = []
    per = target // len(INTENTS)
    for it in INTENTS:
        cand = by_intent.get(it, [])
        rnd.shuffle(cand)
        # 跨书目均衡：每本书最多取 3 条
        counter, picked = Counter(), []
        for p in cand:
            if counter[p["book"]] >= 3:
                continue
            picked.append(p)
            counter[p["book"]] += 1
            if len(picked) >= per:
                break
        out.extend(picked)
    for i, p in enumerate(out):
        p["aid"] = f"A{i:04d}"
    return out


def stats(anchors):
    c = Counter(p["intent"] for p in anchors)
    g = Counter(p["genre"] for p in anchors)
    print("锚点:", len(anchors), dict(c), dict(g))


def sample_anchors(anchors, intent, k=4, seed=None):
    rnd = random.Random(seed)
    cand = [p for p in anchors if p["intent"] == intent]
    if len(cand) < k:
        cand = anchors
    rnd.shuffle(cand)
    picked, books = [], set()
    for p in cand:
        if p["book"] in books and len(books) < 4:
            continue
        picked.append(p)
        books.add(p["book"])
        if len(picked) >= k:
            break
    return picked


def dump(anchors, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in anchors:
            f.write(json.dumps({k: p[k] for k in ("aid", "book", "genre", "intent", "intensity", "wc", "text")},
                               ensure_ascii=False) + "\n")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
