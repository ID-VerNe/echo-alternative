"""内容无关的风格向量：虚词 n-gram、句长转移、标点、句首词类、缩约度。
用 numpy 定长向量，方便 kNN 矩阵运算。"""
import math
import re
import statistics
from collections import Counter

import numpy as np

from . import textutil
from .metrics import FUNCTION_WORDS, DISTINCT_PUNCT

FW = set(FUNCTION_WORDS)
OPENERS = ["and", "but", "so", "because", "which", "it", "this", "that", "you", "i",
           "the", "there", "if", "when", "what", "of", "in", "as", "then", "now"]
CONTRACT_RE = re.compile(r"(?:n't|'s|'ll|'re|'ve|'d|'m)\b")
BUCKETS = 6


def raw_feats(text):
    toks = textutil.words(text)
    n = len(toks) or 1
    sents = textutil.split_sentences(text)
    ns = len(sents) or 1
    f = {}
    c = Counter(toks)
    for w in FW:
        f[f"u:{w}"] = c[w] / n * 100
    nb = 0
    bic = Counter()
    for i in range(len(toks) - 1):
        if toks[i] in FW and toks[i + 1] in FW:
            bic[toks[i] + " " + toks[i + 1]] += 1
            nb += 1
    for k, v in bic.items():
        f[f"b:{k}"] = v / max(nb, 1) * 100
    h = [0] * BUCKETS
    trans = Counter()
    prev = None
    for s in sents:
        b = textutil.norm_len_bucket(textutil.word_count(s))
        h[b] += 1
        if prev is not None:
            trans[(prev, b)] += 1
        prev = b
    for i in range(BUCKETS):
        f[f"h:{i}"] = h[i] / ns
    for (a, b), v in trans.items():
        f[f"t:{a}{b}"] = v / max(ns - 1, 1)
    pc = Counter()
    for s in sents:
        for p in DISTINCT_PUNCT:
            pc[p] += s.count(p)
    for p in DISTINCT_PUNCT:
        f[f"p:{p}"] = pc[p] / ns
    oc = Counter()
    for s in sents:
        w = textutil.words(s)
        if w and w[0] in OPENERS:
            oc[w[0]] += 1
    for w in OPENERS:
        f[f"o:{w}"] = oc[w] / ns
    f["c:contr"] = len(CONTRACT_RE.findall(text.lower())) / n * 100
    return f


class StyleVectorizer:
    def __init__(self, docs, max_bigrams=200):
        raw = [raw_feats(d) for d in docs]
        bdf = Counter()
        for v in raw:
            bdf.update(k for k in v if k.startswith("b:"))
        keep_b = {k for k, _ in bdf.most_common(max_bigrams)}
        self.keys = sorted({k for v in raw for k in v if k.startswith("b:") and k in keep_b} |
                           {k for v in raw for k in v if not k.startswith("b:")})
        self.index = {k: i for i, k in enumerate(self.keys)}
        d = len(self.keys)
        rows = []
        for v in raw:
            rows.append([v.get(k, 0.0) for k in self.keys])
        arr = np.array(rows, dtype=np.float64)
        self.mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std[std == 0] = 1e-6
        self.std = std

    def vec(self, text):
        v = raw_feats(text)
        x = np.array([v.get(k, 0.0) for k in self.keys], dtype=np.float64)
        z = (x - self.mean) / self.std
        n = np.linalg.norm(z) or 1e-9
        return z / n

    def matrix(self, texts):
        return np.vstack([self.vec(t) for t in texts])
