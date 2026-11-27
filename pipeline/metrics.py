"""内容无关的风格度量：Cosine Delta（虚词 z-score 质心）、字符 4-gram kNN、
句长 JS 散度、标点余弦、n-gram 重叠（防抄袭）、事实保留率。"""
import math
import random
import re
import statistics
from collections import Counter

from . import textutil

FUNCTION_WORDS = """the a an and or but if because so that this these those it its
i you he she we they me him her us them my your his our their mine yours
is are was were be been being am do does did done have has had having
will would can could may might must shall should
not no nor yes
of in on at to from by with without for about against between into through during
before after above below up down out off over under again further then once
here there when where why how all any both each few more most other some such
only own same than too very just now also even still yet already
as like unlike while although though unless until since
what who whom whose which
per via etc
""".split()

# 只在"有辨识度"的标点上做余弦（句号逗号占比过高会淹没差异）
DISTINCT_PUNCT = [";", ":", "-", "(", ")", "?", "!", '"', "\u201c", "\u201d", "\u2026", "%"]


def cosine(a, b):
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
    na = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    nb = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (na * nb)


class StyleMetrics:
    def __init__(self, ref_paras, neg_paras, knn_ref=400, knn_neg=300, seed=7):
        self.ref, self.neg = ref_paras, neg_paras
        self._fn_stats()
        self._char_features()
        rnd = random.Random(seed)
        self._knn_ref = [self._char_tfidf(p["text"]) for p in rnd.sample(ref_paras, min(knn_ref, len(ref_paras)))]
        self._knn_neg = [self._char_tfidf(p["text"]) for p in rnd.sample(neg_paras, min(knn_neg, len(neg_paras)))]

    # ---------- 虚词 Cosine Delta ----------
    def _fn_vec(self, text):
        toks = textutil.words(text)
        n = len(toks) or 1
        c = Counter(toks)
        return {w: c[w] / n * 100 for w in FUNCTION_WORDS}

    def _z(self, v):
        return {w: (v.get(w, 0) - self.fn_mean[w]) / self.fn_std[w] for w in FUNCTION_WORDS}

    def _fn_stats(self):
        allp = self.ref + self.neg
        vecs = [self._fn_vec(p["text"]) for p in allp]
        self.fn_mean, self.fn_std = {}, {}
        for w in FUNCTION_WORDS:
            vals = [v.get(w, 0) for v in vecs]
            self.fn_mean[w] = statistics.mean(vals)
            self.fn_std[w] = statistics.pstdev(vals) or 1e-6

        def centroid(paras):
            c = Counter()
            for p in paras:
                for w, x in self._z(self._fn_vec(p["text"])).items():
                    c[w] += x / max(len(paras), 1)
            return dict(c)

        self.cent_ref = centroid(self.ref)
        self.cent_neg = centroid(self.neg)

    def delta(self, text):
        """返回 (距离克拉克森质心, 距离其他人质心)，越小越近。"""
        z = self._z(self._fn_vec(text))
        return (round(1 - cosine(z, self.cent_ref), 4),
                round(1 - cosine(z, self.cent_neg), 4))

    # ---------- 字符 4-gram ----------
    @staticmethod
    def _char_vec(text, n=4):
        t = re.sub(r"\s+", " ", text.lower())
        c = Counter(t[i:i + n] for i in range(len(t) - n + 1))
        total = sum(c.values()) or 1
        return {k: v / total for k, v in c.items()}

    def _char_features(self, max_feat=4000):
        docs = [self._char_vec(p["text"]) for p in self.ref + self.neg]
        df = Counter()
        for d in docs:
            df.update(d.keys())
        top = [k for k, _ in df.most_common(max_feat)]
        self.feat = set(top)
        n_doc = len(docs)
        self.idf = {k: math.log((n_doc + 1) / (df.get(k, 0) + 1)) for k in self.feat}

    def _char_tfidf(self, text):
        v = {k: x * self.idf[k] for k, x in self._char_vec(text).items() if k in self.feat}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1e-9
        return {k: x / norm for k, x in v.items()}

    def char_margin(self, text, k=6):
        """与最近的 k 段克拉克森 / 其他人段落的平均余弦相似度。"""
        q = self._char_tfidf(text)
        sr = sorted((cosine(q, v) for v in self._knn_ref), reverse=True)[:k]
        sn = sorted((cosine(q, v) for v in self._knn_neg), reverse=True)[:k]
        return (round(sum(sr) / len(sr), 4) if sr else 0.0,
                round(sum(sn) / len(sn), 4) if sn else 0.0)

    # ---------- 句长 / 标点 ----------
    def ref_hist(self):
        h = [0] * 6
        for p in self.ref:
            for i, v in enumerate(p["hist"]):
                h[i] += v
        n = sum(h) or 1
        return [x / n for x in h]

    def sent_js(self, text):
        h = textutil.sentence_len_hist(text)
        n = sum(h) or 1
        p = [x / n for x in h]
        q = self.ref_hist()
        js = 0.0
        for a, b in zip(p, q):
            m = (a + b) / 2 or 1e-9
            if a > 0:
                js += 0.5 * a * math.log2(a / m)
            if b > 0:
                js += 0.5 * b * math.log2(b / m)
        return round(js, 4)

    def ref_punct(self):
        c = Counter()
        for p in self.ref:
            c.update(p["punct"])
        total = sum(p["wc"] for p in self.ref) or 1
        return {k: c[k] / total * 100 for k in DISTINCT_PUNCT}

    def punct_cos(self, text):
        c = Counter(textutil.punct_profile(text))
        n = textutil.word_count(text) or 1
        v = {k: c.get(k, 0) / n * 100 for k in DISTINCT_PUNCT}
        return round(cosine(v, self.ref_punct()), 4)

    # ---------- 重叠 / 事实 ----------
    @staticmethod
    def ngrams(text, n=8):
        w = textutil.words(text)
        return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

    def overlap(self, text, sources, n=8):
        g = self.ngrams(text, n)
        if not g:
            return 0.0, 0
        src = set()
        for s in sources:
            src |= self.ngrams(s, n)
        hit = len(g & src)
        return round(hit / len(g), 4), hit

    @staticmethod
    def longest_common_words(a, b):
        wa, wb = textutil.words(a), textutil.words(b)
        idx = {}
        for i in range(len(wb) - 2):
            idx.setdefault(" ".join(wb[i:i + 3]), i)
        best = 0
        for i in range(len(wa) - 2):
            key = " ".join(wa[i:i + 3])
            if key in idx:
                j = idx[key]
                k = 3
                while i + k < len(wa) and j + k < len(wb) and wa[i + k] == wb[j + k]:
                    k += 1
                best = max(best, k)
        return best


NUM_RE = re.compile(r"\b\d[\d,.]*\b")
CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
STOP_CAPS = {"The", "This", "That", "But", "And", "For", "With", "You", "They", "There",
             "What", "When", "Why", "How", "Not", "It", "Its", "His", "Her", "Our", "Their"}


def facts(text):
    nums = set(NUM_RE.findall(text))
    caps = {w for w in CAP_RE.findall(text) if w not in STOP_CAPS}
    return nums, caps


def fact_recall(notes, final):
    n0, c0 = facts(notes)
    n1, c1 = facts(final)
    return (round(len(n0 & n1) / len(n0), 4) if n0 else 1.0,
            round(len(c0 & c1) / len(c0), 4) if c0 else 1.0,
            sorted(n0 - n1), sorted(c0 - c1))
