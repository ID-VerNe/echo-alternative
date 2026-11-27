"""文本基础工具：切段、切句、分词、标点画像。只用标准库 + numpy。"""
import re
import unicodedata

# ---------- 清洗 ----------
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+")
DIGITS_ONLY_RE = re.compile(r"^[\d\s.,:;!?#*\-—_=（）()]+$")
ABBREV = ["Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Jr.", "Sr.", "U.S.", "U.K.",
          "a.m.", "p.m.", "No.", "vs.", "etc.", "Inc.", "Ltd.", "Co."]


def is_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def alpha_ratio(s):
    if not s:
        return 0
    letters = sum(1 for c in s if c.isalpha() or c.isspace())
    return letters / len(s)


def clean_line(s):
    s = IMG_RE.sub("", s)
    s = HTML_RE.sub("", s)
    s = URL_RE.sub("", s)
    s = s.replace("\u00ad", "").replace("\ufeff", "")
    return s.strip()


def looks_like_toc(line):
    """目录行：很长、几乎没有句末标点、首字母大写词密集。"""
    if len(line) < 60:
        return False
    if line.count(".") > 2:
        return False
    words = line.split()
    if len(words) < 8:
        return False
    caps = sum(1 for w in words if w[:1].isupper())
    return caps / len(words) > 0.45


def split_paragraphs(raw):
    """按空行切段，把段内单换行合并；丢弃标题、图片、目录、垃圾行。"""
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = IMG_RE.sub("\n\n", raw)
    blocks = re.split(r"\n\s*\n", raw)
    out = []
    for b in blocks:
        lines = [clean_line(x) for x in b.split("\n")]
        lines = [x for x in lines if x]
        if not lines:
            continue
        # 去掉 markdown 标题行
        keep = [x for x in lines if not x.startswith("#")]
        if not keep:
            continue
        text = " ".join(keep)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < 40:
            continue
        if DIGITS_ONLY_RE.match(text):
            continue
        if looks_like_toc(text):
            continue
        if alpha_ratio(text) < 0.65:
            continue
        for bad in ("ISBN", "Copyright ©", "All rights reserved", "By the same author",
                    "Contents", "www.", "@"):
            if bad in text:
                text = ""
                break
        if not text:
            continue
        out.append(text)
    return out


SENT_SPLIT_RE = re.compile(r'(?<=[.!?])["\u201d\u2019]?\s+')


def split_sentences(par):
    s = par
    for ab in ABBREV:
        s = s.replace(ab, ab.replace(".", "\u2024"))
    parts = SENT_SPLIT_RE.split(s)
    parts = [p.replace("\u2024", ".").strip() for p in parts]
    return [p for p in parts if p]


WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_TOKEN_RE = re.compile(r"[A-Za-z']+|\d+|[^\sa-zA-Z0-9]")


def words(s):
    return WORD_RE.findall(s.lower())


def tokens(s):
    return _TOKEN_RE.findall(s)


def word_count(s):
    return len(WORD_RE.findall(s))


PUNCT_SET = list(",.;:\u2014-()?!\"\u201c\u201d\u2018\u2019\u2026%")


def punct_profile(s):
    return {p: s.count(p) for p in PUNCT_SET}


def norm_len_bucket(n):
    if n <= 6:
        return 0
    if n <= 12:
        return 1
    if n <= 18:
        return 2
    if n <= 25:
        return 3
    if n <= 35:
        return 4
    return 5


def sentence_len_hist(par):
    h = [0] * 6
    for sent in split_sentences(par):
        h[norm_len_bucket(word_count(sent))] += 1
    return h


def ttr(texts):
    toks, types = 0, set()
    for t in texts:
        w = words(t)
        toks += len(w)
        types.update(w)
    return len(types) / max(toks, 1)
