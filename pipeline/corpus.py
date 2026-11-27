"""语料加载与清洗：md 目录 → 干净段落（带书目/文体/统计元数据）。"""
import hashlib
import json
import re
from pathlib import Path

from . import config, textutil


def genre_of(name):
    if "车评" in name:
        return "car"
    if "时评" in name:
        return "column"
    if "农场" in name:
        return "farm"
    return "other"


def md_path(book_dir):
    files = list(Path(book_dir).glob("*.md"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size)


def fingerprint(text):
    norm = re.sub(r"[^a-z ]", "", text.lower())
    norm = re.sub(r"\s+", " ", norm).strip()[:120]
    return hashlib.md5(norm.encode()).hexdigest()


def load_book(book_name, min_words=70, min_sents=4):
    p = md_path(config.CORPUS / book_name)
    if p is None:
        return []
    raw = p.read_text(encoding="utf-8", errors="ignore")
    paras = textutil.split_paragraphs(raw)
    genre = genre_of(book_name)
    out = []
    for i, t in enumerate(paras):
        wc = textutil.word_count(t)
        if wc < min_words:
            continue
        sents = textutil.split_sentences(t)
        if len(sents) < min_sents:
            continue
        # 句子完整性：句末标点比例
        term = sum(1 for s in sents if s.endswith((".", "!", "?", "\u201d", "\u2019")) or s[-1] in ".!?")
        if term / len(sents) < 0.8:
            continue
        # 段首必须像句子开头（PDF 分页会把段落拦腰切断）
        head = t.lstrip("\"'\u201c\u2018 ")
        if not head[:1].isupper() and not head[:1].isdigit():
            continue
        out.append({
            "book": book_name,
            "genre": genre,
            "idx": i,
            "text": t,
            "wc": wc,
            "n_sent": len(sents),
            "hist": textutil.sentence_len_hist(t),
            "punct": textutil.punct_profile(t),
        })
    return out


def load_all(books, dedupe=True, seen=None):
    """按书目加载并跨书去重（世界系列是专栏合集，存在重复篇目）。"""
    seen = seen if seen is not None else set()
    out, seen_local = [], set()
    for b in books:
        for p in load_book(b):
            fp = fingerprint(p["text"])
            if dedupe and (fp in seen or fp in seen_local):
                continue
            seen_local.add(fp)
            p["pid"] = f"P{len(out):05d}"
            out.append(p)
    seen.update(seen_local)
    return out


def dump(paras, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in paras:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
