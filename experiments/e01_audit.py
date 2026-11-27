"""E01：语料体检 —— 每本保留/丢弃比例、段落量、清洗抽样。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config, corpus, textutil  # noqa: E402

rows = []
for d in sorted(config.CORPUS.iterdir()):
    if not d.is_dir():
        continue
    p = corpus.md_path(d)
    if not p:
        continue
    raw = p.read_text(encoding="utf-8", errors="ignore")
    all_par = textutil.split_paragraphs(raw)
    kept = corpus.load_book(d.name)
    rows.append((d.name, len(all_par), len(kept), sum(x["wc"] for x in kept)))

rows.sort(key=lambda r: -r[3])
print(f"{'book':<62} {'raw':>5} {'kept':>5} {'words':>8}")
tot_raw = tot_kept = tot_words = 0
for name, a, b, w in rows:
    tot_raw += a; tot_kept += b; tot_words += w
    print(f"{name[:60]:<62} {a:>5} {b:>5} {w:>8}")
print(f"{'TOTAL':<62} {tot_raw:>5} {tot_kept:>5} {tot_words:>8}")

cl = [r for r in rows if r[0] in config.TRAIN_BOOKS]
print("\n克拉克森训练集:", len(cl), "本, 段落", sum(r[2] for r in cl), "词", sum(r[3] for r in cl))

# 抽样
paras = corpus.load_all(config.TRAIN_BOOKS[:3])
print("\n=== 清洗后抽样 ===")
for p in paras[:3]:
    print(f"\n[{p['book'][:20]} | {p['genre']} | {p['wc']}w | {p['n_sent']}s]")
    print(p["text"][:600])

paras = corpus.load_all(config.TRAIN_BOOKS)
corpus.dump(paras, config.ARTIFACTS / "train_paras.jsonl")
neg = corpus.load_all(config.NEG_BOOKS, seen=set())
corpus.dump(neg, config.ARTIFACTS / "neg_paras.jsonl")
held = corpus.load_all(config.HELDOUT_BOOKS)
corpus.dump(held, config.ARTIFACTS / "heldout_paras.jsonl")
print("\n已写出 train/neg/heldout jsonl")
