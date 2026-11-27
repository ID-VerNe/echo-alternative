"""E04：构建锚点库并检查 intent 分布与抽样质量。"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, corpus  # noqa: E402

paras = corpus.load_jsonl(config.ARTIFACTS / "train_paras.jsonl")
anc = anchors.build_anchors(paras)
anchors.stats(anc)
anchors.dump(anc, config.ARTIFACTS / "anchors.jsonl")

print("\n=== 各 intent 强度分布 ===")
for it in anchors.INTENTS:
    sub = [p for p in anc if p["intent"] == it]
    print(f"{it:<10} n={len(sub):<4} intensity={sum(p['intensity'] for p in sub)/max(len(sub),1):.2f} "
          f"avg_wc={sum(p['wc'] for p in sub)/max(len(sub),1):.0f}")

print("\n=== 每条 intent 抽样一段 ===")
for it in anchors.INTENTS:
    p = [x for x in anc if x["intent"] == it][0]
    print(f"\n[{it} | {p['book'][:24]} | {p['wc']}w]\n{p['text'][:500]}")
