"""E02：硬统计画像 + 指标 sanity check（克拉克森 vs 梅 vs 通用 AI）。"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config, corpus, llm, metrics, stats, textutil  # noqa: E402

random.seed(7)
train = corpus.load_jsonl(config.ARTIFACTS / "train_paras.jsonl")
neg = corpus.load_jsonl(config.ARTIFACTS / "neg_paras.jsonl")
print("train", len(train), "neg", len(neg))

stats.show(stats.profile(train), "Clarkson 全量")
may = [p for p in neg if p["book"].startswith("詹姆斯·梅")]
stats.show(stats.profile(may), f"James May ({len(may)} 段)")

qs = ["Is it worth buying an air fryer?",
      "What do you think about people who work from home?",
      "Should I switch to an electric car?"]
ai_texts = [llm.chat([{"role": "user", "content": q + " Answer in about 150 words."}],
                     temperature=0.7, max_tokens=2600) for q in qs]
ai_paras = [{"text": t, "wc": textutil.word_count(t), "punct": textutil.punct_profile(t),
             "hist": textutil.sentence_len_hist(t)} for t in ai_texts]
stats.show(stats.profile(ai_paras), "通用 AI（同模型）")

sm = metrics.StyleMetrics(random.sample(train, 1500), random.sample(neg, 800))


def sample_scores(paras, name, k=100):
    rows = []
    for p in random.sample(paras, min(k, len(paras))):
        d_ref, d_neg = sm.delta(p["text"])
        c, n = sm.char_margin(p["text"])
        rows.append((d_ref, d_neg, c, n, sm.sent_js(p["text"]), sm.punct_cos(p["text"])))
    avg = [sum(r[i] for r in rows) / len(rows) for i in range(6)]
    print(f"{name:<10} delta_ref={avg[0]:.3f} delta_neg={avg[1]:.3f} | "
          f"knn_ref={avg[2]:.3f} knn_neg={avg[3]:.3f} margin={avg[2]-avg[3]:+.3f} | "
          f"sentJS={avg[4]:.3f} punctCos={avg[5]:.3f}")


sample_scores(train, "Clarkson")
sample_scores([p for p in neg if p["book"].startswith("詹姆斯·梅")], "May")
sample_scores([p for p in neg if not p["book"].startswith("詹姆斯·梅")], "Others")
for i, t in enumerate(ai_texts):
    d_ref, d_neg = sm.delta(t)
    c, n = sm.char_margin(t)
    print(f"AI-{i}: delta_ref={d_ref:.3f} delta_neg={d_neg:.3f} | knn_ref={c:.3f} knn_neg={n:.3f} "
          f"margin={c-n:+.3f} | sentJS={sm.sent_js(t):.3f} punctCos={sm.punct_cos(t):.3f}")

print("\n=== AI 样本节选 ===")
print(ai_texts[0][:500])
