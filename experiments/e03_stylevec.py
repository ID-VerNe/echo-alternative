"""E03：风格向量判别力测试（Clarkson vs 同域其他作者 vs 通用 AI）。"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config, corpus, llm, stylevec  # noqa: E402

random.seed(11)
train = corpus.load_jsonl(config.ARTIFACTS / "train_paras.jsonl")
neg = corpus.load_jsonl(config.ARTIFACTS / "neg_paras.jsonl")

bank_c = random.sample(train, 400)
bank_n = random.sample(neg, 400)
rest_c = [p for p in train if p not in bank_c]
rest_n = [p for p in neg if p not in bank_n]
test_c = random.sample(rest_c, 250)
test_n = random.sample(rest_n, 250)

sv = stylevec.StyleVectorizer([p["text"] for p in bank_c + bank_n])
mat = sv.matrix([p["text"] for p in bank_c + bank_n])
labels = np.array([1] * len(bank_c) + [0] * len(bank_n))


def score(text, k=6):
    q = sv.vec(text)
    sims = mat @ q
    idx = np.argsort(-sims)[:k]
    lab = labels[idx]
    pos = int(lab.sum())
    negc = k - pos
    sr = sims[idx][lab == 1].mean() if pos else 0.0
    sn = sims[idx][lab == 0].mean() if negc else 0.0
    return float(sr), float(sn), pos, negc


def acc(tests, name, want):
    ok = 0
    margins = []
    for p in tests:
        sr, sn, pos, negc = score(p["text"])
        ok += (1 if pos > negc else 0) == want
        margins.append(sr - sn)
    print(f"{name:<24} acc={ok/len(tests):.3f}  mean_margin={np.mean(margins):+.4f}")


acc(test_c, "Clarkson（留出段）", 1)
acc(test_n, "其他作者（留出段）", 0)
acc([p for p in test_n if p["book"].startswith("詹姆斯·梅")], "James May", 0)

qs = ["Is it worth buying an air fryer?",
      "What do you think about people who work from home?",
      "Should I switch to an electric car?",
      "Is breakfast the most important meal of the day?"]
ai = [llm.chat([{"role": "user", "content": q + " Answer in about 150 words."}],
               max_tokens=2600) for q in qs]
print("\n通用 AI：")
for i, t in enumerate(ai):
    sr, sn, pos, negc = score(t)
    print(f"AI-{i}: knn_ref={sr:.3f} knn_neg={sn:.3f} margin={sr-sn:+.3f} "
          f"(top6: {pos} 克拉克森 / {negc} 其他)")

print("\n拼接长文本（每 3 段拼一段，各 20 组）：")
for tests, name, want in ((test_c, "Clarkson", 1), (test_n, "Others", 0)):
    ok = 0
    for i in range(0, 60, 3):
        t = " ".join(p["text"] for p in tests[i:i + 3])
        sr, sn, pos, negc = score(t)
        ok += (1 if pos > negc else 0) == want
    print(f"  {name}: {ok}/20")
