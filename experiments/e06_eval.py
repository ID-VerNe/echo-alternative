"""E06：首轮端到端评测 —— 4 节点管线 vs 单 prompt 人设基线。
指标：Cosine Delta、句长 JS、标点余弦、风格 kNN 边缘、锚点重叠、事实保留；
外加同模型 pairwise 盲评（双向顺序各一次）。"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, corpus, generate, llm, metrics, stylevec, textutil  # noqa: E402

OUT = config.OUTPUTS / "e06"
OUT.mkdir(parents=True, exist_ok=True)

QUESTIONS = [
    "Is it worth buying an air fryer?",
    "Should I get a robot vacuum cleaner?",
    "What do you think of people who work from home?",
    "Is it acceptable to use AI to write your emails?",
    "My neighbour's leaf blower goes off at 7am every Sunday. What should I do?",
    "Is vaping better than smoking?",
    "Would you live in a smart home?",
    "Are standing desks a con?",
    "What should I do about a colleague who reheats fish in the office microwave?",
    "Is it worth learning to drive a manual car in 2026?",
    "Should I buy a £4 coffee every morning?",
    "Are electric scooters a menace?",
]

anchors_list = anchors.load(config.ARTIFACTS / "anchors.jsonl")
spec = json.loads((config.ARTIFACTS / "style_spec_v1.json").read_text(encoding="utf-8"))

print("loading metrics …")
train = corpus.load_jsonl(config.ARTIFACTS / "train_paras.jsonl")
neg = corpus.load_jsonl(config.ARTIFACTS / "neg_paras.jsonl")
sm = metrics.StyleMetrics(random.sample(train, 1500), random.sample(neg, 800))
rnd = random.Random(21)
bank_c = rnd.sample(train, 400)
bank_n = rnd.sample(neg, 400)
sv = stylevec.StyleVectorizer([p["text"] for p in bank_c + bank_n])
mat = sv.matrix([p["text"] for p in bank_c + bank_n])


def knn_margin(text, k=6):
    q = sv.vec(text)
    sims = mat @ q
    idx = sims.argsort()[-k:][::-1]
    pos = idx[idx < len(bank_c)]
    negi = idx[idx >= len(bank_c)]
    return (float(sims[pos].mean()) if len(pos) else 0.0,
            float(sims[negi].mean()) if len(negi) else 0.0)


def score_text(text, ref_anchors):
    d_ref, d_neg = sm.delta(text)
    kr, kn = knn_margin(text)
    ov, hits = sm.overlap(text, [a["text"] for a in ref_anchors], n=8)
    return {"words": textutil.word_count(text), "n_sent": len(textutil.split_sentences(text)),
            "delta_ref": d_ref, "delta_neg": d_neg, "delta_margin": round(d_neg - d_ref, 4),
            "knn_ref": round(kr, 3), "knn_neg": round(kn, 3), "knn_margin": round(kr - kn, 3),
            "sent_js": sm.sent_js(text), "punct_cos": sm.punct_cos(text),
            "anchor_overlap8": ov, "anchor_hits8": hits}


def run_one(q):
    r = generate.run(q, anchors_list, spec)
    b = generate.baseline(q)
    r["metrics"] = score_text(r["final"], [a for a in anchors_list if a["aid"] in r["anchors"]])
    b_m = score_text(b, [])
    b_m["facts"] = None
    return {"q": q, "pipeline": r, "baseline": {"text": b, "metrics": b_m}}


with ThreadPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(run_one, QUESTIONS))

(OUT / "runs_v1.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- 汇总 ----------
def avg(rows, key):
    return round(sum(r[key] for r in rows) / len(rows), 4)


print("\n%-3s %-42s %6s %6s %7s %7s %7s %7s" % ("#", "question", "Δref", "Δneg", "knnM", "JS", "ov8", "words"))
for i, r in enumerate(results):
    pm = r["pipeline"]["metrics"]
    print("%-3d %-42s %6.3f %6.3f %+7.3f %7.3f %7.3f %7d" %
          (i, r["q"][:42], pm["delta_ref"], pm["delta_neg"], pm["knn_margin"], pm["sent_js"],
           pm["anchor_overlap8"], pm["words"]))

for name, sel in (("pipeline", lambda r: r["pipeline"]["metrics"]),
                  ("baseline", lambda r: r["baseline"]["metrics"])):
    ms = [sel(r) for r in results]
    print(f"\n{name:<10} delta_ref={avg(ms,'delta_ref'):.3f} delta_margin={avg(ms,'delta_margin'):+.3f} "
          f"knn_margin={avg(ms,'knn_margin'):+.3f} sentJS={avg(ms,'sent_js'):.3f} "
          f"punctCos={avg(ms,'punct_cos'):.3f} words={avg(ms,'words'):.0f}")

# ---------- 盲评 ----------
JUDGE_SYS = ("You are a forensic stylometrist. You compare anonymous texts by how they are written, "
             "not by what they say. Ignore topic and facts. Answer with JSON only.")


def judge(q, a, b, refs, order):
    ref_block = "\n\n".join(f"[reference {i+1}] {t}" for i, t in enumerate(refs))
    user = (f"REFERENCE EXCERPTS by the target author:\n\n{ref_block}\n\n"
            f"QUESTION: {q}\n\nTEXT A:\n{a}\n\nTEXT B:\n{b}\n\n"
            "Which text was more likely written by the same person as the references? "
            'Answer exactly: {"winner":"A" or "B","confidence":1-5,"reason":"<15 words>"}')
    raw = llm.chat([{"role": "system", "content": JUDGE_SYS},
                    {"role": "user", "content": user}], temperature=0.1, max_tokens=1200)
    j = llm.parse_json(raw)
    winner = j.get("winner", "A")
    who = ("pipeline" if winner == "A" else "baseline") if order == 0 else ("baseline" if winner == "A" else "pipeline")
    return {"order": order, "winner": who, "conf": j.get("confidence"), "reason": j.get("reason")}


def judge_one(r):
    refs = [a["text"][:700] for a in anchors_list if a["aid"] in r["pipeline"]["anchors"]][:3]
    p, b = r["pipeline"]["final"], r["baseline"]["text"]
    j0 = judge(r["q"], p, b, refs, 0)
    j1 = judge(r["q"], b, p, refs, 1)
    return {"q": r["q"], "j0": j0, "j1": j1}


with ThreadPoolExecutor(max_workers=4) as ex:
    verdicts = list(ex.map(judge_one, results))

win = sum(1 for v in verdicts if v["j0"]["winner"] == v["j1"]["winner"] == "pipeline")
lose = sum(1 for v in verdicts if v["j0"]["winner"] == v["j1"]["winner"] == "baseline")
split = len(verdicts) - win - lose
print(f"\n盲评（双向一致）: pipeline {win} / baseline {lose} / 分歧 {split}  (n={len(verdicts)})")
for v in verdicts:
    print(f"  {v['q'][:40]:<42} {v['j0']['winner']:<9} {v['j1']['winner']:<9} | {v['j0']['reason']}")

(OUT / "judge_v1.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== 样例（管线输出）===")
for r in results[:3]:
    print(f"\n--- {r['q']}\n{r['pipeline']['final'][:900]}")
