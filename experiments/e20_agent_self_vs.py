"""E20：agent 自己写（不调外部 LLM）vs 管线 best-of-3 vs 人设基线。
另外拆一层：同一 agent 的初稿 vs 定稿（多轮迭代到底有没有用）。"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, corpus, llm, metrics, stylevec, textutil  # noqa: E402

O = config.OUTPUTS
runs = json.loads((O / "e20_self" / "runs_self.json").read_text(encoding="utf-8"))
self_by_q = {r["q"].strip(): r for r in runs}

pipe = json.loads((O / "e11_v4_s12" / "runs_v4.json").read_text(encoding="utf-8"))
pipe_by_q = {r["q"]: r["cands"][r["chosen"]]["text"] for r in pipe}
base = json.loads((O / "base_cache.json").read_text(encoding="utf-8"))

anchors_list = anchors.load(config.ARTIFACTS / "anchors.jsonl")
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


def score(text):
    d_ref, d_neg = sm.delta(text)
    kr, kn = knn_margin(text)
    ov, _ = sm.overlap(text, [a["text"] for a in anchors_list], n=8)
    return {"words": textutil.word_count(text), "delta_ref": d_ref, "delta_neg": d_neg,
            "delta_margin": round(d_neg - d_ref, 4), "knn_margin": round(kr - kn, 3),
            "sent_js": sm.sent_js(text), "punct_cos": sm.punct_cos(text), "anchor_overlap8": ov}


qs = [r["q"] for r in pipe if r["q"] in self_by_q]
print(f"可配对题目 {len(qs)}；agent 轮数 "
      f"{[len(self_by_q[q].get('rounds', [])) for q in qs]}", flush=True)

sets = {
    "self_final": [self_by_q[q]["final"] for q in qs],
    "self_first": [self_by_q[q]["rounds"][0]["text"] for q in qs],
    "pipeline": [pipe_by_q[q] for q in qs],
    "baseline": [base[q]["text"] for q in qs],
}
ms = {k: [score(t) for t in v] for k, v in sets.items()}


def avg(rows, k):
    return round(sum(r[k] for r in rows) / len(rows), 4)


print(f"\n{'方案':<12}{'Δref↓':>7}{'Δmargin↑':>9}{'knn↑':>8}{'JS↓':>7}{'punct↑':>8}{'words':>7}")
for k in sets:
    print(f"{k:<12}{avg(ms[k],'delta_ref'):>7.3f}{avg(ms[k],'delta_margin'):>+9.3f}"
          f"{avg(ms[k],'knn_margin'):>+8.3f}{avg(ms[k],'sent_js'):>7.3f}"
          f"{avg(ms[k],'punct_cos'):>8.3f}{avg(ms[k],'words'):>7.0f}")

JSYS = ("You are a forensic stylometrist. You compare anonymous texts by how they are written, "
       "not by what they say. Ignore topic and facts. Answer with JSON only.")


def judge(q, a, b, refs, order):
    ref_block = "\n\n".join(f"[reference {i+1}] {t}" for i, t in enumerate(refs))
    user = (f"REFERENCE EXCERPTS by the target author:\n\n{ref_block}\n\n"
            f"QUESTION: {q}\n\nTEXT A:\n{a}\n\nTEXT B:\n{b}\n\n"
            "Which text was more likely written by the same person as the references? "
            'Answer exactly: {"winner":"A" or "B","confidence":1-5,"reason":"<15 words>"}')
    j = llm.parse_json(llm.chat([{"role": "system", "content": JSYS},
                                 {"role": "user", "content": user}], temperature=0.1, max_tokens=1200))
    w = j.get("winner", "A")
    return ("first" if w == "A" else "second") if order == 0 else ("second" if w == "A" else "first")


DUELS = [("self_final", "pipeline"), ("self_final", "baseline"),
         ("self_first", "self_final"), ("self_first", "pipeline")]


def run_duel(t):
    q, (a_key, b_key) = t
    refs = [x["text"][:700] for x in anchors_list[:3]]
    j0 = judge(q, sets[a_key][qs.index(q)], sets[b_key][qs.index(q)], refs, 0)
    j1 = judge(q, sets[b_key][qs.index(q)], sets[a_key][qs.index(q)], refs, 1)
    return {"q": q, "label": f"{a_key} vs {b_key}", "j0": j0, "j1": j1}


tasks = [(q, d) for d in DUELS for q in qs]
print(f"\n盲评 {len(tasks)} 组（各双向）…", flush=True)
with ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(run_duel, tasks))

summary = {}
for r in res:
    s = summary.setdefault(r["label"], {"win": 0, "lose": 0, "split": 0})
    if r["j0"] == r["j1"]:
        if r["j0"] == "first":
            s["win"] += 1
        else:
            s["lose"] += 1
    else:
        s["split"] += 1

print()
for label, s in summary.items():
    print(f"{label:<26} 左方 {s['win']} / 右方 {s['lose']} / 分歧 {s['split']}（n={len(qs)}）")
for label, s in summary.items():
    a, b = label.split(" vs ")
    pos = sum(1 for r in res if r["label"] == label and r["j0"] == "first")
    print(f"  {label}: A 位先选左方 {pos}/{len(qs)}")

(O / "e20_self" / "judge_duels.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
(O / "e20_self" / "metrics.json").write_text(json.dumps(ms, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nsaved e20_self/{judge_duels,metrics}.json")
