"""E19：agent 多轮迭代 vs 单次管线（best-of-3）vs 人设基线。
输入 outputs/e19_agent/runs_agent.json（干净上下文 subagent 产出），
其余两套结果从既有实验读回，指标与盲评协议与之前完全一致。"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, corpus, llm, metrics, stylevec, textutil  # noqa: E402

O = config.OUTPUTS
agent_runs = json.loads((O / "e19_agent" / "runs_agent.json").read_text(encoding="utf-8"))
agent_by_q = {r["q"].strip(): r for r in agent_runs}

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


qs = [r["q"] for r in pipe if r["q"] in agent_by_q]
print(f"可配对的题目 {len(qs)}", flush=True)
sets = {"agent": [agent_by_q[q]["final"] for q in qs],
        "pipeline": [pipe_by_q[q] for q in qs],
        "baseline": [base[q]["text"] for q in qs]}
ms = {k: [score(t) for t in v] for k, v in sets.items()}


def avg(rows, k):
    return round(sum(r[k] for r in rows) / len(rows), 4)


print(f"\n{'方案':<10}{'Δref↓':>7}{'Δmargin↑':>9}{'knn↑':>8}{'JS↓':>7}{'punct↑':>8}{'words':>7}")
for k in sets:
    print(f"{k:<10}{avg(ms[k],'delta_ref'):>7.3f}{avg(ms[k],'delta_margin'):>+9.3f}"
          f"{avg(ms[k],'knn_margin'):>+8.3f}{avg(ms[k],'sent_js'):>7.3f}{avg(ms[k],'punct_cos'):>8.3f}"
          f"{avg(ms[k],'words'):>7.0f}")

rounds = [len(r.get("rounds", [])) for r in agent_runs]
print(f"\nagent 平均轮数 {sum(rounds)/len(rounds):.1f}（分布 {rounds}）")

JSYS = ("You are a forensic stylometrist. You compare anonymous texts by how they are written, "
       "not by what they say. Ignore topic and facts. Answer with JSON only.")


def judge(q, a, b, refs, order):
    ref_block = "\n\n".join(f"[reference {i+1}] {t}" for i, t in enumerate(refs))
    user = (f"REFERENCE EXCERPTS by the target author:\n\n{ref_block}\n\n"
            f"QUESTION: {q}\n\nTEXT A:\n{a}\n\nTEXT B:\n{b}\n\n"
            "Which text was more likely written by the same person as the references? "
            'Answer exactly: {"winner":"A" or "B","confidence":1-5,"reason":"<15 words>"}')
    raw = llm.chat([{"role": "system", "content": JSYS},
                    {"role": "user", "content": user}], temperature=0.1, max_tokens=1200)
    j = llm.parse_json(raw)
    w = j.get("winner", "A")
    who = ("first" if w == "A" else "second") if order == 0 else ("second" if w == "A" else "first")
    return {"order": order, "winner": who, "reason": j.get("reason")}


def duel(q, first, second, label):
    refs = [a["text"][:700] for a in anchors_list[:3]]
    j0 = judge(q, first, second, refs, 0)
    j1 = judge(q, second, first, refs, 1)
    return {"q": q, "label": label, "j0": j0, "j1": j1}


duels = []
for q in qs:
    duels.append(("agent", "pipeline", q))
    duels.append(("agent", "baseline", q))

print(f"\n盲评 {len(duels)} 组（各双向）…", flush=True)


def run_duel(t):
    first, second, q = t
    texts = {"agent": agent_by_q[q]["final"], "pipeline": pipe_by_q[q], "baseline": base[q]["text"]}
    return duel(q, texts[first], texts[second], f"{first} vs {second}")


with ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(run_duel, duels))

summary = {}
for r in res:
    label = r["label"]
    summary.setdefault(label, {"win": 0, "lose": 0, "split": 0})
    if r["j0"]["winner"] == r["j1"]["winner"]:
        if r["j0"]["winner"] == "first":
            summary[label]["win"] += 1
        else:
            summary[label]["lose"] += 1
    else:
        summary[label]["split"] += 1

print()
for label, s in summary.items():
    print(f"{label:<22} 左方 {s['win']} / 右方 {s['lose']} / 分歧 {s['split']}（n={len(qs)}）")

(O / "e19_agent" / "judge_duels.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
(O / "e19_agent" / "metrics.json").write_text(json.dumps(ms, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nsaved e19_agent/{judge_duels,metrics}.json")
