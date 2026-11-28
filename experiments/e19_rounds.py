"""E19b：同一篇的初稿（round 1）vs 定稿（final），双向盲评 + 同指标。
回答"多轮迭代到底改没改好"，把 agent 内部增益和 agent vs pipeline 的外部增益分开。"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, corpus, llm, metrics, stylevec, textutil  # noqa: E402

O = config.OUTPUTS / "e19_agent"
runs = json.loads((O / "runs_agent.json").read_text(encoding="utf-8"))

anchors_list = anchors.load(config.ARTIFACTS / "anchors.jsonl")
train = corpus.load_jsonl(config.ARTIFACTS / "train_paras.jsonl")
neg = corpus.load_jsonl(config.ARTIFACTS / "neg_paras.jsonl")
sm = metrics.StyleMetrics(random.sample(train, 1500), random.sample(neg, 800))


def score(text):
    d_ref, d_neg = sm.delta(text)
    return {"words": textutil.word_count(text), "delta_ref": d_ref,
            "delta_margin": round(d_neg - d_ref, 4), "sent_js": sm.sent_js(text),
            "punct_cos": sm.punct_cos(text)}


sets = {"round1": [r["rounds"][0]["text"] for r in runs],
        "final": [r["final"] for r in runs]}
ms = {k: [score(t) for t in v] for k, v in sets.items()}
avg = lambda rows, k: round(sum(r[k] for r in rows) / len(rows), 4)
print(f"{'方案':<8}{'Δref↓':>7}{'Δmargin↑':>9}{'JS↓':>7}{'punct↑':>8}{'words':>7}")
for k in sets:
    print(f"{k:<8}{avg(ms[k],'delta_ref'):>7.3f}{avg(ms[k],'delta_margin'):>+9.3f}"
          f"{avg(ms[k],'sent_js'):>7.3f}{avg(ms[k],'punct_cos'):>8.3f}{avg(ms[k],'words'):>7.0f}")

JSYS = ("You are a forensic stylometrist. You compare anonymous texts by how they are written, "
        "not by what they say. Ignore topic and facts. Answer with JSON only.")

refs = [a["text"][:700] for a in anchors_list[:3]]


def judge(q, a, b, order):
    ref_block = "\n\n".join(f"[reference {i+1}] {t}" for i, t in enumerate(refs))
    user = (f"REFERENCE EXCERPTS by the target author:\n\n{ref_block}\n\n"
            f"QUESTION: {q}\n\nTEXT A:\n{a}\n\nTEXT B:\n{b}\n\n"
            "Which text was more likely written by the same person as the references? "
            'Answer exactly: {"winner":"A" or "B","confidence":1-5,"reason":"<15 words>"}')
    raw = llm.chat([{"role": "system", "content": JSYS},
                    {"role": "user", "content": user}], temperature=0.1, max_tokens=1200)
    j = llm.parse_json(raw)
    slot = "A" if j.get("winner", "A") == "A" else "B"
    return {"order": order, "slot": slot, "confidence": j.get("confidence"), "reason": j.get("reason")}


pipe = json.loads((config.OUTPUTS / "e11_v4_s12" / "runs_v4.json").read_text(encoding="utf-8"))
pipe_by_q = {r["q"]: r["cands"][r["chosen"]]["text"] for r in pipe}
base = json.loads((config.OUTPUTS / "base_cache.json").read_text(encoding="utf-8"))


def duel(q, a, b, na, nb):
    j0 = judge(q, a, b, 0)
    j1 = judge(q, b, a, 1)
    j0["winner_label"] = na if j0["slot"] == "A" else nb
    j1["winner_label"] = nb if j1["slot"] == "A" else na
    return {"q": q, "j0": j0, "j1": j1}


def run_pair(label, na, geta, nb, getb):
    with ThreadPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(lambda r: duel(r["q"], geta(r), getb(r), na, nb), runs))
    win = {na: 0, nb: 0, "split": 0}
    for r in res:
        if r["j0"]["winner_label"] == r["j1"]["winner_label"]:
            win[r["j0"]["winner_label"]] += 1
        else:
            win["split"] += 1
    print(f"{label:<20} {na} {win[na]} / {nb} {win[nb]} / 分歧 {win['split']}")
    return {"label": label, "win": win, "duels": res}


print("\n盲评矩阵（各双向，n=12；只统计两向一致的判胜）")
allres = {}
allres["draft vs final"] = run_pair("draft vs final", "draft", lambda r: r["rounds"][0]["text"],
                                    "final", lambda r: r["final"])
allres["final vs pipeline"] = run_pair("final vs pipeline", "agent", lambda r: r["final"],
                                      "pipeline", lambda r: pipe_by_q[r["q"]])
allres["final vs baseline"] = run_pair("final vs baseline", "agent", lambda r: r["final"],
                                       "baseline", lambda r: base[r["q"]]["text"])
allres["draft vs pipeline"] = run_pair("draft vs pipeline", "draft", lambda r: r["rounds"][0]["text"],
                                       "pipeline", lambda r: pipe_by_q[r["q"]])
allres["draft vs baseline"] = run_pair("draft vs baseline", "draft", lambda r: r["rounds"][0]["text"],
                                       "baseline", lambda r: base[r["q"]]["text"])

(O / "rounds_matrix.json").write_text(
    json.dumps({"metrics": ms, "matrix": allres}, ensure_ascii=False, indent=2), encoding="utf-8")
print("saved e19_agent/rounds_matrix.json")
