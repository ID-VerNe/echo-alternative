"""E10：v3 —— 多候选采样 + 用自建判别指标 rerank（best-of-3）。
单候选的问题：强模型下输出要么"装置堆叠"要么"被限流后变平"。这里让同一 brief 出 3 个
不同开局的候选，再用 delta / knn / 标点 / 句长 复合分挑最像的那个。"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, corpus, generate, llm, metrics, stylevec, textutil  # noqa: E402

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = config.OUTPUTS / f"e10_v3_s{LIMIT}"
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
spec = json.loads((config.ARTIFACTS / "style_spec_v3.json").read_text(encoding="utf-8"))

DIRECTIONS = [
    "",
    "DIRECTION: open with the answer in four words or fewer, then spend the rest of the piece regretting it.",
    "DIRECTION: open on a scene — someone ordinary doing something ordinary, in a named place. The verdict arrives in the second paragraph.",
]

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
    return {"words": textutil.word_count(text), "delta_ref": d_ref, "delta_neg": d_neg,
            "delta_margin": round(d_neg - d_ref, 4), "knn_margin": round(kr - kn, 3),
            "sent_js": sm.sent_js(text), "punct_cos": sm.punct_cos(text),
            "anchor_overlap8": ov}


def composite(m):
    w = m["words"]
    pen = 0.0 if 150 <= w <= 250 else -0.15
    return ((1.10 - m["delta_ref"]) * 1.0 + m["delta_margin"] * 1.0
            + m["punct_cos"] * 0.8 + (0.15 - m["sent_js"]) * 2.0 + pen)


def run_one(q):
    intent = generate.classify(q)
    notes = generate.draft(q, intent)
    cands = []
    for i, d in enumerate(DIRECTIONS):
        spec_i = spec["compiled_prompt"] + ("\n\n" + d if d else "")
        txt = generate.stylize(q, notes, anchors_list, spec_i)
        m = score_text(txt, anchors_list)
        cands.append({"i": i, "text": txt, "metrics": m, "score": round(composite(m), 4)})
    best = max(cands, key=lambda c: c["score"])
    print(f"  [done] {q[:38]:<40} intent={intent} scores="
          + " ".join(f"{c['score']:+.3f}" for c in cands) + f" -> pick {best['i']}", flush=True)
    return {"q": q, "intent": intent, "cands": cands, "chosen": best["i"]}


print(f"running {LIMIT} questions, best-of-{len(DIRECTIONS)} …", flush=True)
results = []
with ThreadPoolExecutor(max_workers=3) as ex:
    futs = [ex.submit(run_one, q) for q in QUESTIONS[:LIMIT]]
    for f in as_completed(futs):
        results.append(f.result())
        (OUT / "runs_partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
results.sort(key=lambda r: QUESTIONS.index(r["q"]))
(OUT / "runs_v3.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

def avg(rows, key):
    return round(sum(r[key] for r in rows) / len(rows), 4)


chosen = [r["cands"][r["chosen"]]["metrics"] for r in results]
print("\n#   question                                     Δref   Δmargin  knnM      JS     punct  words  pick")
for i, r in enumerate(results):
    m = r["cands"][r["chosen"]]["metrics"]
    print("%-3d %-42s %6.3f %+7.3f %+6.3f %6.3f %6.3f %5d  %d" %
          (i, r["q"][:42], m["delta_ref"], m["delta_margin"], m["knn_margin"],
           m["sent_js"], m["punct_cos"], m["words"], r["chosen"]))
print(f"\nv3 rerank  delta_ref={avg(chosen,'delta_ref'):.3f} delta_margin={avg(chosen,'delta_margin'):+.3f} "
      f"knn_margin={avg(chosen,'knn_margin'):+.3f} sentJS={avg(chosen,'sent_js'):.3f} "
      f"punctCos={avg(chosen,'punct_cos'):.3f} words={avg(chosen,'words'):.0f}")

all_c = [c["metrics"] for r in results for c in r["cands"]]
print(f"（对照：全部候选均值 delta_ref={avg(all_c,'delta_ref'):.3f} punctCos={avg(all_c,'punct_cos'):.3f} "
      f"JS={avg(all_c,'sent_js'):.3f}）")

base = json.loads((config.OUTPUTS / "e06_v1_s6" / "runs_v1.json").read_text(encoding="utf-8"))
base_by_q = {b["q"]: b["baseline"] for b in base}

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
    refs = [a["text"][:700] for a in anchors_list][:3]
    p = r["cands"][r["chosen"]]["text"]
    b = base_by_q[r["q"]]["text"]
    return {"q": r["q"], "j0": judge(r["q"], p, b, refs, 0), "j1": judge(r["q"], b, p, refs, 1)}


with ThreadPoolExecutor(max_workers=3) as ex:
    verdicts = list(ex.map(judge_one, results))

win = sum(1 for v in verdicts if v["j0"]["winner"] == v["j1"]["winner"] == "pipeline")
lose = sum(1 for v in verdicts if v["j0"]["winner"] == v["j1"]["winner"] == "baseline")
split = len(verdicts) - win - lose
print(f"\n盲评（双向一致）: pipeline {win} / baseline {lose} / 分歧 {split}  (n={len(verdicts)})")
for v in verdicts:
    print(f"  {v['q'][:38]:<40} {v['j0']['winner']:<9} {v['j1']['winner']:<9} | {v['j0']['reason']}")

(OUT / "judge_v3.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== 样例 ===")
for r in results[:3]:
    print(f"\n--- {r['q']}  (pick {r['chosen']})\n{r['cands'][r['chosen']]['text'][:900]}")
