"""E11：v4 —— 恢复喜剧密度，并用 rubric critic 做 best-of-3 rerank。
v3 的教训：把"装置"当成装饰品限流后，文本变理性、变正确、变无聊；盲评 0:6 全输。
裁判认的是：荒诞类比、层层加码、punchline、英式语域。那就直接优化这个。"""
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, corpus, generate, llm, metrics, stylevec, textutil  # noqa: E402

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OUT = config.OUTPUTS / f"e11_v4_s{LIMIT}"
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
spec = json.loads((config.ARTIFACTS / "style_spec_v4.json").read_text(encoding="utf-8"))

DIRECTIONS = [
    "DIRECTION: build the piece around one absurd physical comparison, and let it escalate until the last line.",
    "DIRECTION: open on a scene, then answer, then end by undercutting your own answer with something small and specific.",
    "DIRECTION: write it as a mock regulation with a time window and a fine, and keep a straight face until the final sentence.",
]

RUBRIC_SYS = """You are the editor of a British newspaper column, and you have spent twenty years cutting
Clarkson's copy. You are ruthless about one thing: it must be funny on the page, not merely correct.

Score the draft 1-5 on each axis:
absurd: does it contain at least one physically absurd, concrete comparison (an object, an animal, a form, a wet Monday)?
escalation: does the joke build, or does it sit flat?
punchline: does the last sentence land, and does it undercut the argument or the reader?
voice: does it sound like a man talking, not a columnist writing?
british: pounds, miles, dry understatement, no American brand names?
NoSummary: 1 if it ends on a moral lesson or a summary, else 5.

Also list the single worst sentence and say why in eight words.
Answer JSON only: {"absurd":n,"escalation":n,"punchline":n,"voice":n,"british":n,"NoSummary":n,"worst":"...","total":n}"""


def rubric(text):
    raw = llm.chat([{"role": "system", "content": RUBRIC_SYS},
                    {"role": "user", "content": text}], temperature=0.0, max_tokens=1500)
    j = llm.parse_json(raw)
    if "total" not in j:
        j["total"] = sum(int(j.get(k, 3)) for k in ("absurd", "escalation", "punchline", "voice", "british", "NoSummary"))
    return j


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


def score_text(text):
    d_ref, d_neg = sm.delta(text)
    kr, kn = knn_margin(text)
    ov, _ = sm.overlap(text, [a["text"] for a in anchors_list], n=8)
    return {"words": textutil.word_count(text), "delta_ref": d_ref, "delta_neg": d_neg,
            "delta_margin": round(d_neg - d_ref, 4), "knn_margin": round(kr - kn, 3),
            "sent_js": sm.sent_js(text), "punct_cos": sm.punct_cos(text), "anchor_overlap8": ov}


def run_one(q):
    intent = generate.classify(q)
    notes = generate.draft(q, intent)
    cands = []
    for i, d in enumerate(DIRECTIONS):
        txt = generate.stylize(q, notes, anchors_list, spec["compiled_prompt"] + "\n\n" + d)
        r = rubric(txt)
        m = score_text(txt)
        cands.append({"i": i, "text": txt, "rubric": r, "metrics": m})
    best = max(cands, key=lambda c: (c["rubric"].get("total", 0), -c["metrics"]["delta_ref"]))
    print(f"  [done] {q[:36]:<38} rubric=" + " ".join(f"{c['rubric'].get('total',0):>3}" for c in cands)
          + f" -> pick {best['i']}", flush=True)
    return {"q": q, "intent": intent, "cands": cands, "chosen": best["i"]}


print(f"running {LIMIT} questions, best-of-{len(DIRECTIONS)} (rubric) …", flush=True)
results = []
with ThreadPoolExecutor(max_workers=3) as ex:
    futs = [ex.submit(run_one, q) for q in QUESTIONS[:LIMIT]]
    for f in as_completed(futs):
        results.append(f.result())
        (OUT / "runs_partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
results.sort(key=lambda r: QUESTIONS.index(r["q"]))
(OUT / "runs_v4.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def avg(rows, key):
    return round(sum(r[key] for r in rows) / len(rows), 4)


chosen = [r["cands"][r["chosen"]]["metrics"] for r in results]
print("\n#   question                                     Δref   Δmargin  knnM      JS     punct  words  rubric")
for i, r in enumerate(results):
    m = r["cands"][r["chosen"]]["metrics"]
    print("%-3d %-42s %6.3f %+7.3f %+6.3f %6.3f %6.3f %5d  %d" %
          (i, r["q"][:42], m["delta_ref"], m["delta_margin"], m["knn_margin"],
           m["sent_js"], m["punct_cos"], m["words"], r["cands"][r["chosen"]]["rubric"].get("total", 0)))
print(f"\nv4  delta_ref={avg(chosen,'delta_ref'):.3f} delta_margin={avg(chosen,'delta_margin'):+.3f} "
      f"knn_margin={avg(chosen,'knn_margin'):+.3f} sentJS={avg(chosen,'sent_js'):.3f} "
      f"punctCos={avg(chosen,'punct_cos'):.3f} words={avg(chosen,'words'):.0f}")
all_c = [c["metrics"] for r in results for c in r["cands"]]
print(f"（对照：全部候选均值 delta_ref={avg(all_c,'delta_ref'):.3f} punctCos={avg(all_c,'punct_cos'):.3f} JS={avg(all_c,'sent_js'):.3f}）")

CACHE_BASE = config.OUTPUTS / "base_cache.json"
base_by_q = json.loads(CACHE_BASE.read_text(encoding="utf-8")) if CACHE_BASE.exists() else {}
for b in json.loads((config.OUTPUTS / "e06_v1_s6" / "runs_v1.json").read_text(encoding="utf-8")):
    base_by_q.setdefault(b["q"], b["baseline"])
missing = [r["q"] for r in results if r["q"] not in base_by_q]
if missing:
    print(f"生成 {len(missing)} 条缺失基线 …", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        for q, txt in zip(missing, ex.map(generate.baseline, missing)):
            base_by_q[q] = {"text": txt, "metrics": score_text(txt)}
    CACHE_BASE.write_text(json.dumps(base_by_q, ensure_ascii=False, indent=2), encoding="utf-8")

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

(OUT / "judge_v4.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== 样例 ===")
for r in results[:3]:
    print(f"\n--- {r['q']}  (pick {r['chosen']})\n{r['cands'][r['chosen']]['text'][:900]}")
