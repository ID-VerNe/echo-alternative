"""E14：可迁移性检验 —— 同一个流程、另一作者（James May）。
离线：同样的清洗 → 锚点 → spec（LLM 自动归纳，无手调）
在线：同样的 1 草稿 + 3 候选 + 参照式 rubric rerank
基线：同样的单条人设 Prompt
唯一不同的只有语料和名字。"""
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, corpus, generate, llm, metrics, spec as spec_mod, stylevec, textutil  # noqa: E402

SPEC_NAME = sys.argv[1] if len(sys.argv) > 1 else "style_spec_may"
OUT = config.OUTPUTS / ("e14_may" if SPEC_NAME == "style_spec_may" else "e14_may_v2")
OUT.mkdir(parents=True, exist_ok=True)

MAY_BOOKS = [b for b in config.NEG_BOOKS if "梅" in b]
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

PERSONA_MAY = """You are James May, the British motoring journalist and TV presenter.
Write in his voice: gentle, precise, engineering-minded, quietly amused, fond of small
mechanical detail and of saying the sensible thing slightly too late. Use plain sentences,
dry understatement and the occasional flight of whimsy. Never rant. Be charming."""

# ---------- 离线：May 的资产包 ----------
may_paras = corpus.load_all(MAY_BOOKS)
corpus.dump(may_paras, config.ARTIFACTS / "may_paras.jsonl")
print(f"May 语料：{len(may_paras)} 段 / {sum(p['wc'] for p in may_paras)} 词", flush=True)

may_anchors = anchors.build_anchors(may_paras, target=400)
anchors.dump(may_anchors, config.ARTIFACTS / "may_anchors.jsonl")
anchors.stats(may_anchors)

spec_path = config.ARTIFACTS / f"{SPEC_NAME}.json"
if spec_path.exists():
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
else:
    spec = spec_mod.build(may_anchors, per_intent=2, seed=3)
    spec_mod.save(spec, spec_path)
print("spec keys:", list(spec)[:8], "| prompt chars:", len(spec["compiled_prompt"]), flush=True)

DEVICES = [d for d in spec.get("signature_devices", [])][:3]
DIRECTIONS = [f"DIRECTION: build this piece around one device, used once: {d}" for d in DEVICES]

# ---------- 指标基座 ----------
train_c = corpus.load_jsonl(config.ARTIFACTS / "train_paras.jsonl")  # 克拉克森 = 负样本
sm = metrics.StyleMetrics(random.sample(may_paras, min(700, len(may_paras))),
                          random.sample(train_c, 700))
rnd = random.Random(21)
bank_m = rnd.sample(may_paras, min(300, len(may_paras)))
bank_c = rnd.sample(train_c, 300)
sv = stylevec.StyleVectorizer([p["text"] for p in bank_m + bank_c])
mat = sv.matrix([p["text"] for p in bank_m + bank_c])


def knn_margin(text, k=6):
    q = sv.vec(text)
    sims = mat @ q
    idx = sims.argsort()[-k:][::-1]
    pos = idx[idx < len(bank_m)]
    neg = idx[idx >= len(bank_m)]
    return (float(sims[pos].mean()) if len(pos) else 0.0,
            float(sims[neg].mean()) if len(neg) else 0.0)


def score_text(text):
    d_ref, d_neg = sm.delta(text)
    kr, kn = knn_margin(text)
    ov, _ = sm.overlap(text, [a["text"] for a in may_anchors], n=8)
    return {"words": textutil.word_count(text), "delta_ref": d_ref, "delta_neg": d_neg,
            "delta_margin": round(d_neg - d_ref, 4), "knn_margin": round(kr - kn, 3),
            "sent_js": sm.sent_js(text), "punct_cos": sm.punct_cos(text), "anchor_overlap8": ov}


RUBRIC_SYS = """You are a stylometrist scoring one anonymous text against excerpts by a known writer.
Score 1-10 on each axis, judging how it is written, never what it says.
Answer JSON only."""

RUBRIC_USER = """REFERENCE EXCERPTS by the target writer:

{refs}

QUESTION THE TEXT ANSWERS: {q}

TEXT:
{text}

Axes:
diction: word choice and register match
rhythm: sentence-length and pacing match
devices: the writer's signature devices, used the way he uses them
voice: sounds like this person, not like someone performing him
restraint: never summarises, never moralises, never over-explains

Answer exactly: {{"diction":n,"rhythm":n,"devices":n,"voice":n,"restraint":n}}"""


def rubric(q, text):
    refs = "\n\n".join(f"[{i+1}] {a['text'][:700]}" for i, a in enumerate(may_anchors[:3]))
    raw = llm.chat([{"role": "system", "content": RUBRIC_SYS},
                    {"role": "user", "content": RUBRIC_USER.format(refs=refs, q=q, text=text)}],
                   temperature=0.0, max_tokens=900)
    j = llm.parse_json(raw)
    j["total"] = sum(float(j.get(k, 0)) for k in ("diction", "rhythm", "devices", "voice", "restraint"))
    return j


def baseline_may(q, min_words=150, max_words=200):
    return llm.chat([{"role": "system", "content": PERSONA_MAY},
                     {"role": "user", "content": f"{q}\n\nWrite {min_words}-{max_words} words."}],
                    temperature=0.85, max_tokens=2500)


def run_one(q):
    t0 = time.time()
    intent = generate.classify(q)
    notes = generate.draft(q, intent)
    cands = []
    for i, d in enumerate(DIRECTIONS):
        txt = generate.stylize(q, notes, may_anchors, spec["compiled_prompt"] + "\n\n" + d)
        r = rubric(q, txt)
        cands.append({"i": i, "text": txt, "rubric": r, "metrics": score_text(txt)})
    best = max(cands, key=lambda c: c["rubric"]["total"])
    b = baseline_may(q)
    print(f"  [done {time.time()-t0:.0f}s] {q[:36]:<38} rubric="
          + " ".join(f"{c['rubric']['total']:>4.0f}" for c in cands) + f" -> {best['i']}", flush=True)
    return {"q": q, "intent": intent, "cands": cands, "chosen": best["i"],
            "baseline": {"text": b, "metrics": score_text(b)}}


print(f"running {len(QUESTIONS)} questions …", flush=True)
results = []
with ThreadPoolExecutor(max_workers=3) as ex:
    futs = [ex.submit(run_one, q) for q in QUESTIONS]
    for f in as_completed(futs):
        results.append(f.result())
        (OUT / "runs_partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
results.sort(key=lambda r: QUESTIONS.index(r["q"]))
(OUT / "runs_may.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def avg(ms, k):
    return round(sum(m[k] for m in ms) / len(ms), 4)


pm = [r["cands"][r["chosen"]]["metrics"] for r in results]
bm = [r["baseline"]["metrics"] for r in results]
print(f"\nMay 管线   delta_ref={avg(pm,'delta_ref'):.3f} delta_margin={avg(pm,'delta_margin'):+.3f} "
      f"knn_margin={avg(pm,'knn_margin'):+.3f} sentJS={avg(pm,'sent_js'):.3f} "
      f"punctCos={avg(pm,'punct_cos'):.3f} words={avg(pm,'words'):.0f}")
print(f"May 基线   delta_ref={avg(bm,'delta_ref'):.3f} delta_margin={avg(bm,'delta_margin'):+.3f} "
      f"knn_margin={avg(bm,'knn_margin'):+.3f} sentJS={avg(bm,'sent_js'):.3f} "
      f"punctCos={avg(bm,'punct_cos'):.3f} words={avg(bm,'words'):.0f}")

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
    w = j.get("winner", "A")
    who = ("pipeline" if w == "A" else "baseline") if order == 0 else ("baseline" if w == "A" else "pipeline")
    return {"order": order, "winner": who, "conf": j.get("confidence"), "reason": j.get("reason")}


def judge_one(r):
    refs = [a["text"][:700] for a in may_anchors[:3]]
    p = r["cands"][r["chosen"]]["text"]
    b = r["baseline"]["text"]
    return {"q": r["q"], "j0": judge(r["q"], p, b, refs, 0), "j1": judge(r["q"], b, p, refs, 1)}


with ThreadPoolExecutor(max_workers=3) as ex:
    verdicts = list(ex.map(judge_one, results))

win = sum(1 for v in verdicts if v["j0"]["winner"] == v["j1"]["winner"] == "pipeline")
lose = sum(1 for v in verdicts if v["j0"]["winner"] == v["j1"]["winner"] == "baseline")
split = len(verdicts) - win - lose
j0p = sum(1 for v in verdicts if v["j0"]["winner"] == "pipeline")
j1p = sum(1 for v in verdicts if v["j1"]["winner"] == "pipeline")
print(f"\n盲评（双向一致）: pipeline {win} / baseline {lose} / 分歧 {split}  (n={len(verdicts)}；"
      f"A位胜率 j0={j0p}/{len(verdicts)} j1={j1p}/{len(verdicts)})")
for v in verdicts:
    print(f"  {v['q'][:36]:<38} {v['j0']['winner']:<9} {v['j1']['winner']:<9} | {v['j0']['reason']}")

(OUT / "judge_may.json").write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== 样例 ===")
for r in results[:2]:
    print(f"\n--- {r['q']}  (pick {r['chosen']})\n{r['cands'][r['chosen']]['text'][:800]}")
