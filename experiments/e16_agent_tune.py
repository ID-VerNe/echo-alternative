"""E16：编译 agent —— 读语料、诊断、改资产、复评的自动循环（通用，不针对某作者）。

协议：
1) 段落级留出（hash 分桶 20%），锚点与 spec 只用训练桶，避免泄漏；
2) 留出段落 → 中性 brief（只留事实）→ 用当前 spec 重写成段 → rubric 打分（对照原段）；
3) agent 读最差的几条 + 各轴得分，输出新的 compiled_prompt 与改动理由；
4) 复评，保留留出均分最高的一版。

用法：python experiments/e16_agent_tune.py [spec_in] [spec_out] [rounds] [n_cases]
"""
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors as anc_mod, config, corpus, llm, spec as spec_mod  # noqa: E402

SPEC_IN = sys.argv[1] if len(sys.argv) > 1 else "style_spec_may_v2"
SPEC_OUT = sys.argv[2] if len(sys.argv) > 2 else "style_spec_may_tuned"
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NCASES = int(sys.argv[4]) if len(sys.argv) > 4 else 10

OUT = config.OUTPUTS / "e16_tune"
OUT.mkdir(parents=True, exist_ok=True)

MAY_BOOKS = [b for b in config.NEG_BOOKS if "梅" in b]


def bucket(text):
    return int(hashlib.md5(text[:200].encode()).hexdigest()[:6], 16) % 5


def split_paras():
    paras = corpus.load_jsonl(config.ARTIFACTS / "may_paras.jsonl")
    held = [p for p in paras if bucket(p["text"]) == 0]
    train = [p for p in paras if bucket(p["text"]) != 0]
    rnd = random.Random(11)
    return rnd.sample(held, min(len(held), 80)), train


held_paras, train_paras = split_paras()
print(f"留出 {len(held_paras)} 段 / 训练 {len(train_paras)} 段", flush=True)

train_anchors = anc_mod.build_anchors(train_paras, target=400)
print("训练桶锚点:", len(train_anchors), flush=True)

spec = json.loads((config.ARTIFACTS / f"{SPEC_IN}.json").read_text(encoding="utf-8")) \
    if (config.ARTIFACTS / f"{SPEC_IN}.json").exists() else None
if spec is None:
    spec = spec_mod.build(train_anchors, per_intent=3, seed=3)
    spec_mod.save(spec, config.ARTIFACTS / f"{SPEC_IN}.json")
print(f"起始 spec：{SPEC_IN}，compiled_prompt {len(spec['compiled_prompt'].split())} 词", flush=True)

BRIEF_SYS = """You are a fact extractor. You receive a paragraph from a book.
Return its CONTENT as a neutral brief, with no trace of the original style.
Output plain text:
STANCE: one line.
CLAIMS: 3-5 bullets.
SPECIFICS: 2-4 concrete details (objects, prices, times, places).
Rules: no metaphor, no rhetorical questions, no evaluative adjectives, no echo of the source's voice.
Max 120 words."""

BRIEF_USER = "PARAGRAPH:\n{text}\n\nReturn the neutral brief."

RUBRIC_SYS = """You are a stylometrist. You compare one generated text against a target paragraph by the
same author and two other excerpts. Judge how it is written, never what it says. Answer JSON only."""

RUBRIC_USER = """TARGET PARAGRAPH (the real author, for reference):
{target}

OTHER EXCERPTS by the same author:
{refs}

GENERATED TEXT:
{text}

Score 1-10:
diction, rhythm, devices (his signature moves used the way he uses them),
voice (sounds like this person, not a performance), restraint (no summary, no moral).
Answer exactly: {{"diction":n,"rhythm":n,"devices":n,"voice":n,"restraint":n}}"""

DIAG_SYS = """You are the compiler agent for a style asset. You have read the whole corpus, and you
are now looking at failures of your own compiled prompt. You do not defend your work.

You receive: the current compiled_prompt, the failure cases with per-axis scores, and the real
target paragraphs they failed to match.

Diagnose what the prompt is NOT doing — not what it says badly. Prefer fixes that change the
mechanism (what the writer does first, how the argument turns, how the piece ends) over fixes
that add adjectives. Never add percentage tables or histogram numbers.

Return STRICT JSON:
{"diagnosis": "2-3 sentences naming the weakest axis and its cause",
 "changes": ["3-6 short imperative rules to add or replace"],
 "compiled_prompt": "the full new prompt, 200-260 words, executable without a calculator,
   preserving everything that was already working"}"""

DIAG_USER = """CURRENT compiled_prompt:
{prompt}

FAILURE CASES (worst {n} of {total} on held-out paragraphs):
{cases}

Write the JSON."""


def brief(text):
    return llm.chat([{"role": "system", "content": BRIEF_SYS},
                     {"role": "user", "content": BRIEF_USER.format(text=text)}],
                    temperature=0.0, max_tokens=900)


def write_like(b, prompt, refs):
    user = (f"=== STYLE REFERENCES (not sources, never reuse their facts) ===\n{refs}\n\n"
            f"=== BRIEF ===\n{b}\n\n=== TASK ===\nMake this argument in 2-3 paragraphs, 170-230 words. "
            "No headings, no bullets, no markdown.")
    return llm.chat([{"role": "system", "content": prompt}, {"role": "user", "content": user}],
                    temperature=0.85, max_tokens=2500)


def rubric(target, refs, text):
    raw = llm.chat([{"role": "system", "content": RUBRIC_SYS},
                    {"role": "user", "content": RUBRIC_USER.format(target=target[:900], refs=refs, text=text)}],
                   temperature=0.0, max_tokens=900)
    j = llm.parse_json(raw)
    j["total"] = sum(float(j.get(k, 0)) for k in ("diction", "rhythm", "devices", "voice", "restraint"))
    return j


cases = held_paras[:NCASES]
refs_pool = [a["text"] for a in train_anchors[:6]]


def evaluate(prompt, tag):
    bs = [brief(c["text"]) for c in cases]

    def one(b, c):
        text = write_like(b, prompt, "\n\n".join(f"[ref] {r[:700]}" for r in refs_pool[:3]))
        r = rubric(c["text"], "\n\n".join(f"[ref] {x[:700]}" for x in refs_pool[:2]), text)
        return {"target": c["text"], "gen": text, "rubric": r}

    with ThreadPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(lambda b_c: one(*b_c), zip(bs, cases)))
    mean = sum(r["rubric"]["total"] for r in rows) / len(rows)
    axes = {k: round(sum(r["rubric"].get(k, 0) for r in rows) / len(rows), 2)
            for k in ("diction", "rhythm", "devices", "voice", "restraint")}
    print(f"[{tag}] 留出均分 {mean:.1f}/50  轴 {axes}", flush=True)
    return {"tag": tag, "mean": mean, "axes": axes, "rows": rows}


history = []
best = evaluate(spec["compiled_prompt"], "round0")
best_spec, best_tag = spec, "round0"

for r in range(1, ROUNDS + 1):
    worst = sorted(best["rows"], key=lambda x: x["rubric"]["total"])[:4]
    KEYS = ("diction", "rhythm", "devices", "voice", "restraint")
    blocks = []
    for i, w in enumerate(worst):
        ax = " ".join(f"{k}={w['rubric'].get(k)}" for k in KEYS)
        blocks.append(f"--- case {i+1} (total {w['rubric']['total']:.0f}/50 | {ax})\n"
                      f"TARGET:\n{w['target'][:700]}\n\nGENERATED:\n{w['gen'][:700]}")
    case_block = "\n\n".join(blocks)
    raw = llm.chat([{"role": "system", "content": DIAG_SYS},
                    {"role": "user", "content": DIAG_USER.format(
                        prompt=spec["compiled_prompt"], n=len(worst), total=len(best["rows"]), cases=case_block)}],
                   temperature=0.3, max_tokens=3000)
    patch = llm.parse_json(raw)
    print(f"\n== 第 {r} 轮诊断：{patch.get('diagnosis', '')}", flush=True)
    for c in patch.get("changes", []):
        print("   -", c, flush=True)
    spec_new = json.loads(json.dumps(spec))
    spec_new["compiled_prompt"] = patch["compiled_prompt"]
    spec_new.setdefault("_tune_history", []).append({"round": r, "diagnosis": patch.get("diagnosis"),
                                                   "changes": patch.get("changes")})
    spec = spec_new
    history.append(best)
    best = evaluate(spec["compiled_prompt"], f"round{r}")
    if best["mean"] > max(h["mean"] for h in history):
        best_spec, best_tag = spec, f"round{r}"
    else:
        print(f"   [regress] round{r} {best['mean']:.1f} ≤ 历史最好，保留 {best_tag}", flush=True)

(config.ARTIFACTS / f"{SPEC_OUT}.json").write_text(
    json.dumps(best_spec, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"kept best = {best_tag}（{max(h['mean'] for h in history + [best]):.1f}/50）", flush=True)
(OUT / "history.json").write_text(json.dumps(
    {"rounds": [{"tag": h["tag"], "mean": h["mean"], "axes": h["axes"]} for h in history + [best]],
     "spec_out": SPEC_OUT, "n_cases": NCASES,
     "n_held": len(held_paras), "n_train": len(train_paras)}, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== 曲线 ===")
for h in history + [best]:
    print(f"{h['tag']:<10} {h['mean']:.1f}   {h['axes']}")
print(f"\nwrote artifacts/{SPEC_OUT}.json")
