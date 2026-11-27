"""E18：把"裁判判词"喂回编译 agent（而不是用自己的 rubric 当目标）。
输入：E16b 的段落级盲评记录（调优版输在哪、裁判的原话），当前 spec；
输出：新 spec；然后在没参与过任何调优的 10 条留出段落上验收。"""
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors as anc_mod, config, corpus, llm  # noqa: E402

OUT = config.OUTPUTS / "e18_judge_loop"
OUT.mkdir(parents=True, exist_ok=True)
TUNED = config.ARTIFACTS / "style_spec_may_tuned.json"
BEST = config.ARTIFACTS / "style_spec_may_tuned2.json"

rows = json.loads((config.OUTPUTS / "e17_holdout" / "judge.json").read_text(encoding="utf-8"))
spec = json.loads(TUNED.read_text(encoding="utf-8"))
untuned = json.loads((config.ARTIFACTS / "style_spec_may_v2.json").read_text(encoding="utf-8"))

# 裁判认为"调优版不如未调优"的那些条（双向里至少一次输了）
FIND_MIN = [r for r in rows if r["j0"]["winner"] == "untuned" or r["j1"]["winner"] == "untuned"]

cases = []
for i, r in enumerate(FIND_MIN[:5]):
    reasons = "; ".join(x["reason"] for x in (r["j0"], r["j1"]) if x and x.get("winner") == "untuned")
    cases.append(f"--- case {i+1}\nTARGET:\n{r['target'][:600]}\n\n"
                 f"YOUR TUNED VERSION (lost):\n{r['tuned'][:600]}\n\n"
                 f"THE UNTUNED VERSION THAT WON:\n{r['untuned'][:600]}\n\n"
                 f"JUDGE'S REASONS:\n{reasons}")

AGENT_SYS = """You are the compiler agent for a style asset. A blind judge — not your own rubric —
compared your tuned version against the untuned version and preferred the untuned one.

Your rubric scores went UP while the judge's preference went DOWN. That means your rules are
optimising something measurable instead of something audible.

Read the judge's reasons and the two versions. Diagnose which rules in your prompt caused the loss.
Then rewrite the compiled_prompt so that the qualities the judge praised in the untuned version
are produced deliberately, without reverting to the untuned prompt.

Constraints: 200-260 words, no percentage tables, at most three numbers, executable without a calculator.
Return STRICT JSON: {"diagnosis":"2-3 sentences","changes":["3-6 rules"],"compiled_prompt":"..."}"""

USER = """CURRENT compiled_prompt (yours, the loser):
{prompt}

LOSS CASES:
{cases}

Write the JSON."""

raw = llm.chat([{"role": "system", "content": AGENT_SYS},
                {"role": "user", "content": USER.format(prompt=spec["compiled_prompt"],
                                                        cases="\n\n".join(cases))}],
               temperature=0.3, max_tokens=3000)
patch = llm.parse_json(raw)
print("诊断：", patch.get("diagnosis"), flush=True)
for c in patch.get("changes", []):
    print(" -", c, flush=True)

new = json.loads(json.dumps(spec))
new["compiled_prompt"] = patch["compiled_prompt"]
new.setdefault("_tune_history", []).append({"round": "judge_loop", "diagnosis": patch.get("diagnosis"),
                                            "changes": patch.get("changes")})
BEST.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
print("new prompt words:", len(patch["compiled_prompt"].split()), flush=True)

# ---------- 在没参与过调优的留出段落上验收 ----------
def bucket(t):
    return int(hashlib.md5(t[:200].encode()).hexdigest()[:6], 16) % 5


paras = corpus.load_jsonl(config.ARTIFACTS / "may_paras.jsonl")
held = [p for p in paras if bucket(p["text"]) == 0][20:30]
train = [p for p in paras if bucket(p["text"]) != 0]
anchors = anc_mod.build_anchors(train, target=400)
refs = "\n\n".join(f"[ref] {a['text'][:700]}" for a in anchors[:3])

BRIEF_SYS = """You are a fact extractor. Return the CONTENT of the paragraph as a neutral brief:
STANCE, CLAIMS (3-5), SPECIFICS (2-4). No metaphor, no rhetorical questions, no style. Max 120 words."""
TASK = ("=== STYLE REFERENCES (never reuse their facts) ===\n" + refs +
        "\n\n=== BRIEF ===\n{b}\n\n=== TASK ===\nMake this argument in 2-3 paragraphs, 170-230 words. "
        "No headings, no bullets, no markdown.")
JSYS = ("You are a forensic stylometrist comparing two texts against one target author. "
        "Judge how they are written. Answer JSON only.")


def retry(fn, *a, **k):
    for i in range(3):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            print(f"   [retry {i+1}] {str(e)[:60]}", flush=True)
            time.sleep(6)
    raise RuntimeError("failed")


def gen(prompt, b):
    return llm.chat([{"role": "system", "content": prompt},
                     {"role": "user", "content": TASK.format(b=b)}], temperature=0.85, max_tokens=2500)


def judge(target, a, b, order):
    user = (f"TARGET PARAGRAPH by the author:\n{target[:900]}\n\nTEXT A:\n{a}\n\nTEXT B:\n{b}\n\n"
            "Which text is closer to how the target author writes? "
            'Answer exactly: {"winner":"A" or "B","reason":"<12 words>"}')
    j = llm.parse_json(retry(llm.chat, [{"role": "system", "content": JSYS},
                                        {"role": "user", "content": user}], temperature=0.1, max_tokens=800))
    w = j.get("winner", "A")
    who = ("agent" if w == "A" else "untuned") if order == 0 else ("untuned" if w == "A" else "agent")
    return {"winner": who, "reason": j.get("reason")}


def one(p):
    b = retry(llm.chat, [{"role": "system", "content": BRIEF_SYS},
                         {"role": "user", "content": f"PARAGRAPH:\n{p['text']}"}], temperature=0.0, max_tokens=900)
    a = retry(gen, new["compiled_prompt"], b)
    u = retry(gen, untuned["compiled_prompt"], b)
    j0 = retry(judge, p["text"], a, u, 0)
    j1 = retry(judge, p["text"], u, a, 1)
    print(f"  {p['text'][:38]:<42} {j0['winner']:<8} {j1['winner']:<8} | {j0['reason']}", flush=True)
    return {"target": p["text"][:400], "agent": a, "untuned": u, "j0": j0, "j1": j1}


with ThreadPoolExecutor(max_workers=2) as ex:
    out = list(ex.map(one, held))

w = sum(1 for r in out if r["j0"]["winner"] == r["j1"]["winner"] == "agent")
l = sum(1 for r in out if r["j0"]["winner"] == r["j1"]["winner"] == "untuned")
s = len(out) - w - l
print(f"\n新留出验收：agent 版 {w} / 未调优 {l} / 分歧 {s}（n={len(out)}；"
      f"A 位胜率 {sum(1 for r in out if r['j0']['winner']=='agent')}/{len(out)}）")
(OUT / "judge.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
