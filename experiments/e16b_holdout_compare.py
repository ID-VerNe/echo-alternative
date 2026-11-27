"""E17：段落级留出验收 —— 调优版 vs 未调优版，同一批没参与过调优的留出段落。
生成两端各一次，再让裁判双向盲选（不做分数比较，避免直接复用调优目标）。"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors as anc_mod, config, corpus, llm  # noqa: E402

A_SPEC = sys.argv[1] if len(sys.argv) > 1 else "style_spec_may_v2"
B_SPEC = sys.argv[2] if len(sys.argv) > 2 else "style_spec_may_tuned"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10

OUT = config.OUTPUTS / "e17_holdout"
OUT.mkdir(parents=True, exist_ok=True)


def bucket(text):
    return int(hashlib.md5(text[:200].encode()).hexdigest()[:6], 16) % 5


paras = corpus.load_jsonl(config.ARTIFACTS / "may_paras.jsonl")
held = [p for p in paras if bucket(p["text"]) == 0][10:10 + N]   # 跳过调优用过的前 10 条
train = [p for p in paras if bucket(p["text"]) != 0]
anchors = anc_mod.build_anchors(train, target=400)
refs = "\n\n".join(f"[ref] {a['text'][:700]}" for a in anchors[:3])
spec_a = json.loads((config.ARTIFACTS / f"{A_SPEC}.json").read_text(encoding="utf-8"))["compiled_prompt"]
spec_b = json.loads((config.ARTIFACTS / f"{B_SPEC}.json").read_text(encoding="utf-8"))["compiled_prompt"]

BRIEF_SYS = """You are a fact extractor. Return the CONTENT of the paragraph as a neutral brief:
STANCE, CLAIMS (3-5), SPECIFICS (2-4). No metaphor, no rhetorical questions, no style. Max 120 words."""
TASK = ("=== STYLE REFERENCES (never reuse their facts) ===\n" + refs +
        "\n\n=== BRIEF ===\n{b}\n\n=== TASK ===\nMake this argument in 2-3 paragraphs, 170-230 words. "
        "No headings, no bullets, no markdown.")

JSYS = ("You are a forensic stylometrist comparing two texts against one target author. "
        "Judge how they are written. Answer JSON only.")


def gen(prompt, b):
    return llm.chat([{"role": "system", "content": prompt},
                     {"role": "user", "content": TASK.format(b=b)}],
                    temperature=0.85, max_tokens=2500)


def judge(target, a, b, order):
    user = (f"TARGET PARAGRAPH by the author:\n{target[:900]}\n\nTEXT A:\n{a}\n\nTEXT B:\n{b}\n\n"
            "Which text is closer to how the target author writes? "
            'Answer exactly: {"winner":"A" or "B","reason":"<12 words>"}')
    j = llm.parse_json(llm.chat([{"role": "system", "content": JSYS},
                                 {"role": "user", "content": user}], temperature=0.1, max_tokens=800))
    w = j.get("winner", "A")
    who = ("tuned" if w == "A" else "untuned") if order == 0 else ("untuned" if w == "A" else "tuned")
    return {"winner": who, "reason": j.get("reason")}


def with_retry(fn, *a, **k):
    import time
    for i in range(3):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            print(f"   [retry {i+1}] {str(e)[:70]}", flush=True)
            time.sleep(6)
    raise RuntimeError("call failed after retries")


def one(p):
    b = with_retry(llm.chat, [{"role": "system", "content": BRIEF_SYS},
                              {"role": "user", "content": f"PARAGRAPH:\n{p['text']}"}],
                   temperature=0.0, max_tokens=900)
    a = with_retry(gen, spec_b, b)
    u = with_retry(gen, spec_a, b)
    j0 = with_retry(judge, p["text"], a, u, 0)
    j1 = with_retry(judge, p["text"], u, a, 1)
    print(f"  {p['text'][:40]:<44} {j0['winner']:<8} {j1['winner']:<8} | {j0['reason']}", flush=True)
    return {"target": p["text"][:400], "tuned": a, "untuned": u, "j0": j0, "j1": j1}


print(f"段落级验收 {N} 条：{B_SPEC}（调优） vs {A_SPEC}（未调优）", flush=True)
with ThreadPoolExecutor(max_workers=3) as ex:
    rows = list(ex.map(one, held))

w = sum(1 for r in rows if r["j0"]["winner"] == r["j1"]["winner"] == "tuned")
l = sum(1 for r in rows if r["j0"]["winner"] == r["j1"]["winner"] == "untuned")
s = len(rows) - w - l
first = sum(1 for r in rows if r["j0"]["winner"] == "tuned")
print(f"\n双向一致：调优 {w} / 未调优 {l} / 分歧 {s}（n={len(rows)}；A 位胜率 {first}/{len(rows)}）")
(OUT / "judge.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n=== 样例（调优版）===")
for r in rows[:2]:
    print(f"\nTARGET: {r['target'][:300]}\n\nTUNED: {r['tuned'][:500]}")
