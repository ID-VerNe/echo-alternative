"""E13：绝对评分复核 —— pairwise 有强位置偏差（j0 选管线 11/12，j1 只 6/12）。
改成每条文本独立打分（1-10 的 Clarkson-ness），不看对手，再做配对比较。"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, llm  # noqa: E402

O = config.OUTPUTS
runs = json.loads((O / "e11_v4_s12" / "runs_v4.json").read_text(encoding="utf-8"))
cache = json.loads((O / "base_cache.json").read_text(encoding="utf-8"))
anchors_list = anchors.load(config.ARTIFACTS / "anchors.jsonl")

SYS = """You are a forensic stylometrist. You score one anonymous text at a time against reference
excerpts by a known author. You judge how it is written, never what it says. Answer JSON only."""

USER = """REFERENCE EXCERPTS by the target author:

{refs}

QUESTION THE TEXT ANSWERS: {q}

TEXT TO SCORE:
{text}

Score this text alone, 1-10, on each axis:
diction: word choice and register match
rhythm: sentence-length pattern and pacing match
comedy: absurd comparisons, escalation, punchline — the author's engine
voice: sounds like a person talking, not a columnist performing
restraint: never summarises or moralises
Answer exactly: {{"diction":n,"rhythm":n,"comedy":n,"voice":n,"restraint":n,"total":n}}"""


def score(q, text):
    refs = "\n\n".join(f"[{i+1}] {a['text'][:600]}" for i, a in enumerate(anchors_list[:3]))
    raw = llm.chat([{"role": "system", "content": SYS},
                    {"role": "user", "content": USER.format(refs=refs, q=q, text=text)}],
                   temperature=0.0, max_tokens=900)
    j = llm.parse_json(raw)
    # 模型自报的 total 量表不稳（有时 5 分制、有时五轴之和），一律按五轴重算
    j["total"] = sum(float(j.get(k, 0)) for k in ("diction", "rhythm", "comedy", "voice", "restraint"))
    return j


tasks = []
for r in runs:
    tasks.append(("pipeline", r["q"], r["cands"][r["chosen"]]["text"]))
    tasks.append(("baseline", r["q"], cache[r["q"]]["text"]))

print(f"absolute scoring {len(tasks)} texts …", flush=True)
with ThreadPoolExecutor(max_workers=3) as ex:
    scores = list(ex.map(lambda t: score(t[1], t[2]), tasks))

p = {(t[1]): s for t, s in zip(tasks, scores) if t[0] == "pipeline"}
b = {(t[1]): s for t, s in zip(tasks, scores) if t[0] == "baseline"}

keys = ("diction", "rhythm", "comedy", "voice", "restraint", "total")
pm = {k: sum(p[q].get(k, 0) for q in p) / len(p) for k in keys}
bm = {k: sum(b[q].get(k, 0) for q in b) / len(b) for k in keys}
print(f"\n{'轴':<10}{'管线':>7}{'基线':>7}")
for k in keys:
    print(f"{k:<10}{pm[k]:>7.2f}{bm[k]:>7.2f}")

pw = sum(1 for q in p if p[q].get("total", 0) > b[q].get("total", 0))
bw = sum(1 for q in p if p[q].get("total", 0) < b[q].get("total", 0))
tie = len(p) - pw - bw
print(f"\n配对胜负（total）：管线 {pw} / 基线 {bw} / 平 {tie}（n={len(p)}）")
for q in p:
    print(f"  {q[:44]:<46} {p[q].get('total',0):>3} vs {b[q].get('total',0):>3}"
          f"  comedy {p[q].get('comedy',0)}/{b[q].get('comedy',0)}")

(O / "e13_absolute.json").write_text(json.dumps(
    {"pipeline": p, "baseline": b, "mean_pipeline": pm, "mean_baseline": bm},
    ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote outputs/e13_absolute.json")
