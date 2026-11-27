"""单步执行器：一次调用只做一件事，便于在慢/抖接口下逐步推进。

  python experiments/one.py draft <idx>
  python experiments/one.py style <idx>
  python experiments/one.py base  <idx>
  python experiments/one.py judge <idx>      # 需要 style/base 已生成
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, generate, llm, metrics  # noqa: E402

QUESTIONS = [
    "Is it worth buying an air fryer?",
    "Should I get a robot vacuum cleaner?",
    "What do you think of people who work from home?",
    "Is it acceptable to use AI to write your emails?",
    "My neighbour's leaf blower goes off at 7am every Sunday. What should I do?",
    "Is vaping better than smoking?",
]

OUT = config.OUTPUTS / "one"
OUT.mkdir(parents=True, exist_ok=True)
SPECV = "v1"

anc = anchors.load(config.ARTIFACTS / "anchors.jsonl")
spec = json.loads((config.ARTIFACTS / f"style_spec_{SPECV}.json").read_text(encoding="utf-8"))


def p(name, idx):
    return OUT / f"{name}_{idx}.json"


def main():
    cmd, idx = sys.argv[1], int(sys.argv[2])
    q = QUESTIONS[idx]
    t0 = time.time()

    if cmd == "draft":
        intent = generate.classify(q)
        notes = generate.draft(q, intent)
        p("draft", idx).write_text(json.dumps({"intent": intent, "notes": notes},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"intent={intent}  {time.time()-t0:.0f}s\n{notes[:600]}")

    elif cmd == "style":
        d = json.loads(p("draft", idx).read_text(encoding="utf-8"))
        refs = anchors.sample_anchors(anc, d["intent"], k=4, seed=generate.stable_seed(q))
        out = generate.stylize(q, d["notes"], refs, spec["compiled_prompt"])
        issues = generate.critic(out, spec.get("banned_phrases"))
        g = metrics.StyleMetrics.ngrams(out, 10)
        src = set()
        for a in refs:
            src |= metrics.StyleMetrics.ngrams(a["text"], 10)
        copied = len(g & src)
        if copied >= 2:
            issues.append(f"copies {copied} long phrases")
        p("style", idx).write_text(json.dumps({"text": out, "issues": issues,
                                               "anchors": [a["aid"] for a in refs]},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"issues={issues}  {time.time()-t0:.0f}s\n{out}")

    elif cmd == "base":
        out = generate.baseline(q)
        p("base", idx).write_text(json.dumps({"text": out}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{time.time()-t0:.0f}s\n{out}")

    elif cmd == "judge":
        refs = [a["text"][:700] for a in anc if a["aid"] in json.loads(p("style", idx).read_text(encoding="utf-8"))["anchors"]][:3]
        a = json.loads(p("style", idx).read_text(encoding="utf-8"))["text"]
        b = json.loads(p("base", idx).read_text(encoding="utf-8"))["text"]
        verdicts = []
        for order in (0, 1):
            first, second = (a, b) if order == 0 else (b, a)
            ref_block = "\n\n".join(f"[reference {i+1}] {t}" for i, t in enumerate(refs))
            user = (f"REFERENCE EXCERPTS by the target author:\n\n{ref_block}\n\nQUESTION: {q}\n\n"
                    f"TEXT A:\n{first}\n\nTEXT B:\n{second}\n\n"
                    "Which text was more likely written by the same person as the references? "
                    'Answer exactly: {"winner":"A" or "B","confidence":1-5,"reason":"<15 words>"}')
            raw = llm.chat([{"role": "system", "content": "You are a forensic stylometrist. Compare how texts are written, not what they say. JSON only."},
                            {"role": "user", "content": user}], temperature=0.1, max_tokens=1200)
            j = llm.parse_json(raw)
            who = ("pipeline" if j["winner"] == "A" else "baseline") if order == 0 else ("baseline" if j["winner"] == "A" else "pipeline")
            verdicts.append({"order": order, "raw_winner": j["winner"], "winner": who, "conf": j.get("confidence"), "reason": j.get("reason")})
        p("judge", idx).write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(verdicts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
