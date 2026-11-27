"""E15：带预算的编译器（通用约束，不针对某个作者）。
诊断：一次性归纳出的 compiled_prompt 会把直方图百分比原样抄进去（17.22 词、32.2%、9.47%…），
模型于是去凑数字而不是学语感；克拉克森的 v1 也有这毛病，当时是手工压到 450 词、v4 压到 250 词。
这里把"手工压缩"变成编译器的固定约束。"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors as anc_mod, config, llm, spec as spec_mod, stats  # noqa: E402

BUDGET = """
ADDITIONAL CONSTRAINTS for "compiled_prompt":
- 200-260 words. Not one word more.
- At most three numbers in total. No percentage tables, no histogram dumps, no per-100-word rates.
- Write it as instructions a writer can follow while thinking about the argument, not as a measurement report.
- Every rule must be executable without a calculator.
The hard statistics are given so you can judge what matters; they are not text to transcribe."""


def budget_ok(p):
    wc = len(p.split())
    return 180 <= wc <= 280 and p.count("%") <= 2


def compile_budgeted(anchors_path, out_path, per_intent=3, seed=3):
    anchors = anc_mod.load(anchors_path)
    prof = stats.profile(anchors)
    ex = []
    for it in anc_mod.INTENTS:
        for p in anc_mod.sample_anchors(anchors, it, k=per_intent, seed=seed):
            ex.append(f"[{it}] {p['text'][:900]}")
    user = spec_mod.SPEC_USER.format(stats=spec_mod.hard_stats_block(prof), excerpts="\n\n".join(ex)) + BUDGET
    raw = llm.chat([{"role": "system", "content": spec_mod.SPEC_SYS},
                    {"role": "user", "content": user}], temperature=0.4, max_tokens=6000)
    spec = llm.parse_json(raw)
    p = spec["compiled_prompt"]
    if not budget_ok(p):
        fix = llm.chat([{"role": "system", "content": spec_mod.SPEC_SYS},
                        {"role": "user", "content": user +
                         f"\n\nYour previous compiled_prompt was {len(p.split())} words and contained "
                         f"{p.count('%')} percent signs. Rewrite the WHOLE JSON again, with a compiled_prompt "
                         "of 200-260 words and at most three numbers."}], temperature=0.2, max_tokens=6000)
        spec = llm.parse_json(fix)
        p = spec["compiled_prompt"]
    spec["_meta"] = {"n_anchors": len(anchors), "stats": prof, "budgeted": budget_ok(p)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path.name}: {len(p.split())} words, {p.count('%')} percent signs, budget_ok={budget_ok(p)}")
    return spec


if __name__ == "__main__":
    compile_budgeted(config.ARTIFACTS / "may_anchors.jsonl",
                     config.ARTIFACTS / "style_spec_may_v2.json")
