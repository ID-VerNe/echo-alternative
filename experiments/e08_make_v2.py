"""E08：v1 → v2 的定向修订（基于人工审读 + 盲评失败例）。
只加可执行约束，不改结构。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config  # noqa: E402

v1 = json.loads((config.ARTIFACTS / "style_spec_v1.json").read_text(encoding="utf-8"))
v2 = json.loads(json.dumps(v1))

v2["compiled_prompt"] = v1["compiled_prompt"].rstrip() + (
    "\n\nWrite British English, not American: prices in pounds, speeds in mph, distances in miles, "
    "spelling in -our and -ise. Do not cite American institutions, shops or testing labs "
    "(Wirecutter, America's Test Kitchen, the FDA, Target) unless the question is explicitly about America. "
    "When you need a number you do not know, make it plausible and British, then move on."
)

extra_banned = ["as an ai", "in summary", "overall", "it depends on",
                "on the one hand", "the landscape of", "robust", "seamless",
                "at the end of the day", "when it comes to"]
v2["banned_phrases"] = sorted(set(v1.get("banned_phrases", []) + extra_banned))

v2["do_rules"] = v1.get("do_rules", []) + [
    "Write British English: pounds, mph, miles, -ise spellings.",
    "Never cite an American brand or testing lab as evidence.",
]

config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
(config.ARTIFACTS / "style_spec_v2.json").write_text(json.dumps(v2, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote style_spec_v2.json")
print("compiled_prompt chars:", len(v2["compiled_prompt"]))
print("banned:", len(v2["banned_phrases"]))
