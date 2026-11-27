"""E05：生成 style_spec.json 并打印关键字段，人工校验质量。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, spec  # noqa: E402

anc = anchors.load(config.ARTIFACTS / "anchors.jsonl")
print("锚点", len(anc))
s = spec.build(anc, per_intent=2)
spec.save(s, config.ARTIFACTS / "style_spec_v1.json")

for k in ("voice", "thinking_moves", "signature_devices", "openers", "closers",
          "rhythm_rules", "lexicon_use", "banned_phrases", "never_does", "do_rules"):
    v = s.get(k)
    print(f"\n### {k}\n" + (str(v) if not isinstance(v, list) else "\n".join(f"- {x}" for x in v)))
print("\n### compiled_prompt\n" + str(s.get("compiled_prompt")))
