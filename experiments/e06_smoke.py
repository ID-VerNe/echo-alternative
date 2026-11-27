"""冒烟测试：单题跑通 4 节点，检查产出与质检。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import anchors, config, generate  # noqa: E402

anc = anchors.load(config.ARTIFACTS / "anchors.jsonl")
spec = json.loads((config.ARTIFACTS / "style_spec_v1.json").read_text(encoding="utf-8"))

q = sys.argv[1] if len(sys.argv) > 1 else "Is it worth buying an air fryer?"
r = generate.run(q, anc, spec)
print("INTENT:", r["intent"])
print("\n=== NOTES ===\n", r["notes"])
print("\n=== ISSUES ===\n", r["issues"])
print("\n=== RAW ===\n", r["raw"])
if r["final"] != r["raw"]:
    print("\n=== FINAL (after repair) ===\n", r["final"])
