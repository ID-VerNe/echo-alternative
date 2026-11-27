"""命令行入口：python -m pipeline.cli "你的问题"
走完整 4 节点管线，只打印最终回复（--trace 打印中间态）。"""
import json
import sys

from . import anchors, config, generate


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    trace = "--trace" in sys.argv
    if not args:
        print('usage: python -m pipeline.cli "question"')
        return
    q = " ".join(args)
    anc = anchors.load(config.ARTIFACTS / "anchors.jsonl")
    spec_path = config.ARTIFACTS / "style_spec_v2.json"
    if not spec_path.exists():
        spec_path = config.ARTIFACTS / "style_spec_v1.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    r = generate.run(q, anc, spec)
    if trace:
        print("INTENT:", r["intent"])
        print("ANCHORS:", r["anchors"])
        print("ISSUES:", r["issues"])
        print("\nNOTES:\n" + r["notes"] + "\n")
    print(r["final"])


if __name__ == "__main__":
    main()
