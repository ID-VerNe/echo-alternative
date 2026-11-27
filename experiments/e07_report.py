"""E07：把评测结果整理成 markdown 报告。用法：python e07_report.py <e06_dir> <out.md>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config  # noqa: E402

src = config.OUTPUTS / (sys.argv[1] if len(sys.argv) > 1 else "e06_s6")
out = Path(sys.argv[2]) if len(sys.argv) > 2 else (src / "report.md")

runs = json.loads((src / "runs_v1.json").read_text(encoding="utf-8"))
try:
    judge = json.loads((src / "judge_v1.json").read_text(encoding="utf-8"))
except FileNotFoundError:
    judge = []

lines = [f"# 评测报告：{src.name}", ""]

keys = [("delta_ref", "Δref ↓"), ("delta_margin", "Δmargin ↑"), ("knn_margin", "knnM ↑"),
        ("sent_js", "JS ↓"), ("punct_cos", "punct ↑"), ("anchor_overlap8", "overlap ↓"),
        ("words", "words")]

lines += ["## 逐题指标", "",
          "| # | 问题 | 系统 | " + " | ".join(k[1] for k in keys) + " | 质检问题 |",
          "|---|---|---|" + "---|" * len(keys) + "---|"]
for i, r in enumerate(runs):
    for name, m, iss in (("管线", r["pipeline"]["metrics"], r["pipeline"]["issues"]),
                         ("基线", r["baseline"]["metrics"], [])):
        lines.append(f"| {i} | {r['q'][:34]} | {name} | " +
                     " | ".join(f"{m[k]:.3f}" if isinstance(m[k], float) else str(m[k]) for k, _ in keys) +
                     f" | {len(iss)} |")

lines += ["", "## 均值", ""]
for name, sel in (("管线", lambda r: r["pipeline"]["metrics"]),
                  ("基线", lambda r: r["baseline"]["metrics"])):
    ms = [sel(r) for r in runs]
    row = {k: sum(m[k] for m in ms) / len(ms) for k, _ in keys}
    lines.append(f"- **{name}**: " + " / ".join(f"{k}={row[k]:.3f}" for k, _ in keys))

if judge:
    win = sum(1 for v in judge if v["j0"]["winner"] == v["j1"]["winner"] == "pipeline")
    lose = sum(1 for v in judge if v["j0"]["winner"] == v["j1"]["winner"] == "baseline")
    lines += ["", "## 盲评（双向顺序一致）", "",
              f"- 管线胜 {win} / 基线胜 {lose} / 分歧 {len(judge) - win - lose}，共 {len(judge)} 题", ""]
    for v in judge:
        lines.append(f"- {v['q'][:40]}：{v['j0']['winner']}/{v['j1']['winner']} — {v['j0']['reason']}")

lines += ["", "## 样例", ""]
for r in runs[:3]:
    lines += [f"### {r['q']}", "", "```text", r["pipeline"]["final"].strip(), "```", "",
              "基线对照：", "", "```text", r["baseline"]["text"].strip()[:900], "```", ""]

out.write_text("\n".join(lines), encoding="utf-8")
print(f"written {out}  ({len(runs)} runs, {len(judge)} judge pairs)")
