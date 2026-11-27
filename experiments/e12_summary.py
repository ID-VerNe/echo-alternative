"""E12：汇总 v1/v2/v3/v4 与基线的指标+盲评，并做 rerank 消融，输出 docs/REPORT.md。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config  # noqa: E402

O = config.OUTPUTS


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def judge_stats(path):
    v = load(path)
    w = sum(1 for x in v if x["j0"]["winner"] == x["j1"]["winner"] == "pipeline")
    l = sum(1 for x in v if x["j0"]["winner"] == x["j1"]["winner"] == "baseline")
    s = len(v) - w - l
    first = sum(1 for x in v if x["j0"]["winner"] == "pipeline")
    return w, l, s, len(v), first


def avg(ms, k):
    return sum(m[k] for m in ms) / len(ms)


rows = []

r1 = load(O / "e06_v1_s6" / "runs_v1.json")
ms1 = [x["pipeline"]["metrics"] for x in r1]
rows.append(("v1：4 节点，单候选", ms1, judge_stats(O / "e06_v1_s6" / "judge_v1.json")))

r2 = load(O / "e06_v2_s6" / "runs_v1.json")
ms2 = [x["pipeline"]["metrics"] for x in r2]
rows.append(("v2：+ 英式语域 + 限流", ms2, judge_stats(O / "e06_v2_s6" / "judge_v1.json")))

r3 = load(O / "e10_v3_s6" / "runs_v3.json")
ms3 = [x["cands"][x["chosen"]]["metrics"] for x in r3]
rows.append(("v3：+ 场景 + 指标 rerank", ms3, judge_stats(O / "e10_v3_s6" / "judge_v3.json")))

r4 = load(O / "e11_v4_s12" / "runs_v4.json")
ms4 = [x["cands"][x["chosen"]]["metrics"] for x in r4]
j4 = judge_stats(O / "e11_v4_s12" / "judge_v4.json")
rows.append(("v4：+ 喜剧密度 + rubric rerank", ms4, j4))

cache = load(O / "base_cache.json")
base_ms = [cache[x["q"]]["metrics"] for x in r4]
rows.append(("基线：单条人设 Prompt", base_ms, (-1, -1, -1, len(r4), -1)))

# 消融：同一批 12 题，取候选 0 而不是 rubric 选的
c0 = [x["cands"][0]["metrics"] for x in r4]
rub = [x["cands"][x["chosen"]]["rubric"].get("total", 0) for x in r4]
rub_all = [c["rubric"].get("total", 0) for x in r4 for c in x["cands"]]
rows.append(("v4 消融：只用候选 0（无 rerank）", c0, (-1, -1, -1, len(r4), -1)))

print(f"{'方案':<32}{'Δref↓':>7}{'margin↑':>9}{'knn↑':>8}{'JS↓':>7}{'punct↑':>8}{'盲评 胜/负/分歧':>18}")
for name, ms, js in rows:
    jtxt = "—" if js[0] < 0 else f"{js[0]}/{js[1]}/{js[2]}  (n={js[3]})"
    print(f"{name:<32}{avg(ms,'delta_ref'):>7.3f}{avg(ms,'delta_margin'):>+9.3f}"
          f"{avg(ms,'knn_margin'):>+8.3f}{avg(ms,'sent_js'):>7.3f}{avg(ms,'punct_cos'):>8.3f}{jtxt:>18}")

print(f"\nrerank 消融：rubric 均分 选中={sum(rub)/len(rub):.2f} 全部候选={sum(rub_all)/len(rub_all):.2f}")
print(f"             Δref 选中={avg(ms4,'delta_ref'):.3f} 候选0={avg(c0,'delta_ref'):.3f}")
print(f"             首轮选中管线的题数 j0={j4[4]}/{j4[3]}（j1 反向={sum(1 for x in load(O/'e11_v4_s12'/'judge_v4.json') if x['j1']['winner']=='pipeline')}）")

# ---------- markdown ----------
md = ["# E12 汇总：从 v1 到 v4\n",
      f"盲评为同模型双向顺序（两次都选同一方才算胜）。\n",
      "| 方案 | Δref ↓ | Δmargin ↑ | kNN ↑ | 句长JS ↓ | 标点cos ↑ | 盲评 胜/负/分歧 |",
      "|---|---|---|---|---|---|---|"]
for name, ms, js in rows:
    jtxt = "—" if js[0] < 0 else f"{js[0]}/{js[1]}/{js[2]}（n={js[3]}）"
    md.append(f"| {name} | {avg(ms,'delta_ref'):.3f} | {avg(ms,'delta_margin'):+.3f} | "
              f"{avg(ms,'knn_margin'):+.3f} | {avg(ms,'sent_js'):.3f} | {avg(ms,'punct_cos'):.3f} | {jtxt} |")

md.append(
    f"\nrerank 消融（12 题同一批候选）：rubric 均分 选中 {sum(rub)/len(rub):.2f} / 全部候选 {sum(rub_all)/len(rub_all):.2f}；"
    f"Δref 选中 {avg(ms4,'delta_ref'):.3f} / 候选0 {avg(c0,'delta_ref'):.3f}。\n")

Path(config.ROOT / "docs" / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
print("\nwrote docs/REPORT.md")
