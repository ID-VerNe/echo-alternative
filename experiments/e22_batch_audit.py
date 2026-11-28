"""E22：批级门禁 —— 把外部批评指出的"模板痕迹"变成可执行的硬检查。
检查 5 项：
1) 每篇数字预算 ≤1，且不得出现房产级价格（≥£100,000）
2) 英镑价格 + 英国地名 + 钟点 不得同篇共现（防"三件套"）
3) 意象域轮转：同一意象不得跨 2 篇以上复用
4) 结尾 ≤6 词的"自拆台"最多占 1/4
5) 开头前三词不得重复
输出每篇违规项与整批通过率。"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\VerNe\Downloads\Documents\echo替代")
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "agent_self_runs.json"
rows = json.loads(path.read_text(encoding="utf-8"))
texts = [r.get("final") or r["rounds"][-1]["text"] for r in rows]
qs = [r["q"] for r in rows]

UK_PLACE = r"(Croydon|Leeds|London|Cheltenham|Slough|Reading|Bristol|Manchester|Surrey|Kent|Sussex|Yorkshire|Devon|Cornwall|Norwich|Hull|Glasgow|Edinburgh|Cardiff|Wales|Scotland|Birmingham|Liverpool|Sheffield|Nottingham)"
CLOCK = r"\b(\d{1,2}[.:]\d{2}\s?(am|pm)|\d{1,2}\s?(am|pm)|at (seven|eight|nine|ten|eleven|six))\b"
PRICE = r"£\s?[\d,]+"
DOMAINS = {
    "church": r"\b(church|vicar|pew|hymn|prayer|beadle|chapel|organist)\b",
    "farm": r"\b(farm|farmer|farmhand|sheep|tractor|harvest|scarecrow|livestock)\b",
    "army": r"\b(army|sergeant|regiment|barracks|parade|major|private)\b",
    "kitchen": r"\b(kettle|teapot|sink|worktop|fridge|oven|mug|crockery)\b",
    "weather": r"\b(fog|drizzle|gale|frost|heatwave|downpour)\b",
    "tax": r"\b(tax|VAT|HMRC|invoice|receipt|form|rebate)\b",
    "school": r"\b(school|teacher|prefect|classroom|playground|homework)\b",
    "dentist": r"\b(dentist|tooth|filling|drill|gum|molar)\b",
}


def sents(t):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", t.strip()) if x.strip()]


def audit_piece(q, t):
    v = []
    prices = re.findall(PRICE, t)
    nums = re.findall(r"\d[\d,]*", t)
    if len(nums) > 1:
        v.append(f"数字超额({len(nums)}个)")
    for p in prices:
        try:
            val = int(p.replace("£", "").replace(",", "").strip())
        except ValueError:
            continue
        if val >= 100000:
            v.append(f"房产级价格{p}")
    if prices and re.search(UK_PLACE, t) and re.search(CLOCK, t):
        v.append("三件套共现(£+地名+钟点)")
    s = sents(t)
    if len(s[-1].split()) <= 6:
        v.append("五词短收尾")
    return v


viol = {q: audit_piece(q, t) for q, t in zip(qs, texts)}

# 意象域复用
domain_hits = {d: [i for i, t in enumerate(texts) if re.search(pat, t, re.I)] for d, pat in DOMAINS.items()}
overused = {d: hs for d, hs in domain_hits.items() if len(hs) >= 2}

# 开头重复
openers = Counter(" ".join(t.strip().replace("\n", " ").split()[:3]) for t in texts)
dup_openers = {k: c for k, c in openers.items() if c >= 2}

print(f"审计 {path.name}：{len(texts)} 篇\n")
for i, q in enumerate(qs):
    marks = "；".join(viol[q]) or "ok"
    print(f"{i+1:>2} {q[:40]:<42} {marks}")

print("\n--- 批级 ---")
print(f"短收尾占比 {sum(1 for t in texts if len(sents(t)[-1].split()) <= 6)}/{len(texts)}"
      f"（上限 {len(texts)//4}）")
print("意象域跨篇复用：", {d: len(h) for d, h in overused.items()} or "无")
print("开头前三词重复：", dup_openers or "无")

fails = sum(1 for q in qs if viol[q])
print(f"\n有违规的篇数 {fails}/{len(texts)}；"
      f"其中三件套 {sum(1 for q in qs if '三件套共现(£+地名+钟点)' in viol[q])}、"
      f"房产价 {sum(1 for q in qs if any(x.startswith('房产级') for x in viol[q]))}、"
      f"数字超额 {sum(1 for q in qs if any(x.startswith('数字超额') for x in viol[q]))}")
