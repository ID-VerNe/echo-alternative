"""E21：对照 Gemini 的"法医式挑刺"，量化我们自己文本里的同款痕迹。
检查项：① 英镑价格+英国地名+钟点 三件套共现；② 跨篇重复意象；③ 五词短句收尾；④ 开头类型。"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\VerNe\Downloads\Documents\echo替代")
ART = ROOT / "artifacts"
OUT = ROOT / "outputs"

SETS = {
    "v4 管线": json.loads((OUT / "e11_v4_s12" / "runs_v4.json").read_text(encoding="utf-8")),
}
# 统一成 {q: [texts...]}
data = {"v4 管线": {r["q"]: r["cands"][r["chosen"]]["text"] for r in SETS["v4 管线"]}}

api = json.loads((ART / "agent_api_runs.json").read_text(encoding="utf-8"))
data["agent 调接口·初稿"] = {r["q"]: r["rounds"][0]["text"] for r in api}
data["agent 调接口·定稿"] = {r["q"]: r["final"] for r in api}

slf = json.loads((ART / "agent_self_runs.json").read_text(encoding="utf-8"))
data["agent 自写·初稿"] = {r["q"]: r["rounds"][0]["text"] for r in slf}
data["agent 自写·定稿"] = {r["q"]: r["final"] for r in slf}

# ---------- 先认出 PDF 里那四段出自哪一版 ----------
PROBES = {
    "空气炸锅": "saucepan that hates you",
    "扫地机器人": "never met a cable it couldn't hate",
    "居家办公": "people who work from home. I mind what it does",
    "AI 写邮件": "never let a machine write an email",
}
print("=== PDF 四段来源定位 ===")
for name, probe in PROBES.items():
    hits = []
    for set_name, qs in data.items():
        for q, t in qs.items():
            if probe.lower() in t.lower().replace("\n", " "):
                hits.append(set_name)
    print(f"{name:<10} -> {sorted(set(hits)) or '未命中'}")

# ---------- 痕迹统计 ----------
UK_PLACE = r"(Croydon|Leeds|London|Cheltenham|Slough|Reading|Bristol|Manchester|Surrey|Kent|Sussex|Yorkshire|Devon|Cornwall|Norwich|Hull|Glasgow|Edinburgh|Cardiff|Wales|Scotland)"
PRICE = r"£\s?\d[\d,]*"
CLOCK = r"\b(\d{1,2}[.:]\d{2}\s?(am|pm)|at (seven|eight|nine|ten|eleven|six|seven fifteen|eight fifteen))\b"
MOTIFS = ["sergeant major", "vicar", "church", "farmhand", "farmer", "wardrobe", "kettle",
          "spaniel", "dog", "mop", "beadle", "sheep", "scarecrow", "tractor", "piano",
          "hamster", "wheelbarrow", "organist", "denture", "sock"]


def closing_len(t):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()]
    return len(sents[-1].split()) if sents else 0


def opening_type(t):
    first = re.split(r"(?<=[.!?])\s+", t.strip())[0].lower()
    if first.rstrip().endswith("?"):
        return "问句"
    if re.match(r"^(yes|no|probably|not really|absolutely|i'd|i would)", first):
        return "判断"
    if re.match(r"^(at|on|when|last|this|one|yesterday|my )", first):
        return "场景"
    return "陈述"


for name, qs in data.items():
    texts = list(qs.values())
    n = len(texts)
    triple = sum(1 for t in texts
                 if re.search(PRICE, t) and re.search(UK_PLACE, t) and re.search(CLOCK, t))
    price = sum(1 for t in texts if re.search(PRICE, t))
    place = sum(1 for t in texts if re.search(UK_PLACE, t))
    clock = sum(1 for t in texts if re.search(CLOCK, t))
    motif_count = defaultdict(int)
    for t in texts:
        low = t.lower()
        for m in MOTIFS:
            if m in low:
                motif_count[m] += 1
    repeats = [f"{m}×{c}" for m, c in motif_count.items() if c >= 2]
    short_closing = sum(1 for t in texts if closing_len(t) <= 6)
    openers = Counter(opening_type(t) for t in texts)
    print(f"\n=== {name}（n={n}）===")
    print(f"三件套共现 {triple}/{n} ｜ £价格 {price} ｜ 英国地名 {place} ｜ 钟点 {clock}")
    print(f"五词内收尾 {short_closing}/{n} ｜ 开头类型 {dict(openers)}")
    print(f"跨篇重复意象（≥2 篇）：{', '.join(repeats) if repeats else '无'}")
