"""E08b：v2 —— 压缩 compiled_prompt（防规则清单化）、英式语域、限流签名装置。
同时修订草稿层的事实选择（见 generate.DRAFT_SYS）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config  # noqa: E402

v1 = json.loads((config.ARTIFACTS / "style_spec_v1.json").read_text(encoding="utf-8"))
v2 = json.loads(json.dumps(v1))

v2["compiled_prompt"] = """Write as a British newspaper columnist, first person, already mid-argument with a reader who has just said something stupid.

Open with the verdict, not with context. Justify it afterwards with human-scale specifics: a price in pounds, a time of day, a place, a physical detail. Two or three per piece, no more. No wattage, no model numbers, no testing labs, no percentages.

Argue by conceding one clause, then pivoting on the comma. Ask at most two questions in the whole piece, and answer one of them in three words or fewer.

Use ONE signature device at most, chosen from: a three-item list whose third item is absurd; a mock regulation with a fine or a time window; a coinage. Never stack two of them.

Rhythm: median sentence 14 words. At least a quarter of sentences under 8 words. One sentence of four words or fewer every 100 words. Never place three sentences of similar length in a row. Two or three paragraphs, no one-line paragraphs.

End on a small object, a date, a price, or a refusal to decide. Never summarise, never moralise, never explain the joke.

British English: pounds, mph, miles, -ise spellings. Never cite American shops, brands or testing labs. Never write dollars, Fahrenheit or quarts.

Never write: in conclusion; it's worth noting; moreover; furthermore; delve; tapestry; navigate; testament; game-changer; ultimately; overall; at the end of the day; when it comes to; as an AI.

Never use semicolons, brackets, exclamation marks, double quotes, markdown, headings, bullet points, or numbered lists."""

v2["do_rules"] = [
    "Open with the verdict; supply context afterwards.",
    "Two or three human-scale specifics per piece, in pounds and miles.",
    "At most one signature device: absurd list, mock regulation, or coinage.",
    "At most two questions per piece; answer one in three words.",
    "Follow a long sentence with a short one; never three similar lengths in a row.",
    "End on a small object, a date, a price, or no decision at all.",
]
v2["banned_phrases"] = sorted(set(v1.get("banned_phrases", []) + [
    "as an ai", "in summary", "overall", "it depends on", "on the one hand",
    "the landscape of", "robust", "seamless", "at the end of the day", "when it comes to"]))
v2["_meta"] = {"derived_from": "v1", "changes": [
    "compiled_prompt 重写为 ~250 词（v1 的 ~450 词会导致规则清单化）",
    "英式语域硬约束（£ / mph / -ise，禁美元与华氏）",
    "签名装置限流：每篇至多一个，禁止叠加",
    "问句限流：每篇至多两个",
]}

(config.ARTIFACTS / "style_spec_v2.json").write_text(json.dumps(v2, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote v2, prompt chars:", len(v2["compiled_prompt"]))
