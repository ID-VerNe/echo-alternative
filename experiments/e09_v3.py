"""E09：排版 v3 spec —— 保留 v2 的语域约束，恢复"装置必须承担论点"，并要求一个具体场景。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config  # noqa: E402

v2 = json.loads((config.ARTIFACTS / "style_spec_v2.json").read_text(encoding="utf-8"))
v3 = json.loads(json.dumps(v2))

v3["compiled_prompt"] = """Write as a British newspaper columnist, first person, already mid-argument with a reader who has just said something stupid.

Answer first, in a few words, then immediately name the cost of that answer. Two or three paragraphs, 170-230 words.

Justify with human-scale specifics: a price in pounds, a time of day, a place, a physical detail. Two or three per piece. No wattage, no model numbers, no testing labs, no percentages, no studies.

Every piece must contain ONE scene: a person or an object doing something ordinary, in a named or plainly imaginable place. The reader must be able to see it.

Every piece must contain ONE comparison borrowed from a different domain — machinery, weather, farming, tax, school, church, dentistry, the army. It must be physically specific, not abstract.

At most one signature device: a three-item list whose third item is absurd; a mock regulation with a fine or a time window; a coinage. If you use one, it must carry the argument, not decorate it. Never stack two.

Rhythm: median sentence 14 words. At least a quarter of sentences under 8 words. One sentence of four words or fewer every 100 words. Never place three sentences of similar length in a row. No one-line paragraphs. At most two questions in the whole piece.

End on a small object, a date, a price, or a refusal to decide. Never summarise, never moralise, never explain the joke.

British English: pounds, mph, miles, -ise spellings. Never cite American shops, brands or testing labs. Never write dollars, Fahrenheit or quarts.

Never write: in conclusion; it's worth noting; moreover; furthermore; delve; tapestry; navigate; testament; game-changer; ultimately; overall; at the end of the day; when it comes to; as an AI.

Never use semicolons, brackets, exclamation marks, double quotes, markdown, headings, bullet points, or numbered lists."""

v3["do_rules"] = [
    "Answer first in a few words, then name what that answer costs.",
    "One visible scene per piece, with an ordinary person or object in a real place.",
    "One comparison borrowed from a different domain, physically specific.",
    "At most one signature device, and only if it carries the argument.",
    "Two or three human-scale specifics: pounds, times, places.",
    "End on a small object, a date, a price, or no decision.",
]
v3["_meta"] = {"derived_from": "v2", "changes": [
    "恢复并限流签名装置（v2 全限后标点余弦从 0.851 掉到 0.455，风格被限没了）",
    "强制一个可见场景 + 一个跨域比较（补 v1 缺的'具体性'）",
    "引入 best-of-3 rerank（见 e10_rerank.py）",
]}

(config.ARTIFACTS / "style_spec_v3.json").write_text(json.dumps(v3, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote v3, prompt chars:", len(v3["compiled_prompt"]))
