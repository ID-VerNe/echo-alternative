"""在线推理管线：意图分类 → 事实清单 → 风格转译 → 规则质检/局部重写。
另含 naive persona baseline 作为对照组。"""
import json
import re

from . import anchors as anc_mod
from . import config, llm, metrics, textutil

CLASSIFY_SYS = ("Classify the user's message into exactly one rhetorical intent. "
                "Reply with JSON only: {\"intent\": one of [\"rant\",\"argument\",\"anecdote\",\"selfmock\"], "
                "\"why\": \"<8 words>\"}.")

DRAFT_SYS = """You are a fact-checker and outliner. You do NOT write prose and you do NOT imitate anyone.
Given a question, produce a tight brief that a British newspaper columnist will later turn into an article.

Output plain notes with these headings:
STANCE: one sentence, the actual position (take a side; no both-sidesing).
POINTS: 4-6 bullets, each a concrete claim that supports the stance.
EVIDENCE: 3-4 HUMAN-SCALE specifics only: a price in pounds, a time of day, a place name, one physical detail.
  No wattage, no model numbers, no testing labs, no percentages, no studies. If you must use a number, make it odd and precise.
COUNTER: the strongest objection, one line.
IMAGE: one physical, visual example that could anchor the piece.
ANGLE: the emotional temperature (mild annoyance / disbelief / delight / self-mockery).
Do not use metaphor. Do not use rhetorical questions. Do not write sentences longer than 25 words."""

STYLE_GUARD = """STYLE REFERENCES follow. They are not sources. They are not facts.
Learn ONLY their sentence rhythm, punctuation habits, sentence-opening habits and closing habits.
NEVER reuse their names, numbers, places, products, jokes, examples or claims in your answer.
If any specific from the references reappears in your text, the answer is wrong.

Write about the user's question only. Use the brief's facts, not the references' facts."""

CRITIC_BANNED = [
    "in conclusion", "to sum up", "it's worth noting", "it is important to note",
    "it is worth noting", "moreover", "furthermore", "additionally", "delve",
    "tapestry", "navigate the", "testament to", "game-changer", "game changer",
    "let's explore", "in today's", "overall,", "ultimately,", "firstly", "secondly",
    "as an ai", "i cannot", "it depends", "on one hand", "on the other hand",
    "this raises", "shed light", " multifaceted", "leverage", "robust", "seamless",
]


def classify(question):
    """规则优先（零调用、零延迟），不确定才问模型。"""
    q = question.lower()
    if re.search(r"\b(should i|is it worth|would you|is it acceptable|are .{0,20} worth|"
                 r"better than|do i need)\b", q):
        return "argument"
    if re.search(r"\b(what do you think|why|menace|con|scam|rip-?off|hate)\b", q):
        return "rant"
    if re.search(r"\b(my |i have|my neighbour|my colleague|i'm |what should i do)\b", q):
        return "anecdote"
    try:
        raw = llm.chat([{"role": "system", "content": CLASSIFY_SYS},
                        {"role": "user", "content": question}],
                       temperature=0.0, max_tokens=600)
        return llm.parse_json(raw).get("intent", "argument")
    except Exception:  # noqa: BLE001
        return "argument"


def draft(question, intent):
    return llm.chat([{"role": "system", "content": DRAFT_SYS},
                     {"role": "user", "content": f"QUESTION: {question}\nINTENT: {intent}"}],
                    temperature=0.3, max_tokens=2000)


def stylize(question, notes, anchors, spec_prompt, min_words=170, max_words=230):
    refs = "\n\n".join(f"[{a['intent']}] {a['text'][:900]}" for a in anchors)
    user = (f"{STYLE_GUARD}\n\n=== STYLE REFERENCES ===\n{refs}\n\n"
            f"=== BRIEF ===\n{notes}\n\n"
            f"=== TASK ===\nAnswer this question: {question}\n"
            f"Length: {min_words}-{max_words} words, 2 to 3 paragraphs. No headings, no bullets, no markdown.")
    return llm.chat([{"role": "system", "content": spec_prompt},
                     {"role": "user", "content": user}],
                    temperature=0.85, max_tokens=2500)


def critic(text, banned_extra=None):
    t = text.lower()
    issues = []
    for b in CRITIC_BANNED + (banned_extra or []):
        if b.strip().lower() in t:
            issues.append(f"banned phrase: {b}")
    if ";" in text:
        issues.append("semicolon")
    if re.search(r"[()]", text):
        issues.append("parentheses")
    if "!" in text:
        issues.append("exclamation mark")
    if '"' in text:
        issues.append("double quotes")
    if re.search(r"^\s*(#|\*|\d\.|-)", text, re.M):
        issues.append("markdown list/heading")
    sents = textutil.split_sentences(text)
    ns = len(sents) or 1
    lens = [textutil.word_count(s) for s in sents]
    short = sum(1 for x in lens if x <= 8) / ns
    if short < 0.18:
        issues.append(f"too few short sentences ({short:.0%} < 18%)")
    if (sum((x - sum(lens) / ns) ** 2 for x in lens) / ns) ** 0.5 < 7.5:
        issues.append("sentence lengths too uniform")
    if ns < 8:
        issues.append("too few sentences")
    if re.search(r"\b(in conclusion|to conclude|overall|in summary)\b", t):
        issues.append("summary ending")
    paras = [p for p in text.split("\n\n") if p.strip()]
    if len(paras) > 4:
        issues.append(f"too many paragraphs ({len(paras)})")
    return issues


REPAIR_SYS = ("You are a line editor. Fix ONLY the listed violations. "
              "Do not change the stance, facts, numbers, or structure. "
              "Do not add new claims. Keep the same length class.")


def repair(text, issues, spec_prompt):
    user = (f"VIOLATIONS:\n" + "\n".join("- " + i for i in issues) +
            f"\n\nTEXT:\n{text}\n\nRewrite the text so that none of the violations remain.")
    return llm.chat([{"role": "system", "content": spec_prompt + "\n\n" + REPAIR_SYS},
                    {"role": "user", "content": user}],
                   temperature=0.5, max_tokens=2600)


PERSONA = """You are Jeremy Clarkson, the British motoring journalist and TV presenter.
Write in his voice: blunt, opinionated, funny, exaggerated, rude about things you dislike,
fond of Britain and machines. Use short punchy sentences. Be entertaining."""


def baseline(question, min_words=150, max_words=200):
    return llm.chat([{"role": "system", "content": PERSONA},
                     {"role": "user", "content": f"{question}\n\nWrite {min_words}-{max_words} words."}],
                    temperature=0.85, max_tokens=2500)


def stable_seed(text):
    import hashlib
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16) % 9999


def run(question, anchors, spec, trace=None, min_words=170, max_words=230):
    intent = classify(question)
    notes = draft(question, intent)
    refs = anc_mod.sample_anchors(anchors, intent, k=4, seed=stable_seed(question))
    spec_prompt = spec["compiled_prompt"]
    out = stylize(question, notes, refs, spec_prompt, min_words, max_words)
    # 防串台：与参考段落的 10-gram 重叠
    ref_texts = [a["text"] for a in refs]
    g = metrics.StyleMetrics.ngrams(out, 10)
    src = set()
    for t in ref_texts:
        src |= metrics.StyleMetrics.ngrams(t, 10)
    copied = len(g & src)
    issues = critic(out, spec.get("banned_phrases"))
    if copied >= 2:
        issues = [i for i in issues if not i.startswith("banned")] + [f"copies {copied} long phrases from the references"]
    fixed = None
    if issues:
        fixed = repair(out, issues, spec_prompt)
    result = {"question": question, "intent": intent, "notes": notes,
              "anchors": [a["aid"] for a in refs], "raw": out,
              "issues": issues, "final": fixed or out}
    if trace:
        trace.append(result)
    return result
