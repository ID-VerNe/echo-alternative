"""离线 spec 生成：LLM 只做定性归纳，硬数字由代码注入。产出 style_spec.json。"""
import json

from . import anchors as anc_mod
from . import config, llm, stats

SPEC_SYS = """You are a stylometrist. Your job is to describe HOW a writer writes, never WHAT they say.
Rules:
- Never mention the topics, plots, people, brands or events in the excerpts. Only the machinery of the prose.
- Be specific and operational: every rule must be checkable by counting or by pattern matching.
- Prefer negative constraints (what the writer never does) over vague praise.
- Output STRICT JSON, no markdown, no commentary."""

SPEC_USER = """Below are excerpts from one writer's books, plus hard statistics computed over ~1.9M words of his writing.

HARD STATS (trust these, do not estimate):
{stats}

EXCERPTS (by rhetorical function):
{excerpts}

Produce JSON with exactly these keys:
{{
  "voice": "one paragraph, max 80 words, describing the voice as a mechanism, not adjectives",
  "thinking_moves": [6-8 short items: how he starts a piece / how he argues / how he ends],
  "signature_devices": [6-10 items: concrete devices, each with a mini example pattern],
  "openers": [5-8 patterns he actually uses to open a paragraph, quoted from the excerpts where possible],
  "closers": [4-6 patterns he uses to end, especially how he refuses to sum up],
  "rhythm_rules": [5-8 numeric or countable rules about sentence and paragraph length],
  "lexicon_use": [4-6 items about word choice: register, hyphenated compounds, profanity, brand names],
  "banned_phrases": [10-16 phrases this writer would NEVER use; include generic AI/essay phrases],
  "never_does": [6-10 structural things he never does],
  "do_rules": [6-10 imperative instructions for an imitator],
  "compiled_prompt": "a compact system prompt, max 450 words, second person imperative, that would make a language model write like him; include the numeric rhythm rules and the banned phrases"
}}"""


def hard_stats_block(profile):
    keys = ["sent_len_mean", "sent_len_median", "sent_len_stdev", "pct_short_8",
            "pct_long_30", "question_rate", "para_wc_mean", "first_person_per_100w"]
    lines = [f"- {k}: {profile[k]}" for k in keys]
    lines.append(f"- sentence length histogram (<=6/7-12/13-18/19-25/26-35/36+): {profile['sent_len_hist_pct']}")
    p = profile["punct_per_100w"]
    lines.append("- punctuation per 100 words: " + ", ".join(f"{k}={v}" for k, v in p.items()))
    lines.append(f"- most frequent paragraph openers: {profile['top_openers'][:10]}")
    return "\n".join(lines)


def build(anchors, per_intent=2, seed=3):
    prof = stats.profile(anchors)
    ex = []
    for it in anc_mod.INTENTS:
        for p in anc_mod.sample_anchors(anchors, it, k=per_intent, seed=seed):
            ex.append(f"[{it}] {p['text'][:900]}")
    user = SPEC_USER.format(stats=hard_stats_block(prof), excerpts="\n\n".join(ex))
    raw = llm.chat([{"role": "system", "content": SPEC_SYS},
                    {"role": "user", "content": user}],
                   temperature=0.4, max_tokens=6000)
    try:
        spec = llm.parse_json(raw)
    except Exception:  # 一次重试，带上"只输出 JSON"的强约束
        raw2 = llm.chat([{"role": "system", "content": SPEC_SYS},
                         {"role": "user", "content": user + "\n\nReturn ONLY the JSON object. No prose."}],
                        temperature=0.2, max_tokens=6000)
        config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (config.ARTIFACTS / "raw_spec_failed.txt").write_text(raw + "\n\n---RETRY---\n" + raw2, encoding="utf-8")
        spec = llm.parse_json(raw2)
    spec["_meta"] = {"n_anchors": len(anchors), "stats": prof}
    return spec


def save(spec, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
