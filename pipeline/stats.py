"""硬统计：句长、标点、词汇多样性、句首模式。全部代码计算，不让 LLM 猜。"""
import statistics
from collections import Counter

from . import textutil

BINS = ["<=6", "7-12", "13-18", "19-25", "26-35", "36+"]


def hist_to_pct(h):
    n = sum(h) or 1
    return [round(x / n, 4) for x in h]


def profile(paras):
    sents, hist, punct = [], [0] * 6, Counter()
    para_wc = []
    first_words = Counter()
    for p in paras:
        ss = textutil.split_sentences(p["text"])
        sents.extend(ss)
        for s in ss:
            hist[textutil.norm_len_bucket(textutil.word_count(s))] += 1
        punct.update(textutil.punct_profile(p["text"]))
        para_wc.append(textutil.word_count(p["text"]))
        w = textutil.words(p["text"])[:3]
        if w:
            first_words[" ".join(w)] += 1
    lens = [textutil.word_count(s) for s in sents]
    total_w = sum(lens) or 1
    total_s = len(lens) or 1
    words_all = [w for p in paras for w in textutil.words(p["text"])]
    return {
        "n_para": len(paras),
        "n_sent": total_s,
        "n_words": total_w,
        "sent_len_mean": round(statistics.mean(lens), 2),
        "sent_len_median": round(statistics.median(lens), 1),
        "sent_len_stdev": round(statistics.pstdev(lens), 2),
        "sent_len_hist_pct": hist_to_pct(hist),
        "pct_short_8": round(sum(1 for x in lens if x <= 8) / total_s, 4),
        "pct_long_30": round(sum(1 for x in lens if x >= 30) / total_s, 4),
        "question_rate": round(sum(1 for s in sents if s.rstrip('"\u201d').endswith("?")) / total_s, 4),
        "punct_per_100w": {k: round(v / total_w * 100, 2) for k, v in punct.items() if v},
        "para_wc_mean": round(statistics.mean(para_wc), 1),
        "para_wc_median": round(statistics.median(para_wc), 1),
        "ttr": round(len(set(words_all)) / max(len(words_all), 1), 4),
        "first_person_per_100w": round(sum(1 for w in words_all if w in ("i", "my", "me", "mine", "myself")) / total_w * 100, 2),
        "top_openers": first_words.most_common(15),
    }


def show(profile_dict, title):
    print(f"\n=== {title} ===")
    for k in ("n_para", "n_sent", "n_words", "sent_len_mean", "sent_len_median",
              "sent_len_stdev", "pct_short_8", "pct_long_30", "question_rate",
              "para_wc_mean", "ttr", "first_person_per_100w"):
        print(f"{k:>22}: {profile_dict[k]}")
    print(f"{'sent_len_hist_pct':>22}: {profile_dict['sent_len_hist_pct']}")
    print(f"{'punct_per_100w':>22}: {profile_dict['punct_per_100w']}")
    print(f"{'top_openers':>22}: {profile_dict['top_openers'][:8]}")
