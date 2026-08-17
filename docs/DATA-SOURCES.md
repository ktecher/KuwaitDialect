# Kuwaiti Dialect Data — What Actually Exists

Searched 2026-08-16. Every row verified live against Hugging Face; nothing here is recalled from memory.
Datasets marked ✅ PULLED are already on disk.

---

## Headline: the brief's three named corpora are all unavailable — but a better source exists and is already downloaded

| Source named in brief | Status |
|---|---|
| **Hazawi+** | ❌ Paper only — [ACM TALLIP, doi 10.1145/3800688](https://dl.acm.org/doi/10.1145/3800688). 7M+ tokens of Kuwaiti stories/novels from forums + surveys, CAMeL-annotated. **No public download. Author contact required.** |
| **~16k Kuwaiti X/Twitter sentiment** | ❌ Not publicly hosted. Traces to ["Bridging the Kuwaiti Dialect Gap in NLP"](https://ieeexplore.ieee.org/document/10430151/) (IEEE, 2024) and related Kuwaiti sentiment work. No repo found. |
| **KuHateCorpus** | ❌ Paper only — [ACM CCSE '25, doi 10.1145/3769694.3771113](https://dl.acm.org/doi/10.1145/3769694.3771113). 2,200 annotated sentences. **No public download. Author contact required.** |

**All three need an email to the authors.** None is a `git clone` away.

---

## ✅ What I found and pulled instead: QADI Kuwait subset — 40,442 real Kuwaiti tweets

**`Abdelrahman-Rezk/Arabic_Dialect_Identification`** — the QADI dataset ([arXiv 2005.06557](https://arxiv.org/pdf/2005.06557.pdf), QCRI/HBKU), mirrored on HF under a name that doesn't mention Kuwait, which is why earlier searches missed it.

- 440,052 train rows, 18 country labels
- Label map from the dataset card: `['OM','SD','SA','KW','QA','LB','JO','SY','IQ','MA','EG','PL','YE','BH','DZ','AE','TN','LY']` → **`label == 3` is `KW`**
- **Kuwait rows: 40,442** (the largest Gulf slice; more than Saudi's 25,769 or Qatar's 29,839)
- MSA tweets were **deliberately filtered out** during construction — exactly what we want
- Label accuracy: **91.5%** on intrinsic evaluation of randomly selected tweets (labels derived from user account descriptions, not hand-annotated)

### On disk now
| File | Rows | Content |
|---|---|---|
| `data/raw/qadi_kuwait_all.txt` | 40,442 | All KW rows, @mentions and URLs stripped |
| `data/raw/qadi_kuwait.jsonl` | **38,775** | Filtered to 3–40 words — the usable set |

*(`data/raw/` is gitignored — this is local working data, not committed.)*

### Why this is a strong match for the locked decisions

**Length distribution — matches "young casual social media" almost exactly:**
```
min=2  p25=8  median=13  p75=21  p95=39  max=70   (mean 16.0 words)
```
This independently confirms the methodology review's call to **drop `max_seq_len` to 512**. Real Kuwaiti social text is short.

**Kuwaiti markers are attested at real, natural rates** (document frequency across all 40,442 rows):

| Marker | Rows | % | | Marker | Rows | % |
|---|---:|---:|---|---|---:|---:|
| بعد | 2,367 | 5.9% | | چ | **963** | **2.4%** |
| مب | 1,804 | 4.5% | | اكو | 590 | 1.5% |
| يبي | 1,478 | 3.7% | | وايد | 526 | 1.3% |
| شنو | 1,331 | 3.3% | | جذي | 522 | 1.3% |
| زين | 1,100 | 2.7% | | ماكو | 370 | 0.9% |
| شلون | 1,029 | 2.5% | | عيل | 304 | 0.8% |
| ترا | 1,028 | 2.5% | | گ | 142 | 0.4% |

**The locked orthography is real, not aspirational.** 1,100 rows contain چ or گ. Real attested examples:

> چان انتي الأولى ب قلبي · تآكلين كل شي **چدامچ** · ما **عليچ** إلا العافية · اكو **چم** واحد · هي و**سلوگيها** · نهائي ال**چامبيونز** · يآ **ضحگ** المّـسْى

Note these also show the hallmark Kuwaiti 2nd-person-feminine `ـچ` suffix (عليچ، چدامچ، معاچ) — a feature no MSA-trained model will produce on its own. This is exactly the signal worth training on, and it confirms the چ/گ tokenizer check is a genuine priority, not a theoretical one.

### ⚠️ Caveats you must weigh before using it

1. **License is undeclared.** The HF mirror's card says `Licensing Information: [Needs More Information]`. Original QADI is QCRI/HBKU research output. Treat as research use; **do not assume redistribution rights.** If this project ever ships publicly, resolve this first.
2. **Twitter ToS.** Original QADI distributed *tweet IDs* (as Twitter's terms require); this mirror contains the *text*. That repackaging is the mirror author's decision, not something this project did — but it is worth knowing.
3. **Labels are automatic, not human-verified** — 91.5% accurate, so expect ~1 in 11 rows to be non-Kuwaiti.
4. **Content filtering is mandatory before training.** The raw sample contains profanity, sectarian and political content (e.g. البدون/stateless debates), and personal insults. A demo model trained unfiltered on this will reproduce it.
5. **58% contained @mentions, 19% URLs** — already stripped in the files above, but usernames may still appear inline in text bodies.
6. **These are real people's public posts.** Fine for model training under research use; do not republish the text verbatim.

---

## Other verified datasets worth knowing about

| Dataset | Size | License | Verdict |
|---|---|---|---|
| [`AhmedSSabir/Gulf-Arabic-Tweets-2018-2020`](https://hf.co/datasets/AhmedSSabir/Gulf-Arabic-Tweets-2018-2020) | 108.1K rows, text only | **cc-by-4.0** | ⚠️ **Weak.** Inspection shows heavy MSA news/COVID contamination, no country labels, and **aggressively normalised orthography** (hamzas stripped: جاءحه for جائحة; ى→ي). The normalisation destroys the exact orthographic signal we're training on. Skip. |
| [`ebubekr53/organic-gulf-arabic-dialect-dataset`](https://hf.co/datasets/ebubekr53/organic-gulf-arabic-dialect-dataset) | 1.0K rows, has `source_country` incl. `kuwait` | cc-by-4.0 | ❌ **Do not use.** Mostly isolated words from a translation app ("الوان", "شركه", "صد"), not conversation. Worse, it contains **leaked WhatsApp logs with real names and phone numbers** (e.g. `+973 3849 9318`, named individuals). PII in training data is both an ethical problem and a memorisation/regurgitation risk. |
| [`HeshamHaroon/arabic-dialect-dpo`](https://hf.co/datasets/HeshamHaroon/arabic-dialect-dpo) | 22,538 preference triples (Egyptian 11,038 / Saudi 11,500), schema `prompt/chosen/rejected` | **apache-2.0** | ✅ **Useful as a template.** Not Kuwaiti, but it is exactly the DPO structure planned for Step 2.5 — worth copying the format, and usable as a negative/other-dialect source for the "other-Gulf variant" 20% of the rejected set. |
| [`MBZUAI/Dialectal-Arabic-MMLU`](https://hf.co/datasets/MBZUAI/Dialectal-Arabic-MMLU) | 10K–100K MCQ | — | ✅ Candidate for the **capability canary** (dialect-aware, so it tests capability without penalising dialect). |
| [`PRAli22/Arabic_dialects_to_MSA`](https://hf.co/datasets/PRAli22/Arabic_dialects_to_MSA) | 100K–1M parallel | afl-3.0 | ✅ Useful for building the **contrastive `[MSA]` pairs** (250 planned) and MSA hard negatives for DPO. |
| [`dataflare/arabic-dialect-corpus`](https://hf.co/datasets/dataflare/arabic-dialect-corpus) | 211.6K rows, 5 dialect categories, has `topic` | **mit** | ➖ Gulf-level at best, no Kuwait class. Marginal. |
| [`Gumar`](https://camel.abudhabi.nyu.edu/gumar/) (NYUAD CAMeL) | 112M words, 1,200+ Gulf forum novels, GCC incl. Kuwait | — | ⚠️ **Only n-grams are distributable.** Site states verbatim: *"Due to copyright restrictions, we are only legally allowed to share the n-grams representation of the Gumar corpus."* Full text is browse-only, via Google Form request. N-grams are useless for SFT. |

**Not found as ready HF datasets:** NADI shared-task data, MADAR corpus. Both distribute through their own project pages / LDC, not the Hub.

---

## Public social platforms — what I did and didn't do

Your brief already set this boundary correctly:

> *"manually collected, not scraped at scale (scraping at scale raises ToS/legal questions per-platform, so treat this as manual collection of a few hundred good examples, not an automated crawler)"*

I did **not** build a scraper for X, Reddit, YouTube, or Kuwaiti forums. Mass scraping breaches the ToS of every one of those platforms, and X in particular now pursues it. The good news: **QADI already gives you 38,775 Kuwaiti social-media posts collected under a research protocol**, which is far more than a hand-scrape would produce and comes without the legal exposure.

If you still want platform-native data, the legitimate routes are:
- **Reddit** — official API via PRAW is free for low-volume research use; r/Kuwait is the obvious target. This is sanctioned, not scraping.
- **X/Twitter** — API access is paid and expensive at any useful volume. Not worth it given QADI exists.
- **YouTube** — the official Data API has a comment-threads endpoint with a free quota. Sanctioned.
- **Your own farouk.ai content** — no third-party ToS applies to your own scripts and your own channel's comments. **This remains your highest-trust source** and the one no dataset can replace.

---

## Recommended data plan given all of the above

1. **Real gold test set (300)** — hand-picked by you from **your own writing and your own channel's comments**, plus your own dictation. Do **not** draw it from QADI: QADI labels are automatic and ~9% wrong, and it should stay a training resource so it can't contaminate the headline metric.
2. **Real train seeds (350–450)** — draw from QADI after content-filtering, stratified by topic, plus your own material.
3. **QADI's remaining ~38K** — too noisy for direct SFT rows, but excellent for:
   - building the **KFI marker reference band** (the "what does real Kuwaiti frequency look like" baseline — the methodology review requires this, and 38K rows gives a very tight estimate)
   - **calibrating the dialect classifier** (the Step 0 gate: run it over real Kuwaiti and measure recall)
   - measuring the **real length distribution** for the KS test against synthetic data
4. **Email the Hazawi+ and KuHateCorpus authors now** — it's a slow path and costs you one email each. Hazawi+ (7M tokens of Kuwaiti narrative) is the single biggest prize still outstanding.
