# Baseline probe (P0.3)

**The fork.** Does prompting alone already produce acceptable Kuwaiti? If yes, ~35 hours of
fine-tuning work is unnecessary.

Read [`../docs/PREREGISTRATION.md`](../docs/PREREGISTRATION.md) first. It must be frozen before any of
this runs, or the decision rule is worthless.

## Order

```bash
# 1. Generate three arms (needs 15-20 author-written seed items)
python probe/run_probe.py --seeds probe/seeds.jsonl --out eval/raw/probe.jsonl

# 2. Assemble the blind rating set (60 items: 30 model, 15 real, 8 caricature, 7 floor)
python probe/build_rating_set.py \
    --probe eval/raw/probe.jsonl \
    --qadi data/raw/qadi_kuwait.jsonl \
    --out eval/raw/rating_set.jsonl

# 3. Rate blind, then get the pre-registered verdict automatically
python probe/rate.py --set eval/raw/rating_set.jsonl --out eval/raw/probe_ratings.jsonl

# re-score later without re-rating
python probe/rate.py --score eval/raw/probe_ratings.jsonl
```

## What you need to provide

`probe/seeds.jsonl` — **15–20 items**, one JSON object per line:

```json
{"id": "KW-001", "topic": "greetings", "user": "...", "reply": "..."}
```

Both `user` and `reply` in Kuwaiti, in your orthography. See
[`../docs/WRITE-THESE.md`](../docs/WRITE-THESE.md) for the situations — the ★ items are enough.
Write in the worksheet and I'll convert it to JSONL; you don't need to hand-write JSON.

5 items become few-shot exemplars, the rest become eval prompts.

## Design notes

**The rating task is discrimination, not a naturalness Likert.** You're asked *"Kuwaiti person, or
machine?"* — binary. This matters because the task has **ground truth**, so it scores the rater as well
as the model. A naturalness rating cannot do that, and cannot detect over-marking at all.

**Rater validity is checked before any model number.** If you don't identify the real human anchors at
≥80%, the session is declared void and no model result is reported. That's deliberate.

**Real anchors come from QADI, not your own writing** — you'd recognise your own text and the anchor
would be void.

**The caricature anchors are the interesting ones.** If marker-stuffed text reads as human to you, that's
a finding about the measurement instrument, and it's worth more than any model score.

**The best-of-N scorer does not use the Kuwaiti marker list.** That list is UNVALIDATED — only 2 of 22
markers actually discriminate Kuwaiti (see
[`../docs/evidence/marker_discrimination.md`](../docs/evidence/marker_discrimination.md)). The scorer
uses MSA function words and a length prior instead, which are independent of it.

## Pinned decoding

`temperature=0.8` · `top_p=0.9` · `seed=42` · `max_new_tokens=120` · `best-of-8`

Recorded in the output file. Changing any of them invalidates the probe.
