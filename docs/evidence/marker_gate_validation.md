# Marker gate validation — the gate would have manufactured caricature

**Run:** 2026-08-18 · **Corpus:** 38,775 real Kuwaiti rows (QADI `label==3`, cleaned)
**Verdict: the v1 marker gate is invalid and has been replaced by Gate C.**

## The v1 rule

> D4: *"Lexical marker gate: auto-reject <2 accepted markers or ≥1 rejected marker."*

Marker list (22): شلون، وايد، هالحين، چذي، چم، گال، زين، مب، موب، شنو، وين، يبي، ترا، عيل، سالفة، دش، جذي، اكو، ماكو، شفيك، عبالي، الحين

## Measured against real Kuwaiti

```
markers per row:
  0 markers : 28,959  (74.7%)
  1 marker  :  7,926  (20.4%)
  2 markers :  1,600   (4.1%)
  3 markers :    256   (0.7%)
  4+        :     33   (0.1%)

gate ">= 1 marker"  passes 25.3% of REAL Kuwaiti
gate ">= 2 markers" passes  4.9% of REAL Kuwaiti
gate ">= 3 markers" passes  0.7% of REAL Kuwaiti

mean markers/row       :  0.31
KFI, real corpus       :  2.28 markers / 100 words
KFI, >=2-gated subset  : 11.48  ->  5.04x the real corpus
```

## What this means

**74.7% of genuine Kuwaiti tweets contain none of the 22 markers.** The `≥2` gate rejects **95.1% of
real Kuwaiti** and yields text **5.04× more marker-dense than how Kuwaitis actually write.**

Applied as specified, the gate *defines* the training distribution as caricature. And because KFI —
the metric meant to catch caricature — is built from the same word list, it would have scored that
caricature as success. The gate, the KFI metric and the reranker score were the same 22 words: three
nominally independent instruments that were one instrument.

## Real rows the gate would reject (all score 0 markers)

> اريج لو مو صايره حادثة ال👡 **چان** انتي الأولى ب قلبي بس صار اللي صار

Contains **چان** — a چ-word, unmistakably Kuwaiti — and scores zero, because `چان` isn't on the list.

> اللهم آمين ،، بو بدر والنعم وثلاثة انعام فيك وفيه ،، فعلا اخو دنيا **عسي ربي يحفظه** 🇰🇼
> ههههههه عاد تخيلت إن **خشمك** بالعرض **مو** بالطول
> الكويتية ما يأجرون طيارات من شركات أجنبية مثل غيرهم

**Kuwaiti-ness lives in morphology, orthography and syntax — not in a 22-word vocabulary list.**
The `ـچ` suffix, the negator `مو/مب`, `عسى`, `عاد`, verb forms. A lexical whitelist is the wrong
instrument for the job it was given.

## Replacement — Gate C

1. **Markers become a veto, not a requirement.** Reject on *anti-markers* (`ازاي`/`كده` Egyptian,
   `هلق`/`شو` Levantine, `ابغى`/`ايش` Saudi, `هسه` Iraqi). This is a precision filter against
   wrong-dialect contamination — what the list is genuinely good for, given QADI's 9% label noise.
2. **No pro-marker requirement per row.** Individual rows may legitimately contain zero markers.
3. **Constrain at corpus level**, not per row: aggregate KFI within ±20% of the reference band.

## Orthographic variants — measured

Checked because the project's own documents disagreed (`جذي` in one, `چذي` in another):

| Word | چ/گ form | ج/ق form |
|---|---|---|
| "like this" | چذي 91 (0.23%) | **جذي 472 (1.22%)** — 5:1 |
| "on you" (2fs) | عليچ 31 (0.08%) | عليج 131 · عليك 992 |
| "how many" | چم 40 (0.10%) | كم 3,856 (9.94%) |
| "he said" | **گال 0 (0.00%)** | قال 1,000 (2.58%) |

```
rows containing چ : 884 (2.28%)     rows containing گ : 142 (0.37%)
rows with چ or گ  : 1,021 (2.63%)   rows with NEITHER : 37,754 (97.37%)
```

**گال appears zero times in 38,775 rows.** 97.4% of real Kuwaiti tweets use neither letter.

The author has chosen to keep the چ/گ orthography, trusting their own ear over a 2018–20 Twitter
corpus. That is legitimate — but it means **QADI cannot supply the majority of training responses, and
cannot define the KFI band.** See `PLAN.md`.

## Also measured — code-switching

```
rows with a 3+ letter Latin word : 724 (1.87%)
rows with Arabizi-style digits   : 109 (0.28%)
```

A review claimed young Kuwaiti social media is "heavily Arabizi and code-switched." **Not supported by
this corpus.** The decision to reject Arabizi is well-aligned with the data. (Caveat: QADI is Twitter;
WhatsApp may differ.)

Reproduce: `scratchpad/marker_gate.py`, `scratchpad/variant_check.py`
