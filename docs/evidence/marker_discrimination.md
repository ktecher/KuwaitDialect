# Marker discrimination — the list is mostly not Kuwaiti

**Run:** 2026-08-18 · Triggered by the author (native Kuwaiti speaker) flagging that several markers
looked Saudi rather than Kuwaiti. **He was right, and the problem is larger than that.**

**Method:** document frequency of each marker across QADI's Gulf country labels.
Ratio = KW% ÷ (highest other-Gulf %). A marker only discriminates Kuwaiti if that ratio is well above 1.

Rows: KW 40,442 · SA 25,769 · QA 29,839 · BH 25,251 · AE 25,254 · OM 18,359 · IQ 14,883

## Result

| marker | KW% | SA% | QA% | BH% | AE% | OM% | IQ% | ratio | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **چذي** | 0.27 | 0.00 | 0.06 | 0.05 | 0.00 | 0.00 | 0.00 | **4.73** | ✅ **KUWAITI** |
| **جذي** | 1.29 | 0.01 | 0.20 | 0.60 | 0.25 | 0.00 | 0.02 | **2.14** | ✅ **KUWAITI** |
| چم | 0.11 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.07 | 1.62 | 🟡 leans KW |
| دش | 0.69 | 0.21 | 0.36 | 0.54 | 0.27 | 0.23 | 0.24 | 1.27 | ➖ pan-Gulf |
| يبي | 3.65 | 2.52 | 3.04 | 3.14 | 1.75 | 1.15 | 1.65 | 1.16 | ➖ pan-Gulf |
| زين | 2.72 | 1.66 | 2.46 | 2.60 | 2.18 | 2.54 | 2.31 | 1.05 | ➖ pan-Gulf |
| وايد | 1.30 | 0.03 | 0.94 | 0.90 | **1.34** | 0.32 | 0.05 | 0.97 | ➖ pan-Gulf |
| شفيك | 0.26 | 0.14 | 0.27 | 0.15 | 0.09 | 0.05 | 0.03 | 0.96 | ➖ pan-Gulf |
| وين | 3.85 | 2.83 | 4.19 | 3.43 | 3.30 | 3.03 | 3.86 | 0.92 | ➖ pan-Gulf |
| الحين | 2.51 | 1.36 | 2.78 | 1.40 | 2.87 | 2.03 | 0.19 | 0.87 | ➖ pan-Gulf |
| شلون | 2.54 | 0.47 | 1.64 | 1.42 | 0.45 | 0.54 | **3.00** | 0.85 | ➖ pan-Gulf |
| عيل | 0.75 | 0.07 | 0.45 | 0.63 | 0.93 | 0.50 | 0.11 | 0.80 | ➖ pan-Gulf |
| ترا | 2.54 | 2.77 | 3.05 | 1.56 | 3.24 | 3.44 | 2.03 | 0.74 | ➖ pan-Gulf |
| شنو | 3.29 | 0.23 | 1.26 | 1.72 | 0.19 | 0.17 | **5.33** | 0.62 | ❌ Iraqi |
| سالفة | 0.44 | 0.33 | 0.47 | 0.55 | 0.39 | 0.54 | **0.80** | 0.55 | ❌ Iraqi |
| ماكو | 0.91 | 0.02 | 0.07 | 0.04 | 0.02 | 0.01 | **1.98** | 0.46 | ❌ Iraqi |
| عبالي | 0.06 | 0.00 | 0.03 | 0.10 | 0.04 | 0.08 | **0.15** | 0.43 | ❌ Iraqi |
| مب | 4.46 | 4.52 | 8.15 | 4.23 | **10.87** | 4.68 | 2.99 | 0.41 | ❌ Emirati |
| **هالحين** | 0.06 | **0.16** | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.35 | ❌ **Saudi** |
| اكو | 1.46 | 0.25 | 0.26 | 0.36 | 0.30 | 0.17 | **4.54** | 0.32 | ❌ Iraqi |
| موب | 0.10 | 0.31 | 0.12 | 0.14 | **0.37** | 0.03 | 0.27 | 0.27 | ❌ Emirati |
| **گال** | **0.00** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.20** | 0.00 | ❌ **Iraqi** |

## Summary

| Verdict | Count |
|---|---|
| ✅ Genuinely Kuwait-discriminative | **2** (چذي، جذي) |
| 🟡 Leans Kuwaiti | 1 (چم) |
| ➖ Pan-Gulf — no discriminative power | 10 |
| ❌ Belongs to another dialect | **9** |

**The author's specific call was correct: `هالحين` is Saudi-leaning** (0.16% SA vs 0.06% KW).
So is `مب` — Emirati at 10.87%, more than double Kuwaiti's 4.46%. And `گال`, which the KFI list
treated as a Kuwaiti marker, appears **zero times** in 40,442 Kuwaiti rows and is Iraqi.

## Why this matters

1. **KFI as designed measures almost nothing about Kuwaiti-ness.** 19 of 22 markers are pan-Gulf or
   another dialect entirely. A model maximising this list would drift toward generic Gulf with Iraqi
   and Emirati colouring — the exact failure the metric was supposed to detect.
2. **It confirms the structural finding independently.** Kuwaiti-ness is carried by morphology,
   orthography and syntax — not vocabulary. The only strong signals in the whole list are the چ/ج
   spellings of "like this," which are *orthographic*.
3. **It vindicates inverting the gate (Gate C).** A pro-marker requirement built on this list would
   have selected for pan-Gulf caricature. Anti-markers as a veto remain sound, but the anti-marker list
   needs the same native review before use.
4. **Cross-Gulf discrimination cannot be built from this list.** That metric needs redesign around
   morphology and orthography.

## Status

🚧 **The marker list is UNVALIDATED and must not be used as-is.** Native review deferred by the author.
Anything depending on it — KFI band, Gate C anti-markers, best-of-N marker term, cross-Gulf
discrimination — is blocked on that review.

When the review happens, this table is the starting point: the ratio column shows which candidates are
worth keeping and which are actively misleading.

Reproduce: `scratchpad/marker_discrim.py`
