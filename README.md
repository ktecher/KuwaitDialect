# Kuwaiti Dialect LLM

Fine-tuning an open Arabic LLM with QLoRA so it responds in **Kuwaiti Gulf Arabic**. Fully local — zero
external API calls in the runtime pipeline.

> **Status: pre-build.** Research and review complete; no training code written yet. Environment verified,
> plan corrected, data sources located.

## Target setup

| | |
|---|---|
| Base model | [`humain-ai/ALLaM-7B-Instruct-preview`](https://hf.co/humain-ai/ALLaM-7B-Instruct-preview) (Apache-2.0, llama arch) |
| Method | 4-bit QLoRA (NF4), LoRA r=16 α=32 on attention + MLP, then a DPO preference stage |
| Hardware | RTX 5080 Laptop, 16 GB VRAM (sm_120), WSL2 Ubuntu 24.04 |
| Python | 3.12 |

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/REVIEW-FINDINGS.md`](docs/REVIEW-FINDINGS.md) | Full pre-build review — environment verification, six factual corrections to the original brief, pinned dependency versions for Blackwell/sm_120, VRAM math, evaluation methodology, and revised success criteria |
| [`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md) | What Kuwaiti dialect data actually exists, verified live — including which named corpora are unavailable and what to use instead |
| [`docs/WRITE-THESE.md`](docs/WRITE-THESE.md) | Native-speaker writing worksheet for producing real Kuwaiti training and evaluation data |
| [`docs/BRIEF-original.md`](docs/BRIEF-original.md) | The original project brief, kept unedited for reference |

## Key findings so far

- **bitsandbytes ships working sm_120 NF4 kernels** — 4-bit QLoRA on Blackwell is viable. Use the `cu130`
  wheel index; `cu126` has no sm_120 in its arch list.
- **MARBERTv2 is a masked-LM, not a classifier** — the original evaluation plan couldn't run as written.
  Replaced with a calibrated two-stage cascade.
- **Kuwait *is* a distinct label** in the NADI country-level dialect classifier, so Kuwaiti-vs-other-Gulf is
  measurable — but the classifier is weak and must be calibrated before it's trusted.
- **The main risk isn't technical.** It's that the model learns a synthetic approximation of Kuwaiti,
  scores well on a classifier, and is obviously wrong to a native speaker. The mitigation is a quarantined
  set of real human Kuwaiti text that never touches training.

## Layout

```
data/           # local working data — gitignored, not published
docs/           # research, review, and worksheets
```

## Licence note

Training data provenance is tracked per-record. Third-party corpora referenced in `docs/DATA-SOURCES.md`
carry their own terms — several are research-use-only or have undeclared licences, and none are
redistributed in this repository.
