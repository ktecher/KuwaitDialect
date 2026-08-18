# Project Plan v2

**Revised 2026-08-18** after two review rounds (5 internal + 3 independent peer) and four measurements
run against real data. v1 is in git history.

**The shape changed.** v1 was a ~100h research pipeline. Every peer reviewer independently concluded it
was over-built for what it can prove: DPO and best-of-N increments are 5–8pp against a detectable
effect of ~9pp, the primary classifier is predicted to fail its own gate, and the sole rater is the
person who wrote the training data. v2 spends a few hours finding out whether the expensive path is
needed at all, then commits only if it is.

Status key: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## Locked decisions

| Decision | Choice |
|---|---|
| Runtime | WSL2 Ubuntu 24.04, Python **3.12** |
| Base model | `humain-ai/ALLaM-7B-Instruct-preview` |
| **Orthography** | **چ/گ as the author writes them.** Author's ear overrides the corpus — see the note below |
| Register | Young casual / social media |
| Real gold test set | 300 items, quarantined before generation |
| Scope | **Baseline probe first**, then lean fine-tune only if justified |

> ### ⚠️ The orthography decision and what it costs
> Measured across 38,775 real Kuwaiti tweets: **جذي outnumbers چذي 5:1** (472 vs 91), **عليج beats عليچ
> 4:1**, and **گال appears zero times** (قال: 1,000). **97.4% of real Kuwaiti tweets contain neither چ
> nor گ.**
>
> The author has chosen to trust their own ear over the corpus — legitimate, since QADI is 2018–20
> Twitter and the author writes in a register it doesn't cover. **Three consequences follow and are
> binding:**
> 1. **QADI cannot supply the majority of training responses.** Its orthography contradicts the target.
>    It remains valuable for vocabulary, morphology, syntax, length priors and classifier calibration.
> 2. **The KFI reference band cannot come from QADI.** It must be measured from the author's own writing
>    once ≥150 items exist.
> 3. **No bulk ج→چ transliteration, ever.** `جذي`→`چذي` is correct; `جميل`→`چميل` is nonsense. There is
>    no safe automatic rule.
>
> This also means the author's own writing is now *more* load-bearing, not less.

---

## Gates

### 🚪 Gate A — tokenizer handles چ / گ · ✅ **PASSED 2026-08-18**
Run on tokenizer files only, no GPU, no weights.

```
چ → [6883]   1 token        round-trip failures : 0/10
گ → [15120]  1 token        sentences with UNK  : 0/10
پ → [7485]   1 token        tokens/word         : 1.57
fertility ratio <= 2.0 on all pairs
```
**No `add_tokens`, no `resize_token_embeddings`, no `modules_to_save`, no +4.88 GiB VRAM penalty.**
Fanar also passes but is worse where it counts (2.00 tokens/word vs ALLaM's 1.57) — ALLaM wins on merit.
ڤ/ژ take 3 tokens via byte-fallback but round-trip cleanly.
Evidence: `docs/evidence/gate_a_tokenizer.md`

### 🚪 Gate B — dialect classifier recall `R` on real Kuwaiti
**Moved to Phase 0** (v1 had it in E5, contradicting our own review which put it at Step 0).
**Computed on the QADI-derived calibration set only — never on the gold set.** Calibrating on gold
means gold chose the metric, which is quarantine violation by another name.

| `R` | Consequence |
|---|---|
| ≥ 0.60 | `% top-1 == Kuwait` usable as an automatic metric |
| 0.30–0.60 | Relative only: Δ mean `P(Kuwait)`, mean Gulf-rank |
| **< 0.30** | **Retire from headline.** Likelihood contrast + human discrimination carry the weight |

Branch on the **confidence interval**, not the point estimate; if the CI crosses a boundary, take the
more conservative branch. Persist the **full 21-class probability vector** per response — argmax-only
storage makes the middle branch unrecoverable without a re-run.
Evidence: `docs/evidence/gate_b_calibration.json`

### 🚪 Gate C — training corpus is not caricature · **NEW, and non-negotiable**
Runs on the assembled corpus *before* any GPU time.

> **Why this exists.** v1's marker gate ("reject rows with <2 accepted markers") was measured against
> 38,775 real Kuwaiti rows:
> ```
> 74.7% of real Kuwaiti contains ZERO markers from the list
> gate ">=2 markers" passes  4.9% of real Kuwaiti
> KFI real corpus      :  2.28 markers / 100 words
> KFI >=2-gated subset : 11.48  ->  5.04x reality
> ```
> The gate would have **rejected 95.1% of genuine Kuwaiti** and produced training data **5× more
> marker-dense than how Kuwaitis write**. Worse, KFI — built from the same word list — would have
> scored that caricature as success. A row containing **چان** scores zero, because چان isn't on the list.

**The gate is inverted.** Markers are now a **veto**, not a requirement:
- ❌ Reject on **anti-markers**: `ازاي`/`كده` (Egyptian), `هلق`/`شو` (Levantine), `ابغى`/`ايش` (Saudi), `هسه` (Iraqi)
- ✅ **No pro-marker requirement per row.** Kuwaiti-ness lives in morphology, orthography and syntax — not a 22-word list
- 📊 Constrain at **corpus level**: aggregate KFI within ±20% of the author-derived band; individual rows may have zero markers

**Pass condition:** corpus KFI within band · length KS `D < 0.15` vs the author's own writing · zero
anti-marker survivors in a 150-row audit.

---

## Phase 0 — Foundation and the fork *(~6h, mostly unattended download)*

### P0.1 · Environment `[ ]`
Order matters — Gate A moved earlier in v1 review; it is now already done, so this is shorter.
- [ ] `nvidia-smi` in WSL (baseline) → `wsl --update` → `.wslconfig` (`memory=40GB`, `processors=20`) → restart → `nvidia-smi` again
- [ ] `uv venv --python 3.12` on **ext4**; `HF_HOME` on ext4; assert ≥60 GB free
- [ ] **Start the 14 GB ALLaM download in the background immediately** — it blocks nothing until P0.3
- [ ] Install torch from `--index-url .../cu130` (**not cu126 — no sm_120 in its arch list**)
- [ ] **Decisive check:** `torch.cuda.get_arch_list()` contains `sm_120`, capability `(12,0)`
- [ ] bf16 matmul on GPU (clearer failure signature than bitsandbytes gives)
- [ ] Rest of stack → `python -m bitsandbytes` must not fall back to CPU
- [ ] `HF_HUB_OFFLINE=1` round-trip **now**, not at ship time
- [ ] Lockfile; record arch list, driver version, model revision SHA. Pin `revision=<sha>` in every `from_pretrained` — this repo has already renamed org once

**Already verified, do not re-litigate:**
- ✅ Gate A passed
- ✅ `chat_template` present. Llama-2 format; generation prompt ends at `[/INST]` — **that is the response-masking boundary**
- ⚠️ **`pad_token` is `None`.** Set `pad_token = unk_token` (id 0). **Not `eos`** — masking pads would also mask the real `</s>` and the model would never learn to stop, surfacing much later as runaway generation

**Done when:** ALLaM loads in NF4 and completes one training step. Expect `max_memory_allocated` **≈6.8 GiB** at seq 512 / B=2 (v1 said 10–11 GB — that was the seq-1024 figure; seeing 6.8 is correct, not a misconfiguration).

### P0.2 · Author seed batch `[ ]` 🔴 blocks the probe
- [ ] **15–20 items** from `docs/WRITE-THESE.md` (★ items), in the target orthography
- [ ] These become the few-shot exemplars and the probe prompts

**Why so few:** the probe needs exemplars in *your* orthography, and QADI can't supply them.
**Effort:** 20–30 min. **Not blocked by anything.**

### P0.3 · 🎯 **THE FORK — baseline probe** `[ ]`
**Pre-register the kill rule before running. See `docs/PREREGISTRATION.md`.**
- [ ] Base ALLaM, no training: system prompt + 5 verbatim real shots from P0.2
- [ ] Generate on ~40 held-out prompts, best-of-8 @ temp 0.8
- [ ] Also run zero-shot and system-prompt-only, for the honest three-way contrast
- [ ] **Author reads 30 outputs blind** (shuffled with 10 real human items as anchors)

| Outcome | Action |
|---|---|
| Prompting reads as genuinely Kuwaiti | **Ship the prompted demo.** Stop here. ~1 week total |
| Close but flawed in a specific way | Fine-tune targeted at *that* gap only |
| Clearly short | Proceed to Phase 1 with justification |

**This is the only step that can save the other ~40 hours.** AL-QASIDA's finding — that models are
*reluctant* to emit dialect rather than unable — makes "prompting wins" the modal outcome, not a
long shot.

### P0.4 · Gate B calibration `[ ]` *(parallel with P0.1)*
- [ ] Content-filter QADI's 38,775 rows; report % removed by category
- [ ] Build calibration sets: 150 Kuwaiti · 150 MSA · 150 non-Kuwaiti Gulf
- [ ] Run Gate B, record `R` with CI, branch

### P0.5 · Author outreach `[ ]` 🐢 *(start now, 3-week latency)*
- [ ] **Fatemah Husain**, Kuwait University — `f.husain@ku.edu.kw` — Hazawi+ (7M tokens Kuwaiti narrative), and ask about KuHateCorpus in the same email
- [ ] **Nizar Habash**, NYU Abu Dhabi CAMeL Lab — `nizar.habash@nyu.edu` — likely fastest reply on corpus-sharing logistics

---

## Phase 1 — Lean fine-tune *(~25h, only if P0.3 says so)*

### P1.1 · Data tooling `[ ]`
- [ ] Schema: `messages[]` + `system` + `source` + `seed_id` + `split_group` + `topic` + `generator` + `license_consent`
- [ ] Script validation whitelisting **چ گ پ ڤ ژ by codepoint** (a naive "Arabic only" regex rejects them — the most likely silent bug in the pipeline; unit-test it on `چذي`/`گلت`)
- [ ] Dedup: exact SHA-1 · MinHash char-5-gram @ J≥0.80 · **cross-split guard @ J≥0.60 (discard the gold item, never move it)** · semantic cosine ≥0.92
- [ ] **Group-splitter on `split_group`**, stratified by topic and source, with a split-integrity assertion (no group straddles splits) that **fails loudly**
- [ ] Reserve `split_group=dpo` now, even though DPO is cut — cheap insurance

### P1.2 · Corpus `[ ]`
- [ ] Author writing: the 300 gold + 50 dev + remaining worksheet items
- [ ] Synthetic: seed-anchored, ≤4 descendants per seed, each changing topic or intent
- [ ] QADI: **vocabulary/syntax reference and hard negatives only** — not the majority of responses (orthography conflict)
- [ ] 🚪 **Gate C** before any GPU time

**Stopping rule for generation** — stop on the first of: last 200 rows add <3 new topic×marker cells · near-dup rejection >25% in a batch · 1,800 cap.

### P1.3 · SFT `[ ]`
- [ ] **`batch=8, grad_accum=2`, `max_seq_len=512`, `group_by_length=True`** — at seq 512 the VRAM total is ~10.1 GiB against ~14.5 usable, so batch 2 was over-conservative by 4×
- [ ] `paged_adamw_8bit`, nf4 + double-quant + bf16, `sdpa`, gradient checkpointing
- [ ] Loss masked to response tokens only, boundary at `[/INST]`
- [ ] `dataloader_persistent_workers=True` (**not** `persistent_workers=True` — that's the raw PyTorch kwarg and is silently ignored by TrainingArguments)
- [ ] Seeds: `seed`, `data_seed`, split seed — all recorded. Bit-exact reproducibility is **not** achievable with gradient checkpointing + non-deterministic cuDNN; claim similarity, not identity
- [ ] **Abort conditions:** halt if the canary drops >10pp at any checkpoint, or 10 sampled generations fail an eyeball check

**Runtime:** ~20–60 min at 160W (v1's 2.5–3h assumed seq 2048 and 5,000 rows).

**Cut from v1:** DPO/ORPO stage, fitted reranker, 3-rater apparatus, McNemar/Wilcoxon/Holm.
Reason: all four buy statistical claims unprovable at n=300 with one rater.

---

## Phase 2 — Evaluation *(~10h)*

**The v1 metric suite was circular.** The marker gate, KFI, and the reranker score were the same 22
words — three instruments that were one instrument, structurally blind to the caricature the gate
creates. v2 leads with two metrics that are not.

### P2.1 · Likelihood contrast `[ ]` — **primary automatic metric**
Δ(log P(kuwaiti) − log P(msa)), base vs tuned, on held-out real text.
No classifier, no rater, no prompt/reference pairs. Scored over **whole strings**, so it cannot be gamed
by sprinkling markers — the exact failure mode Gate C exists to prevent. MSA side from
`PRAli22/Arabic_dialects_to_MSA`. ~2h to build, minutes of GPU.

### P2.2 · Human discrimination test `[ ]` — **headline metric**
Interleave model output with **real, length- and topic-matched** human Kuwaiti, unlabelled. Ask:
*"written by a Kuwaiti, or by a machine?"*

This has **ground truth**, which means it **scores the rater**. A rater who identifies real text at 95%
and calls the model real at 40% has produced a hard number. A rater at chance on the real anchors has
invalidated their own session — and you know it. A naturalness Likert can tell you neither, and cannot
detect over-marking at all.

Every batch carries anchors: ~60 real human · ~30 deliberately marker-stuffed · ~15 other-Gulf · ~15 MSA.
**If the marker-stuffed anchors score as human, you have learned the rater cannot detect caricature** —
more valuable than any model number.

- [ ] Neutral framing: "rate these Arabic texts," never "rate how Kuwaiti this is"
- [ ] Stored `order_seed`; test–retest on 60 items ≥72h later
- [ ] **Recruit ≥1 rater outside the author's dialectal and social group.** The person who wrote the training data must not be the sole judge of the headline metric

### P2.3 · Capability canary `[ ]`
- [ ] 100 items from `MBZUAI/ArabicMMLU` (`All` config, `test` split) + 30 Arabic instruction + 20 English, frozen and hash-checked
- [ ] Smoke test at every checkpoint; trip at **>10pp** (at n=150 the CI is ±7–8pp, so "within 5%" is below the instrument's resolution)
- [ ] Full 1,000-item run on the final candidate only; accept ≤2pp, while stating 2pp is not distinguishable from zero

### P2.4 · Secondary `[ ]`
- [ ] Classifier Gulf% as a **tripwire only**, reported as a delta over base with paired CIs
- [ ] KFI vs the **author-derived** band — a training-conformance check, never evidence
- [ ] Anti-marker cross-Gulf audit
- [ ] Red-team spot-check: verify the model doesn't emit the sectarian/political content QADI contains

### P2.5 · Serving `[ ]`
- [ ] Best-of-8 as an **unfitted heuristic** (no 250-pair calibration — the classifier term is predicted dead)
- [ ] **Define behaviour when all 8 candidates fail: return the highest-scoring one and log it. Never return nothing.**
- [ ] Merge: load base in **bf16 on CPU**, apply adapter, merge, save. **Never `merge_and_unload()` on the 4-bit model** — that bakes quantization error in permanently. Won't fit on GPU anyway
- [ ] `serve/chat.py`; verify offline by actually disconnecting the network

---

## Ship criteria

| # | Criterion |
|---|---|
| **a** | **Headline** — model output is judged human-written at a rate approaching the real anchors in the same session, by a rater who passes the anchor checks |
| b | Likelihood contrast moves in the right direction, base vs tuned |
| c | Capability canary shows no regression whose 95% CI excludes zero |
| d | Classifier delta reported as a tripwire, with its own calibrated accuracy printed beside it |

> **Removed from v1:** the "~80–85%" target and the "60–70% Kuwaiti-specific" figure. The first is
> unmeasurable at any n available here — at n=150 the 95% CI around 80% is [72.9, 85.6], so 85% sits
> inside it, and separating them needs ~900 per arm. **The second had no source at all**; it was one
> reviewer's judgement written up as though calibrated. It is removed rather than re-anchored.
>
> **The comparable study does not calibrate this project.** Saudi-Dialect-ALLaM's 84.21% was measured
> with MARBERTv2's **`GLF` macro-label** — a *regional* class. Perfect Emirati output would also score
> ~84%. It says nothing about Kuwaiti-specificity, and we independently rejected MARBERTv2 as unusable
> (it has no classification head). Its human annotators separately flagged "over-formality and
> repetition" — i.e. our caricature failure mode, in print, happening to someone else.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Prompting alone already suffices; the fine-tune adds little** | **High** | This is now *tested first* (P0.3), not discovered at the end. A win here is a win, not a failure |
| **Model learns marker-stuffed caricature; all automatic checks pass** | **High** | Gate C (inverted gate, corpus-level band) + discrimination test + marker-stuffed anchors |
| Real data collection stalls | **High** | Author writing is now the binding constraint, and QADI can't substitute for it on orthography |
| Gate B fails (`R` < 0.30) | **High** | Pre-planned: likelihood contrast is primary, classifier is a tripwire |
| No second rater available | **High** | Discrimination test still yields a scored, self-validating number from one rater — degraded but not void |
| Author idiolect ≠ "Kuwaiti" | Medium | Second rater outside the author's social group; anchors drawn from QADI, not the author |
| Thermal throttling | Low | Checkpoint/resume; Turbo; disable sleep (Modern Standby kills the WSL2 VM) |

---

## Evidence artifacts

Claims are only as good as what's recorded. Required:

```
docs/evidence/gate_a_tokenizer.md        [x] done
docs/evidence/gate_b_calibration.json    [ ]
docs/evidence/gate_c_corpus.json         [ ]
docs/evidence/marker_gate_validation.md  [x] done (the 4.9% measurement)
docs/PREREGISTRATION.md                  [ ] before P0.3 runs
data/DATA-VERSION.md                     [ ] SHA-256 per file
eval/raw/*.jsonl                         [ ] per-item outputs; row count must match reported n
eval/ratings/*.jsonl                     [ ] raw per-rater judgements, not just aggregates
runs/manifest.json                       [ ] git SHA + data SHA + config hash + all seeds
```

A gold-hash assertion runs in the eval harness and **fails loudly** on mismatch — "burned if inspected"
is a policy with nothing behind it otherwise.

---

## Open

- [ ] Kill-rule threshold for P0.3 — must be written down *before* the probe runs
- [ ] Second rater — who, and from which social group
- [ ] `[KWT]` tag: contrastive with `[MSA]` rows, or dropped? *(Saudi-Dialect-ALLaM's best variant used an explicit dialect token, so contrastive is evidence-backed — but that finding is now weakened by the GLF-label problem above)*

Evidence for every decision: [`docs/REVIEW-FINDINGS.md`](docs/REVIEW-FINDINGS.md) ·
[`docs/DATA-SOURCES.md`](docs/DATA-SOURCES.md)
