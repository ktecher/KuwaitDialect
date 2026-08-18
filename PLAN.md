# Project Plan

Two tracks run in parallel. **Track D (data) is the critical path** — it's the long pole and it needs a
human, so it should start first and never idle. Track E (engineering) is mostly waiting on data, not on
itself.

Status key: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked

---

## The two gates that control everything

Most of the plan's ordering exists to serve these. Get them wrong and large amounts of work get redone.

### 🚪 Gate A — tokenizer handles چ / گ
**Blocks:** training config, and possibly the whole orthography decision
**Where:** E0
چ (U+0686) and گ (U+06AF) are not standard Arabic letters. If ALLaM's SentencePiece tokenizer folds them,
drops them, or emits `<unk>`, the locked orthography silently degrades every training pair.

| Result | Consequence |
|---|---|
| Round-trips clean, fertility ≤2× | Proceed unchanged |
| Round-trips, fertility 2–4× | Proceed, budget ~10–20% more tokens |
| **Fails / emits UNK** | Must `add_tokens` + `resize_token_embeddings` + `modules_to_save=["embed_tokens","lm_head"]` — **materially raises VRAM** and may force smaller batch/seq |

**Do not** normalise چ→ك as a workaround. That silently reverses a locked decision.

### 🚪 Gate B — dialect classifier recall on real Kuwaiti (`R`)
**Blocks:** E5 eval design, E6 best-of-N scorer, and every reported number
**Where:** E5a, needs real Kuwaiti from D0/D3

| `R` | Consequence |
|---|---|
| ≥ 0.60 | `% top-1 == Kuwait` becomes the primary automatic metric |
| 0.30–0.60 | Top-1 barred. Relative only: Δ mean `P(Kuwait)`, paired Wilcoxon, mean Gulf-rank |
| **< 0.30** | **Retire the classifier from headline reporting.** Kuwaiti-specificity rests entirely on KFI + human raters |

The `شلونك؟ شخبارك؟ → Saudi_Arabia @ 0.267` data point predicts the worst branch. **Plan for `R < 0.30`.**

---

## Track D — Data (start now, human-bound)

### D0 · Quarantine the gold test set 🔴 critical path
**Blocks:** Gate B, all of E5, every honest number the project produces
**Depends on:** nothing — start immediately

- [ ] Collect **300 real human Kuwaiti items** — your own writing, farouk.ai scripts, your channel's comments, your own dictation
- [ ] **Never** used as a generation seed, DPO prompt, or reranker-calibration item
- [ ] Freeze and hash the file; if it's ever inspected during development, it's burned
- [ ] Do **not** draw these from QADI (auto-labelled, ~9% wrong)

**Done when:** 300 items exist, hashed, in `data/gold/` with a recorded SHA.
**Effort:** 4–8 h · **Why 300 not 150:** at n=150 the detectable effect is 12.5pp; DPO and best-of-N gains will be 5–8pp and thus invisible.

### D1 · Worksheet batch 1
**Depends on:** nothing — start immediately, **not blocked by Gate A**
- [ ] `docs/WRITE-THESE.md` ★ items (~45) — enough to validate the whole pipeline
- [ ] Remaining ○ items (~75)
- [ ] Flag any situation that felt fake or un-Kuwaiti

**Done when:** 120 items filled, script-validated, length distribution checked against QADI.
**Effort:** ★ 60–90 min · full 3–4 h

### D2 · Author outreach 🐢 slow, start early
**Depends on:** nothing
- [ ] Email Hazawi+ authors (ACM TALLIP, doi 10.1145/3800688) — 7M tokens Kuwaiti narrative, biggest outstanding prize
- [ ] Email KuHateCorpus authors (ACM CCSE '25, doi 10.1145/3769694.3771113)

**Done when:** replies received or 3 weeks elapsed. **Effort:** 20 min, then waiting.

### D3 · QADI processing
**Depends on:** nothing (data already on disk)
- [ ] Content-filter the 38,775 rows (profanity, sectarian/political, personal insults)
- [ ] Build the **KFI marker reference band** — the "what does real Kuwaiti look like" baseline
- [ ] Build classifier calibration sets: 150 real Kuwaiti · 150 MSA · 150 non-Kuwaiti Gulf
- [ ] Record the real length distribution for the KS test

**Done when:** reference band + 3 calibration sets exist. **Effort:** 2–3 h

### D4 · Synthetic generation
**Depends on:** D0 complete (quarantine first!), D1, E1 schema frozen
- [ ] Seed-anchored batches — every prompt carries 3–5 verbatim real snippets
- [ ] ≤4 descendants per seed; each must change topic or intent, not just wording
- [ ] Lexical marker gate: auto-reject <2 accepted markers or ≥1 rejected marker
- [ ] 250 contrastive `[MSA]` pairs
- [ ] Review 20 per 200-item batch; **reject the batch if ≥3 defects**

**Done when:** ~1,400–1,800 synthetic rows pass validation. **Effort:** 6–10 h over several sessions

### D5 · DPO preference pairs
**Depends on:** D4, E3 (for on-policy half)
- [ ] Reserve **400 instructions held out of SFT** — reusing SFT rows makes DPO a no-op
- [ ] Rejected side: base model at temp 0.7, N=2, keep MSA-scoring outputs
- [ ] Composition: ~60% MSA leakage · ~20% other-Gulf variant · ~20% caricature
- [ ] ≥150 on-policy pairs (edit the SFT model's own output)

**Done when:** 600–800 pairs. **Effort:** 3–5 h

### D6 · Native blind A/B evaluation
**Depends on:** E3/E4 trained models, D0 gold set
- [ ] 3 native raters × 300 items, blinded, randomised L/R with stored `order_seed`
- [ ] 5-point naturalness + 4-way label (Kuwaiti / other Gulf / MSA / unnatural)
- [ ] Krippendorff's α ≥0.67, report Gwet's AC1 alongside

**Done when:** 900 judgements collected. **Effort:** 3–4 h per rater

---

## Track E — Engineering

### E0 · Environment + compatibility gates
**Depends on:** nothing — start immediately
- [ ] `wsl --update` (2.6.3.0 → 2.7.11), create `~/.wslconfig` (`memory=40GB`, `processors=20`)
- [ ] `uv venv --python 3.12` on **ext4**, never `/mnt/c`
- [ ] Install pinned stack (`--index-url .../cu130` — **not cu126, it has no sm_120**)
- [ ] **Decisive check:** `torch.cuda.get_arch_list()` contains `sm_120`, capability `(12,0)`
- [ ] `python -m bitsandbytes` — must not fall back to CPU
- [ ] NF4 smoke test on TinyLlama (forward + backward)
- [ ] Resolve HF auth, download ALLaM (~14 GB) to ext4 `HF_HOME`
- [ ] Load ALLaM in NF4, one step at B=2/seq=512, check `max_memory_allocated` ≈10–11 GB
- [ ] 🚪 **Gate A** — the چ/گ tokenizer check
- [ ] Optional: evaluate `QCRI/Fanar-1-9B-Instruct` as co-primary
- [ ] **Pin every working version to a lockfile and never touch it again**

**Done when:** ALLaM trains one step without OOM, and Gate A has a recorded verdict.
**Effort:** 2–4 h (mostly download) · **Risk:** low — bitsandbytes sm_120 support confirmed

### E1 · Data tooling — *before* any data is written
**Depends on:** E0 (Gate A informs script validation)
- [ ] Freeze the schema: `messages[]` + `system` + `source` + `seed_id` + `split_group` + `topic` + `register`
- [ ] JSONL validator; script rules whitelisting **چ گ پ ڤ ژ** by codepoint (a naive "Arabic only" regex rejects them — most likely silent bug in the pipeline)
- [ ] Arabizi rejection; length bounds 3–40 words
- [ ] Dedup: exact SHA-1 · MinHash char-5-gram @ Jaccard ≥0.80 · cross-split guard @ ≥0.60 · semantic cosine ≥0.92
- [ ] **Group-splitter on `split_group`**, stratified by topic and source
- [ ] Review-sample exporter

**Done when:** tooling round-trips D1's 120 items cleanly. **Effort:** 4–6 h

### E2 · Toy run — plumbing only
**Depends on:** E0, E1, D1 ★ items
- [ ] End-to-end on ~150 rough pairs: train → eval → merge → generate
- [ ] Confirm no OOM, loss decreases, chat template renders correctly

**Done when:** the loop completes without intervention. **Effort:** 2–3 h
**Do not skip.** This is the cheapest place to find plumbing bugs.

### E3 · SFT training
**Depends on:** E2 green, D4 data
- [ ] `batch=2`, `grad_accum=8`, `max_seq_len=512`, dynamic padding, gradient checkpointing
- [ ] `paged_adamw_8bit`, nf4 + double-quant + bf16 compute, `sdpa`
- [ ] Loss masked to response tokens only; render via the model's **own** chat template
- [ ] `save_steps=100`, `save_total_limit=3` — checkpoint/resume, not one monolithic run
- [ ] Log loss/perplexity to CSV

**Done when:** 3 epochs complete, val loss tracked, adapter saved.
**Effort:** 2.5–3 h runtime at 160W · 4–5 h at 80W

### E4 · DPO / ORPO preference stage
**Depends on:** E3, D5
- [ ] Train on the held-out 400-prompt slice
- [ ] Watch for length blowup and degeneration

**Done when:** adapter saved, evaluated on the same 300 gold items.
**Effort:** 1–2 h runtime · **Note:** gain may be real but statistically unprovable at n=300 (MDE ≈8.9pp). Report honestly.

### E5 · Evaluation harness
**Depends on:** D0, D3
- [ ] **E5a — 🚪 Gate B:** calibrate the classifier, measure `R`, branch accordingly
- [ ] Stage A scorer (Dialect vs MSA) · Stage B (`camelbert-mix-did-nadi`, 21-class)
- [ ] **Three arms:** base zero-shot · base 5-shot · fine-tuned (without 5-shot the before/after is dishonest)
- [ ] KFI within **±20%** of the real corpus band — *do not maximise*, overshoot is caricature
- [ ] Cross-Gulf discrimination · MSA leakage · orthography adherence · controllability
- [ ] Capability canary (100 ArabicMMLU + 30 instruction + 20 English), frozen and hash-checked
- [ ] Report real-gold and synthetic-dev **separately**, Wilson CIs, paired McNemar
- [ ] Pin decoding params (temp, top_p, seed) and report them

**Done when:** a markdown report with all three arms and CIs. **Effort:** 5–8 h

### E6 · Serving + export
**Depends on:** E5 (needs the scorer)
- [ ] Best-of-N reranking: N=8 @ temp 0.8, scored by Stage A + Stage B + KFI distance + length prior
- [ ] Fit reranker weights on 250 calibration pairs (Spearman ρ ≥0.5), **never on the gold set**
- [ ] `export/merge_adapter.py` — merge LoRA, keep raw adapter
- [ ] `serve/chat.py` — local terminal chat loop
- [ ] **Report N=1 and BoN@8 as separate rows** — BoN is a decoding gain, not a model gain

**Done when:** chat loop runs fully offline (`HF_HUB_OFFLINE=1`). **Effort:** 4–6 h

---

## Sequencing

```
NOW ──┬─ D0 gold set ──────────────────────┐ (critical path)
      ├─ D1 worksheet ────────┐            │
      ├─ D2 outreach ~~~~~~~~~~~~~~~~~~~~~~~~~~~ (slow, async)
      ├─ D3 QADI ─────────────┤            │
      └─ E0 env + 🚪Gate A ───┤            │
                              │            │
              E1 tooling ─────┤            │
                              │            │
              E2 toy run ─────┘            │
                    │                      │
              D4 synthetic ────────────────┤
                    │                      │
              E3 SFT ──┬─ D5 DPO pairs     │
                       │       │           │
                       │   E4 DPO          │
                       │       │           │
                       └───────┴─ E5 eval ─┘ 🚪Gate B
                                   │
                              E6 serving
                                   │
                              D6 native A/B
```

**Critical path:** D0 → (E5 Gate B) → E5 → D6. Everything else has slack.
**Total effort:** ~35–50 h of work, plus ~8 h of GPU runtime.

---

## Success criteria

Not a single number. Ship only when all four hold:

| # | Criterion |
|---|---|
| **a** | **Headline** — native blind rating ≥4/5 on ≥80% of the **real** gold set |
| b | Native preference for tuned over base ≥70% in blind A/B |
| c | Capability canary within 5% of base |
| d | Classifier Gulf% reported **as a delta over base**, secondary tripwire only |

> ⚠️ **"80–85%" is not a measurable target.** At n=150 the 95% CI around 80% is [72.9, 85.6] — 85% sits
> inside it. Distinguishing the two needs ~900 items per arm. State **one threshold with its CI** instead.

**Calibration:** *Saudi-Dialect-ALLaM* (arXiv 2508.13525) hit 84.21% dialect rate with the same base model,
LoRA, and 5,466 pairs — so 80–85% is realistic **for a Gulf-vs-MSA rate**. For **Kuwaiti-specific**
attribution, expect **60–70%**. Different metrics; report both, never conflate.

---

## Known risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Model learns synthetic Kuwaiti, scores well, is obviously wrong to you** | **High** | Quarantined real gold set · seed-anchored generation · lexical marker gate · 20–30% unaltered real text in train |
| Gate B fails (`R` < 0.30) | **High** | Pre-planned fallback: KFI + human raters carry Kuwaiti-specificity |
| Gate A fails (tokenizer mangles چ/گ) | Medium | `add_tokens` + `modules_to_save`; fall back to 8-bit optimizer before falling back on orthography |
| Real data collection stalls | Medium | Priority order: gold 300 → dev 50 → train seeds (floor 150). Train can lean synthetic; the test cannot |
| Capability regression undetected | Medium | Frozen canary at every checkpoint |
| DPO/BoN gains unprovable at n=300 | **High** | Accept it. Report the CI and say so rather than overclaiming |
| Thermal throttling on a laptop GPU | Low | Checkpoint/resume; Armoury Crate Turbo; disable sleep (Modern Standby kills the WSL2 VM) |

---

## Open decisions

- [ ] **Real-data volume** — methodology says ~800 real items; 150–300 was agreed. Gold 300 is non-negotiable; the rest can flex
- [ ] `[KWT]` tag — drop it, or make it contrastive with 250 `[MSA]` rows? *(recommend contrastive)*
- [ ] Evaluate Fanar-1-9B alongside ALLaM in E0?
- [ ] Does transformers 5.15 use fused/chunked cross-entropy? (would shrink the 2.45 GiB logits term)

See [`docs/REVIEW-FINDINGS.md`](docs/REVIEW-FINDINGS.md) for the evidence behind every decision here.
