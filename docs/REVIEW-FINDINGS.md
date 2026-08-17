# Pre-Build Review — Kuwaiti Dialect LLM

Review date: 2026-08-16. No code written yet. Five specialist reviews + direct verification against
Hugging Face and the local machine. Every claim below marked ✅ was checked directly, not recalled.

**Verdict: GO. Stay on WSL2. The brief is structurally sound — keep its spine, fix six factual errors,
and adopt four additions the user has approved.**

---

## 1. Environment — verified ready

| Item | Finding | Status |
|---|---|---|
| Laptop | ASUS ROG Strix SCAR 16 (G635LW) | — |
| CPU | Intel Core Ultra 9 275HX, 24C/24T | ✅ ample |
| RAM | 64 GB DDR5-5600 (2×32, dual channel, at rated speed) | ✅ ample |
| GPU | RTX 5080 **Laptop**, 16303 MiB GDDR7, **sm_120** (cc 12.0) | ✅ the binding constraint |
| GPU power | default 80 W · **currently enforced 160 W** · max 175 W (Dynamic Boost) | ✅ already on a good profile |
| Power state | On AC, Windows scheme = Performance, Armoury Crate running | ✅ |
| WSL2 | 2.6.3.0, kernel 6.6.87.2, Ubuntu 24.04.4 LTS | ✅ GPU passthrough confirmed working |
| VRAM visible in WSL | full 16303 MiB, `/usr/lib/wsl/lib` populated | ✅ |
| Disk | 837 GB free on WSL ext4 root | ✅ |
| Toolchain in WSL | Python 3.12.3, `uv`, `git`. No `nvcc` (not needed — wheels bundle CUDA) | ✅ |
| Git repo | initialized on `main`, `.gitignore` in place, brief vendored to `docs/` | ✅ done |

**Desktop VRAM reserve is ~0 MiB** — this is a hybrid/Optimus laptop, the compositor sits on the iGPU,
so the dGPU is essentially fully free for compute.

### Actions before first run
- `wsl --update` (2.6.3.0 → 2.7.11 stable; picks up GPU-library overlay + virtiofs fixes)
- Create `~/.wslconfig` — none exists, WSL is defaulting to 31 GB of 64 GB:
  `memory=40GB`, `processors=20`, `swap=8GB`
- Armoury Crate → **Turbo/Manual**; GPU Mode must **not** be Eco (that disables the dGPU entirely)
- **Disable sleep** — Modern Standby suspends the WSL2 VM and kills a running job
- Keep venv and `HF_HOME` on ext4, **never** `/mnt/c` (~10× penalty on the HF cache's many-small-file pattern)
- **Never** install a Linux NVIDIA display driver inside WSL — it overwrites `/usr/lib/wsl/lib` and breaks CUDA

---

## 2. Six factual corrections to the brief

| # | Brief says | Reality | Impact |
|---|---|---|---|
| C1 | `ALLaM-AI/ALLaM-7B-Instruct-preview` | Org renamed → **`humain-ai/ALLaM-7B-Instruct-preview`** ✅ (old ID still redirects). Apache-2.0, llama arch, 7.0B, ungated, 277K downloads | Use new ID |
| C2 | Falcon-Arabic-7B is fallback #1 | ❌ **No open weights exist.** TII ships it as a closed chat product; only eval-detail *datasets* are on HF. Cannot satisfy "fully local" | **Fallback is invalid — remove** |
| C3 | Jais-13B-Chat is fallback #2 | ✅ exists but **gated**, custom `jais` arch needing `trust_remote_code`, and 13B won't fit for training in 16 GB | Deprioritize; worst possible loader-compat target |
| C4 | Score dialect with MARBERTv2 | ❌ **`UBC-NLP/MARBERTv2` is `AutoModelForMaskedLM` / fill-mask — it has no classification head and no dialect labels** ✅ | **Steps 3 AND 4 both hard-depend on this. Must swap.** |
| C5 | Test Unsloth in Step 0 | ❌ **Drop it.** Not a Blackwell problem — a staleness problem. `unsloth` pins `torch<2.12`, `datasets<4.4.0`, which hard-conflicts with `trl` needing `datasets>=4.7.0` | Test only transformers+peft+trl+bnb |
| C6 | No classifier separates Kuwaiti from other Gulf | ❌ **Too pessimistic.** `CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi` is 21-class country-level and **`"6": "Kuwait"` is a distinct label** ✅, alongside Bahrain/Qatar/Saudi/UAE/Oman | Kuwaiti-specificity *is* measurable |

### On C4 — what to use instead
Verified candidates, all `AutoModelForSequenceClassification`, all Apache-2.0, all ungated:

| Model | Labels | Note |
|---|---|---|
| `CAMeL-Lab/bert-base-arabic-camelbert-mix-did-nadi` ✅ | 21 countries **incl. Kuwait** | Only Kuwait-aware option. But Oct 2021, 9.9K downloads, and its own card reportedly returns `Saudi_Arabia @ 0.267` for `شلونك؟ شخبارك؟` — **weak** |
| `CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus26` ✅ | 26 cities + MSA. Gulf: DOH, RIY, JED, MUS, BAS. **No Kuwait** | Stronger/newer (Oct 2025, 97.7K downloads), but Kuwait-blind |
| `IbrahimAmin/marbertv2-arabic-written-dialect-classifier` ✅ | MARBERTv2-based, 162M, has a real head | Alternative |

**Neither is trustworthy out of the box.** Add a **classifier calibration gate to Step 0**: run the chosen
model over ~200 real Kuwaiti and ~200 MSA sentences and report *its own accuracy on your distribution*.
If it can't separate them, Steps 3–4 measure noise and must be redesigned before anything is built on them.

---

## 3. Stack — pinned, validated for Python 3.12

**Python 3.12 confirmed — no hard blocker.** Every package is `py3-none-any`, `abi3`, or ships an explicit
cp312 wheel.

| Component | Pin | Note |
|---|---|---|
| Python | **3.12** | `uv venv --python 3.12`; Ubuntu 24.04 ships 3.12.3 |
| PyTorch | **2.13.0+cu130** | `--index-url https://download.pytorch.org/whl/cu130` |
| bitsandbytes | **0.50.1** | ✅ **ships real sm_120 NF4 kernels** — the #1 project risk is retired |
| transformers | 5.15.0 | v5 rename: `from_pretrained(..., dtype=)` not `torch_dtype=` |
| peft / trl / accelerate / datasets | 0.20.0 / 1.10.0 / 1.14.0 / 5.0.1 | |
| Unsloth | **do not install** | see C5 |

⚠️ **Do not use the cu126 index** — PyTorch's arch table for 12.6 is `{50,60,70,75,80,86,90}`, **no sm_120**.
Wheels ship SASS-only with no PTX, so an absent arch means hard failure. cu130 includes 120.

**The single decisive Step 0 check:**
```python
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list(), torch.cuda.get_device_capability())"
# must contain sm_120 and print (12, 0)
```

---

## 4. VRAM and wall-clock

ALLaM config verified ✅: 32 layers, hidden 4096, intermediate 11008, **`num_key_value_heads: 32` (no GQA)**,
**`vocab_size: 64000`**, `tie_word_embeddings: false`, `max_position_embeddings: 4096`.

**The OOM risk is the 64k-vocab logits tensor, not the weights.**

| | batch 2 @ 2048 | batch 4 @ 2048 |
|---|---|---|
| NF4 weights + untied embed/lm_head | 4.09 GiB | 4.09 |
| LoRA + optimizer + checkpointing + recompute | 1.72 | 3.07 |
| **Logits + fp32 CE upcast** | **2.45** | **4.90** |
| Context + workspace + fragmentation | 1.40 | 1.60 |
| **Total** | **≈ 9.7 GiB** | **≈ 13.7 GiB** |

Usable under WSL2 ≈ 14.5–15.3 GiB. **Batch 4 @ 2048 leaves under 1 GiB margin — do not use it.**

Recommended: `batch=2`, `grad_accum=8` (effective 16), `max_seq_len=1024` (raise only after profiling),
`gradient_checkpointing=True`, `optim="paged_adamw_8bit"`, nf4 + double-quant + bf16 compute,
`attn_implementation="sdpa"`, `dataloader_num_workers=8`, **`persistent_workers=True`**,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `save_steps=100`, `save_total_limit=3`.

### Wall-clock, 3 epochs × 5,000 pairs (estimates, not benchmarks)

| Config | @ 80 W | @ 160–175 W |
|---|---|---|
| seq 2048, fully packed | 26–34 h | 15–20 h |
| **Realistic — dynamic padding, ~280 tok avg** | 4–5 h | **2.5–3 h** |

**`max_seq_len` is a cap, not a length.** Casual Kuwaiti chat pairs are short; with dynamic padding this
finishes in an afternoon. Measure real token lengths before assuming 2048.

**Power/thermals are a throughput multiplier, not a bottleneck.** Expect 10–20% below peak boost after
~15 min sustained. Mitigation is checkpoint-and-resume, not a code change.

---

## 5. WSL2 vs native Windows — stay on WSL2

| Dimension | Finding |
|---|---|
| Throughput | Single-digit % penalty on GPU-bound QLoRA. The widely-cited "20–40%" figures are SEO blogspam. The one real head-to-head traced its gap to DataLoader worker creation — fixed by `persistent_workers=True` |
| **VRAM** | ⚠️ **WSL2 genuinely costs ~0.5–1 GiB more than native Windows** — the only real disadvantage. Doesn't bind at batch 2; *would* decide it if you needed batch 4 |
| Libraries | Delta is much smaller than in 2025. Nothing in the chosen stack is Linux-only. Linux still wins on first-party Triton and a flash-attn build that succeeds |
| Power profile | **Identical.** WSL2 uses the host WDDM driver; `libcuda.so.1` is a shim. Armoury Crate/Dynamic Boost apply the same. No known failure to engage the high-power profile |

Switch to native Windows **only if** you later need batch 4.

> ⚠️ Note: one review claimed "there is no WSL 3" and that search results describing it are fabricated.
> Treat any plan built on a WSL 3 release as unverified.

---

## 6. The core methodological risk

> **The model learns *Claude's approximation* of Kuwaiti — Gulf-flavoured MSA with sprinkled markers —
> scores 82% on the classifier, and a native speaker reads five outputs and knows it's wrong.
> Every metric in the brief as written would report success.**

Training data is Claude-generated from user seeds; the eval split is carved from that *same* batch; the
classifier is a third proxy on top. **No real human ground truth enters the pipeline anywhere.**

Supporting evidence: AL-QASIDA (ACL Findings 2025, arXiv 2412.04193) evaluated 9 LLMs across 8 dialects —
**Kuwait among them** — and found monolingual fidelity scores almost all **below 50%**, concluding LLMs
"do not produce DA as well as they understand it… because they are reluctant to generate DA."

The favourable reframe: **the model doesn't lack Kuwaiti fluency, it has a learned preference against
emitting it.** You're re-weighting a prior, not teaching a language — which is why a few thousand pairs is
a plausible budget.

### Mitigations (mandatory)
1. **Quarantine 150–300 real human Kuwaiti items before generation begins** — never a generation seed,
   never paraphrased into training. This becomes the headline metric. *(User committed ✅)*
2. **Seed-anchored generation** — every batch prompt carries 3–5 verbatim real snippets; outputs must
   reuse attested markers. The difference between "write Kuwaiti" and "recombine this Kuwaiti."
3. **Deterministic lexical gate** — accept/reject marker lists (accept شلون، جذي، وايد، عبالي، الحين، زين;
   reject ازاي، كده، هلق، شو). Auto-reject pairs with <2 accepted or ≥1 rejected marker. Catches
   Egyptian/Levantine leakage no classifier will.
4. **Keep 20–30% of training rows as unaltered real text** (take a real utterance as the response,
   back-write the instruction) to anchor against collapse toward Claude's register.
5. **Native review with a kill rule** — review 10% of each batch; if reject rate >20%, discard and regenerate.

---

## 7. Locked decisions

| Decision | Choice |
|---|---|
| Runtime | WSL2 Ubuntu 24.04 |
| Python | **3.12** |
| Orthography | **Arabic script, natural spelling incl. چ/گ** (چذي، گلت). Arabizi/numerals (3،7،9) **rejected** |
| Register | **Young casual / social media** (X, Instagram, WhatsApp) — short and informal |
| Real held-out set | **150–300 real human items**, quarantined pre-generation |
| Git | initialized, `.gitignore` in place |

⚠️ **New Step 0 check created by the orthography decision:** چ (U+0686) and گ (U+06AF) are non-standard
Arabic letters. Verify the ALLaM tokenizer doesn't fragment them into many tokens or hit UNK **before**
generating data — it would silently degrade every training pair.

### Approved additions (all four in scope)
1. **Native blind A/B eval** — the acceptance gate. 50 held-out prompts, tuned vs base, shuffled and
   label-stripped, rated on Kuwaiti-ness 1–5, Kuwaiti-vs-generic-Gulf binary, usefulness 1–5.
   Strongest signal available; classifier demoted to regression tripwire.
2. **Capability canary** — frozen ~80 items (English, MSA, multi-step, reasoning) run at every checkpoint
   against base and tuned. Guards the most likely silent failure: a model that answers everything with a
   fluent Kuwaiti non-sequitur scores ~100% dialect accuracy and passes.
3. **DPO/ORPO after SFT (new Step 2.5)** — rated the single largest available improvement. Near-zero extra
   data cost: *rejected* = the base model's own MSA output, *chosen* = your Kuwaiti pair. Every serious
   dialect-generation system ends with a preference stage, because "prefer dialect X" is natively a
   preference objective and only awkwardly an SFT objective.
4. **Best-of-N reranking replaces the Step 4 retry** — sample N=4–8 at temp ~0.9, score each with
   classifier + marker count, return argmax. The brief's single identical re-ask doesn't shift a *prior*.

### Schema — must change before any data is written
The brief's `{instruction, response, tag}` is single-turn with no system field, but Step 5 is a **chat loop**
and Step 4 feeds output back as a second turn. Fixing this after 3–5k hand-reviewed pairs means re-cutting
all of them.

```json
{"id":"...","messages":[{"role":"system","content":"..."},
                        {"role":"user","content":"..."},
                        {"role":"assistant","content":"..."}],
 "tag":"KWT","source":"real|synthetic|user_written","seed_id":"...",
 "topic":"...","register":"...","turns":1}
```
Render via the base model's **own chat template** at train time — mismatched templates between train and
serve is a top-three cause of "it trained fine but talks like garbage."

Also: **group-split by `seed_id`**, not randomly — 3–5k rows descending from a few hundred seeds means a
random 85/15 split puts near-paraphrases on both sides and validates nothing. Add dedup (exact → normalized
→ near-dup via char n-gram/MinHash); the brief has none, and batch LLM generation reliably repeats itself.

### `[KWT]` tag — reconsider
Prepended to *every* instruction it is a constant, not a control: it carries no contrast, so the model either
binds behaviour to a token that must then always be injected at serve time (silent breakage if Step 5 forgets)
or ignores it. Either drop it in favour of the system prompt, or make it a real switch by including untagged
MSA-response rows.

---

## 8. Success criteria — replace the single number

The brief's "80–85% dialect accuracy" is gameable by sprinkling شلون/وايد into otherwise-MSA text, which is
exactly the degenerate solution a narrow fine-tune finds. It also has no floor: if untuned ALLaM already
scores 70% Gulf, then 82% is a 12-point delta, not an 82% win.

**Ship criteria (tuple, native judgment primary):**
- (a) native blind rating ≥4/5 on ≥80% of the **real** held-out set ← **headline**
- (b) native preference for tuned over base ≥70% in blind A/B
- (c) capability canary within 5% of base
- (d) classifier Gulf% reported **as a delta over base**, secondary tripwire only

**Calibration.** The brief's "comparable study" was located: **Saudi-Dialect-ALLaM** (arXiv 2508.13525) —
same base model, LoRA, 5,466 pairs, Najdi/Hijazi. Dialect rate **47.97% → 84.21%**, MSA leakage
32.63% → 6.21%. So **80–85% is well-calibrated for a Gulf-vs-MSA rate**. But for *Kuwaiti-specific
attribution*, expect **60–70%** with a large delta over base. Different metrics — report both, don't conflate.

**Add a third eval arm: base 5-shot.** Without it the before/after chart is dishonest — it attributes to
training what may be attributable to prompting.

> ⚠️ **The "80–85%" range is statistically meaningless at any sample size you will have.** At n=150 the 95%
> CI around 80% is [72.9, 85.6] — **85% sits inside it.** At n=300 the intervals around 80% and 85% still
> overlap. Distinguishing 80% from 85% at 80% power needs **~900 items per arm**; at n=600 power is only
> ~63%. This holds for the brief's original 450–750 eval split too. **Drop the range framing entirely** — it
> claims a resolution the data cannot supply. State **one threshold with its confidence interval** instead.

---

## 9. Data sources — the actual bottleneck

⚠️ **None of the three Kuwaiti corpora named in the brief are downloadable from Hugging Face.**

| Source | Status |
|---|---|
| Hazawi+ | Paper only (ACM TALLIP, Mar 2026, doi 10.1145/3800688). 7M+ tokens. Author contact required. Largest and most recent — best target |
| ~16k Kuwaiti X/Twitter sentiment | Not on HF. Academic paper only, access unclear |
| KuHateCorpus | Paper only (ACM CCSE '25, doi 10.1145/3769694.3771113). 2,200 sentences. Author contact required |

All three need outreach. **This makes the user's own material — their farouk.ai Gulf-dialect scripts,
real audience comments, and their own writing — the primary and most reliable source**, not a supplement.

**Seed diversity, not row count, is the binding constraint.** 1,200 pairs from 400 diverse real seeds beats
5,000 paraphrases of 200 seeds. Consider halving the 3–5k target and spending the saved time on the real
held-out set and native evaluation. 3–5k hand-reviewed pairs is realistically 20–40 hours of work.

---

## 9b. Data & evaluation methodology — the quantitative plan

### Dataset size: cut the target
**3,000–5,000 pairs is over-specified.** LIMA (arXiv 2305.11206) reached competitive style alignment from
**1,000** curated examples, and fine-tuning is a poor knowledge vehicle anyway (Ovadia et al., EMNLP 2024:
RAG 0.875 vs FT 0.504 on new knowledge). The binding constraint is **effective diversity** — 4,000 rows from
400 seeds carry ~400 units of information.

**Recommended composition — 2,700 records:**

| Bucket | n | Rule |
|---|---|---|
| **REAL gold test** | **300** | Quarantined before generation; never a seed, never a DPO prompt |
| REAL train seeds | 350–450 | All real, all train |
| REAL dev | 50 | Loss / early-stopping only |
| Synthetic descendants | 1,400–1,800 | **≤4 per seed**; each must change topic or intent, not just wording |
| Contrastive `[MSA]` pairs | 250 | Makes the tag a real control |
| DPO instruction slice | 400 prompts | **Held out of SFT** |
| Reranker calibration | 250 | Generated later, human-scored |

> ⚠️ **This raises your real-data commitment from "150–300" to ~800 real items** (300 gold + 450 seeds + 50 dev).
> If real collection stalls, the priority order is non-negotiable: **gold test 300 first, dev 50 second,
> train seeds take the remainder (floor 150).** Train can lean synthetic; the test cannot. Real fraction of
> *train* must not fall below 20%.

**Honest ceiling:** this is distillation, so the fine-tune's Kuwaiti fluency is bounded by Claude's, not by
native fluency. Only the real gold set can detect that ceiling.

### Splitting — the leak is near-total
At 4 descendants per seed, P(an eval item has a sibling in train) ≈ 1 − 0.15³ = **99.6%**. Group-split on
`split_group` (root seed id), stratified by `topic` and `source` (`StratifiedGroupKFold`). The 15% grouped
split is a **dev** set — report it, never headline it.

### Dedup and script validation
Normalise **for hashing only**, never mutate stored text: NFKC, strip diacritics/tatweel, أ/إ/آ→ا, ى→ي, ة→ه,
collapse repeats >2. **Do not fold چ→ج or گ→ك in stored text — that is the signal you're training on.**

- Exact: SHA-1 on normalised `instruction+response`
- Near-dup: MinHash, char 5-grams, 128 perms, drop at **Jaccard ≥ 0.80** within-split
- **Cross-split guard (stricter): train↔gold at Jaccard ≥ 0.60 → discard the *gold* item**, never move it
- Semantic: multilingual embedding cosine **≥ 0.92** → manual merge (essential — short texts defeat n-grams)

**Script rules for the locked orthography:** whitelist **چ گ پ ڤ ژ** (U+0686, 06AF, 067E, 06A4, 0698); emoji
≤2 per response. **Reject Arabizi** via `[A-Za-z0-9]*[2379][A-Za-z0-9]*` (len ≥3, no adjacent Arabic) and
ASCII-letter ratio >0.05 outside a brand whitelist. Reject repeated-3-gram ratio ≥0.20 and refusal phrases
(`لا أستطيع`، `بصفتي`، `As an AI`).

### Length — drop max_seq_len to 512
Real casual Kuwaiti is short: **response 3–40 words (hard cap 60), instruction 3–25**. Better than bounds,
match the *distribution*: two-sample KS between synthetic and real lengths, require **D < 0.15**.
**Consequence: `max_seq_len` 2048 → 512.** Costs nothing at this length distribution and buys large VRAM
headroom (the logits term scales linearly with sequence length).

### Human review sample size
To estimate defect rate to ±5pp at 95% with p≈0.10: **n = 150**. Per-batch acceptance: **review 20 of each
200-pair batch, reject the batch if ≥3 defects.** If 100 sampled items are clean, the rule of three gives a
95% upper bound of 3% on the true defect rate.

### Metrics — a calibration-gated cascade
- **Stage A (coarse, reliable):** Dialect-vs-MSA. Expect ≥90%. This is your MSA-leakage number.
- **Stage B (fine, unproven):** `camelbert-mix-did-nadi` 21-class → `P(Kuwait)`, `P(Gulf)`.

**Calibrate Stage B before trusting it** — 300 real gold (→ Kuwait recall *R*), 150 MSA (false-Kuwait rate),
150 non-Kuwaiti Gulf (cross-Gulf specificity). Then route:

| R | Rule |
|---|---|
| **≥ 0.60** | Use `% top-1 == Kuwait` as primary automatic metric, with Wilson CIs |
| **0.30–0.60** | Top-1 barred. **Relative only**: Δ mean `P(Kuwait)` base vs tuned, paired **Wilcoxon**, plus mean rank of Kuwait among the 7 Gulf classes (target ≤2) |
| **< 0.30** | **Retire Stage B from headline.** Monitoring only; carry Kuwaiti-specificity on KFI + human raters |

**The fix that makes a weak classifier usable:** report every model number **as a fraction of the human
reference** measured on the same 300 real items ("tuned reaches 78% of the human corpus's mean P(Kuwait)").
The classifier's bias applies to both sides and largely cancels. **Never fine-tune the classifier on your own
generated data** — that's circular. Exclude items <4 words from all classifier metrics and report the
excluded fraction.

**Metric suite:**
1. **HEADLINE — Native Kuwaiti Naturalness Rate (NKNR):** % of the 300 real-gold responses rated ≥4/5 by native raters
2. **Kuwaiti Feature Incidence (KFI):** markers per 100 tokens (شلون، وايد، هالحين، چذي، گال، زين، مو/مب، شنو، وين، يبي، حق، ترا، عيل، سالفة، دشّ). **Do not maximise — target within ±20% of your real corpus's KFI.** Overshoot is dialect caricature, a common failure of dialect fine-tunes
3. **Cross-Gulf discrimination:** ≥1 Kuwait-preferential marker and 0 competing-variant markers (شحالك ≈ Emirati; ابغى/ايش ≈ Saudi)
4. **MSA leakage:** 0 dialect markers AND ≥1 MSA-only marker (سوف، لقد، الذي، ليس، إنّ)
5. **Orthography adherence:** % using چ/گ where a marker word calls for it — did the locked orthography transfer?
6. **Controllability:** %Gulf under `[KWT]` minus under `[MSA]`; target ≥60pp

**Reporting rule:** base vs tuned side by side, 95% Wilson CIs, **paired McNemar** (same prompts — never a
two-sample test). Report real-gold and synthetic-dev separately; expect real to land **5–15pp lower**.

### Statistics — take 300, and freeze it

95% Wilson CI at p̂ = 0.80: **n=150 → [72.9, 85.6], ±6.4pp** · **n=300 → [75.1, 84.1], ±4.5pp**

Paired McNemar MDE at 80% power: **n=150 → 12.5pp** · **n=300 → 8.9pp** (→ 10.2pp with Holm over 3 comparisons)

The base-vs-tuned effect will be enormous (≈15% → ≈80%; McNemar z ≈ 7.8 even at n=100) — either size proves
"the fine-tune worked." But you now have **four variants** (base / SFT / SFT+DPO / +BoN), and the SFT→DPO and
N=1→BoN increments will plausibly be **5–8pp — invisible at n=150, marginal at n=300.** Take 300, freeze it,
and **pre-register** which variants and corrections you'll compare before running.

**Native blind A/B sizing:** n≈85 detects a 65% win rate; n=300 detects 58.1%. Use a 3-point scale
(A better / tie / B better), analyse with ties excluded, report tie rate separately. **3 native raters**,
blinded, randomised left/right with stored `order_seed`. Agreement: **Krippendorff's α (ordinal) ≥0.67**,
reported alongside **Gwet's AC1** (skewed prevalence deflates κ/α — the kappa paradox).

### DPO needs its own prompt slice — the easiest thing to get wrong
**Reserve 400 instructions held out of SFT**, same `split_group` rules. *If you reuse SFT prompts with the SFT
`chosen` response, DPO merely re-fits the SFT distribution: no gain, real risk of degeneration and length
blowup.*

- **Rejected side, free:** run the *base* model over those 400 at temp 0.7, N=2, keep outputs Stage A scores
  as MSA. Yields ≈600–800 pairs
- **Chosen side:** the human-written/reviewed Kuwaiti response. Stronger variant — sample the *SFT* model,
  have a native speaker edit the best sample, use the edit as `chosen` and the raw sample as `rejected`.
  Do this for ≥150 of the 400; it teaches the model to fix *its own* residual MSA
- **Rejected composition matters:** ~60% MSA leakage, ~20% **other-Gulf variant**, ~20% **caricature**
  (over-marked). Without the last two, DPO learns only MSA-avoidance and drifts toward caricature

### Best-of-N needs a calibration split
250 `(response, human_score)` pairs from **dev prompts, never the gold set** — otherwise you tune the reranker
on your headline metric. Reranker = weighted Stage A `P(dialect)` + Stage B `P(Kuwait)` (weight 0 if R<0.30)
+ KFI distance from the human band + length prior. Fit on the 250, report Spearman ρ vs human (target ≥0.5).
Default N=8 at temp 0.8. **Report N=1 and BoN@8 as separate rows** — BoN is a decoding-time gain, not a model
gain, and merging them is the easiest way to overstate the result.

### Capability canary — be honest about its power
100 ArabicMMLU MCQ + 30 Arabic instruction-following + 20 English, version-pinned and hash-checked, frozen at
project start. At n=200 the CI is ~±7pp, so it catches only **catastrophic** breakage (>10pp). Run at every
checkpoint as a smoke test; then run a **full 1,000-item ArabicMMLU subset (±3pp) on the two final candidates
only**, accepting a drop ≤2pp — while stating that a 2pp drop is not distinguishable from zero.

### Tokenizer check — the decision rule
ALLaM ships a `LlamaTokenizer` (SentencePiece). SentencePiece often applies `nmt_nfkc` internally and may fold
or drop non-standard letters. **Measure, don't assume:** round-trip `چ گ پ ڤ ژ` and 20 real sentences; assert
no `unk_token_id`; grep `tokenizer.json` for `"byte_fallback": true`; compute fertility of `چذي`/`گلت` vs
`كذي`/`قلت`.

| Result | Action |
|---|---|
| Clean round-trip, fertility ≤2× | Proceed unchanged |
| Round-trips, fertility 2–4× | Proceed; expect ~10–20% more tokens |
| **Fails round-trip / emits UNK** | `add_tokens(["چ","گ","پ","ڤ","ژ"])` + `resize_token_embeddings`, init new rows as mean of ك/ق rows. **Critical LoRA consequence: new embedding rows are untrained and frozen under standard LoRA — you must add `modules_to_save=["embed_tokens","lm_head"]`, which puts two ~64k×4096 matrices into trainable + optimizer state.** Tight on 16 GB; fall back to 8-bit optimizer + gradient checkpointing **before** falling back on the orthography |

**Do not normalise چ→ك as a workaround** — that silently reverses a locked decision.

---

## 10. What the brief gets right — protect these

1. **Step 0 loader smoke test before writing training code** — the Blackwell situation genuinely was unsettled
2. **Running the untuned base as a control in Step 3** — the single best decision in the document. Without it
   every number is meaningless. First thing cut under time pressure; don't
3. **Step 1 toy run end-to-end before investing in data collection**
4. **Refusing scripted API generation, with human review of every row** — slower, and the only quality gate the data has
5. **Explicitly stating the classifier measures Gulf-vs-MSA** — more honest than most published work
6. **Loss masked to response tokens only; r=16/α=32 on attention+MLP; zero-API runtime; standalone scripts**
7. **Correctly omitting continued pretraining** — that needs ~3.3B tokens on 8×A100; on one 16 GB card it buys
   catastrophic forgetting, not dialect

---

## 11. Revised build order

| Step | Change |
|---|---|
| **0** | + classifier calibration gate · + چ/گ tokenizer check · − Unsloth · use `humain-ai/` ID |
| **0.5** | **NEW** — schema, validator, group-splitter, dedup **before** the toy run (brief had these at 1.5, after) |
| **0.6** | **NEW** — quarantine the 150–300 real held-out items |
| 1 | Toy run — unchanged (keep it) |
| 2 | Training — batch 2 / accum 8 / seq 1024, dynamic padding, chat template |
| **2.5** | **NEW** — DPO/ORPO preference stage |
| 3 | Eval — swap classifier, + base 5-shot arm, + capability canary, + native blind A/B, real vs synthetic reported separately |
| 4 | Best-of-N reranking replaces single retry |
| 5 | Export + chat — unchanged |

---

## 12. Open items

- ❗ **Real-data commitment needs revisiting** — the methodology review puts the requirement at **~800 real
  items**, not the 150–300 originally agreed. Gold test (300) is the non-negotiable part; train seeds can be
  reduced to a floor of 150 if collection stalls
- ❓ **Hugging Face auth** — never resolved; needed before the ~14 GB ALLaM download
- ❓ Outreach to the Hazawi+ / KuHateCorpus authors — none of the named corpora are downloadable, so this is
  on the critical path if you want more than your own material
- ❓ Evaluate `QCRI/Fanar-1-9B-Instruct` ✅ (8.8B, gemma2, Apache-2.0, ungated, 717K downloads) as a
  co-primary against ALLaM in Step 0? Gulf-region-built and validated in 2026 Gulf-steering work
- ❓ Confirm whether transformers 5.15 uses fused/chunked cross-entropy (would shrink the 2.45 GiB logits term)
