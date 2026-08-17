# Project Brief: Kuwaiti Dialect LLM Fine-Tune (Fully Local Demo)

Give this whole document to Claude Code as the project brief.

## Goal

Fine-tune an open-source Arabic LLM using QLoRA so it responds in Kuwaiti Gulf Arabic dialect. This is a testing/demo build — fully self-contained and local, zero API dependency at runtime. No external API calls anywhere in the finished product.

## Hardware

RTX 5080, 16GB VRAM. Assume 4-bit QLoRA on a 7B model throughout, small batch sizes with gradient accumulation.

## Base Model Candidates (in priority order)

1. `ALLaM-AI/ALLaM-7B-Instruct-preview` — best documented Gulf-dialect-control result in a comparable study, Apache 2.0, LLaMA-architecture
2. `Falcon-Arabic-7B` — fallback if ALLaM has compatibility issues
3. `Jais-13B-Chat` — fallback, larger/heavier, only if 7B options underperform in early testing

## Where the Real Kuwaiti Text Comes From

Claude Code cannot find or scrape this on its own — these are the actual sources to pull real Kuwaiti dialect text from, which then become the seed examples fed into Claude Code sessions (per the next section):

- **Existing research datasets** (real people's real Kuwaiti writing, already collected by researchers):
  - Hazawi+ corpus — a Kuwaiti dialect text corpus, check current availability/access on Hugging Face or via the paper it originated from
  - The Kuwaiti X/Twitter sentiment dataset referenced in earlier Kuwaiti-dialect NLP research (~16k posts) — check current public availability
  - KuHateCorpus — smaller annotated Kuwaiti dataset, useful for raw dialect text even though it was built for a different task
  - Search Hugging Face datasets directly for "Kuwaiti Arabic" / "Gulf Arabic dialect" for anything newer than what's referenced here
- **Your own farouk.ai content** — you already write and review Gulf-dialect Reel scripts for Instagram under your own sourcing standard, and the comments/replies on that channel are real Kuwaiti audience text you already have access to and trust
- **Public social platforms** — Kuwaiti subreddits, Kuwaiti X accounts/threads, Kuwaiti YouTube comment sections, Kuwaiti forums — manually collected, not scraped at scale (scraping at scale raises ToS/legal questions per-platform, so treat this as manual collection of a few hundred good examples, not an automated crawler)
- **You, directly** — as a Kuwait-based builder, you writing or dictating real conversational exchanges yourself is some of the highest-quality, most trustworthy seed data available, since it skips the "is this actually authentic" question entirely

None of this is thousands of ready-made examples on its own — it's the small, real seed set (aim for a few hundred solid examples across these sources) that then gets fed into Claude Code chat sessions to multiply into the full few-thousand-pair dataset, per the process below.

## Important: How Training Data Gets Generated (no API scripting)

The training data generation step is done by me, manually, in Claude Code chat sessions — not by a script calling any API. Do not build a `generate_pairs.py` that calls an external LLM API. Instead:
- I will provide real Kuwaiti dialect examples directly in our Claude Code conversation
- I will ask Claude Code to generate batches of additional Kuwaiti-dialect instruction/response pairs in the same style, directly in chat, which I'll review and save to the dataset file myself in batches
- Your job is to build the tooling that receives and structures this data (the JSONL schema, the tagging script, the review/split scripts) — not a data-generation API pipeline

## Build Order

### Step 0 — Environment + compatibility check
- Set up Python 3.11 env with PyTorch (CUDA build matching installed driver), `transformers`, `peft`, `bitsandbytes`, `accelerate`
- Write a small script that attempts to load ALLaM-7B-Instruct-preview in 4-bit via (a) Unsloth's `FastLanguageModel` and (b) plain `transformers` + `bitsandbytes`. Report which path works and use it for everything downstream.
- If ALLaM fails to load cleanly in either path, fall back to Falcon-Arabic-7B and repeat this check.

### Step 1 — Toy run (validate the pipeline before committing real time)
- I will hand you ~150-200 rough instruction/Kuwaiti-response pairs directly in chat (quality doesn't need to be perfect — this is a plumbing test)
- Run one full pass through Steps 2-4 below end-to-end on this toy set
- Confirm: training completes without OOM, loss decreases, the eval script runs and produces a report, the merged/adapter model loads and generates
- Only move to Step 1.5 once this loop works cleanly

### Step 1.5 — Dataset tooling (not generation)
- `data/dialect_tag.py`: prepend a consistent tag (e.g. `[KWT]`) to every instruction in whatever dataset file I provide
- `data/schema.py` or equivalent: validate incoming JSONL matches `{instruction, response, tag}`, catch malformed entries
- `data/split.py`: split into train/eval sets (85/15)
- `data/review_sample.py`: export a random sample to a plain readable file so I can manually review/correct before training — this stays a manual, human-in-the-loop step, not automated
- I will paste/import data in batches over multiple sessions; the target is ~3,000-5,000 pairs total, built up over time, not generated in one go

### Step 2 — Training script
`train/qlora_finetune.py`:
- Load base model in 4-bit (via whichever path Step 0 confirmed)
- LoRA config: rank 16, alpha 32, target attention + MLP projection layers
- Causal LM objective, loss masked to response tokens only
- Cosine LR schedule with warmup, mixed precision BF16, max sequence length ~2048
- Small batch size (2-4) with gradient accumulation to fit VRAM
- Checkpoint every N steps, log loss/perplexity to CSV
- Start with 3 epochs on the real dataset; watch validation loss for overfitting before increasing

### Step 3 — Evaluation script
`eval/score_dialect.py`:
- Hold out the eval split from Step 1.5, never seen in training
- Run test set through both the fine-tuned model AND the un-tuned base model for a clear before/after
- Score outputs with a local Arabic dialect classifier (MARBERTv2 or best currently available equivalent that can run locally — confirm it's actually downloadable/runnable offline before committing to it)
- Report: % classified as Gulf dialect, % MSA leakage, side-by-side outputs saved to a markdown report for my manual read-through

### Step 4 — Local fix-up pass (demo-scope, no API)
`serve/fixup.py`:
- After the fine-tuned model generates a response, run it through the local dialect classifier from Step 3
- If flagged as MSA/non-Kuwaiti, feed the response back into the **same local fine-tuned model** with an instruction like "rewrite this fully in Kuwaiti dialect" and use that regenerated output instead
- This is a full regenerate, not a targeted word-level edit — acceptable for this demo scope, no external API involved anywhere
- Cap retries at 1 (don't loop indefinitely) — if the second attempt still fails the check, return it anyway and log it for later review rather than retrying forever

### Step 5 — Export + local chat test
- `export/merge_adapter.py`: merge LoRA weights into base model for standalone use (also keep the raw adapter for reuse)
- `serve/chat.py`: minimal terminal chat loop wired through the Step 4 fix-up pass, fully local, for interactive testing

## Constraints

- Zero external API calls anywhere in the runtime pipeline (Steps 2-5) — everything must run on local hardware, no network dependency once the model is trained
- Do not skip Step 0 or Step 1
- Do not treat the dialect classifier score as the only success metric — it measures Gulf-vs-MSA, not Kuwaiti-vs-other-Gulf-dialects specifically
- Realistic target based on the closest published comparable result: ~80-85% dialect accuracy, ~6-10% residual MSA leakage before the fix-up pass; the fix-up pass is meant to catch some of that remainder, not eliminate it entirely
- Keep every script runnable standalone so individual stages can be re-run/debugged in isolation
- README at project root should document: which base model ended up being used and why, current eval numbers, and what to try next if quality is insufficient
