# Gate A — Tokenizer handling of چ / گ

**Run:** 2026-08-18 · **Verdict: PASS on ALLaM** · tokenizer files only (~20 MB), no GPU, no weights

## Why this gate exists

چ (U+0686) and گ (U+06AF) are not standard Arabic letters. If the tokenizer folds, drops, or `<unk>`s
them, the locked orthography silently degrades every training pair — and the failure would only surface
much later as poor output quality, with no obvious cause.

## Result — `humain-ai/ALLaM-7B-Instruct-preview`

```
class      : LlamaTokenizer (is_fast=True)
vocab_size : 64000
unk_token  : '<unk>' (id=0)

-- single letters --
  OK  چ  ids=[6883]            1 token
  OK  گ  ids=[15120]           1 token
  OK  پ  ids=[7485]            1 token
  OK  ڤ  ids=[63619,241,187]   3 tokens (byte-fallback, round-trips)
  OK  ژ  ids=[63619,241,175]   3 tokens (byte-fallback, round-trips)

-- fertility (special vs standard counterpart) --
  چذي=2 vs كذي=2   ratio 1.00
  گلت=2 vs قلت=1   ratio 2.00
  عليچ=2 vs عليك=1  ratio 2.00
  چم=2  vs كم=1    ratio 2.00
  ضحگ=2 vs ضحك=1   ratio 2.00

-- round-trip, 10 real Kuwaiti sentences from the QADI KW subset --
  round-trip failures : 0/10
  sentences with UNK  : 0/10
  tokens/word         : 1.57
```

**چ and گ are each a single token in ALLaM's vocabulary.** All fertility ratios ≤2.0.

## What this eliminates

- ❌ `add_tokens` / `resize_token_embeddings`
- ❌ `modules_to_save=["embed_tokens","lm_head"]`
- ❌ **+4.88 GiB VRAM penalty** (524.3M params into trainable + fp32 grads + optimizer state)
- ❌ Fanar as a tokenizer escape hatch

Batch 8 remains affordable, which it would not have been on the failure branch.

## Fanar comparison — ALLaM wins on merit

`QCRI/Fanar-1-9B-Instruct` (GemmaTokenizer, vocab 128256) also passes, and handles ڤ/ژ better
(single tokens). **But it is meaningfully worse where it counts:**

| | ALLaM | Fanar |
|---|---|---|
| tokens/word on Kuwaiti | **1.57** | 2.00 |

ALLaM is ~27% more token-efficient on Kuwaiti text, compounding across every training step and every
context window. ALLaM is the right base on the merits, not merely by default.

## Two follow-ups verified in the same pass

**`chat_template` is present** (816 chars), Llama-2 format:
```
<s> [INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] {assistant} </s>
```
The generation prompt ends at `[/INST]` — **that is the response-masking boundary for training.**

**⚠️ `pad_token` is `None`.** Must be set explicitly before training. **Set `pad_token = unk_token`
(id 0), not `eos`.** With `pad == eos`, masking pads also masks the real `</s>`, so the model never
learns to stop — surfacing much later as runaway generation. Fanar ships a distinct `<pad>` (id 0) and
does not have this problem.

## Caveat

ڤ and ژ cost 3 tokens each via byte-fallback. They round-trip cleanly so no information is lost, and
they are rare in Kuwaiti (mostly loanwords) — but heavy use carries a small efficiency cost.

Reproduce: `scratchpad/gate_a.py`, `scratchpad/gate_a2.py`
