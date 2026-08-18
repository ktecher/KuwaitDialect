#!/usr/bin/env python3
"""
Likelihood contrast (PLAN.md P2.1) -- the primary AUTOMATIC metric.

    D = mean_per_token_logprob(Kuwaiti) - mean_per_token_logprob(MSA)

Reported for base and tuned. A successful fine-tune raises D.

Why this metric exists
----------------------
Every other automatic metric available here is circular. The marker gate, the
KFI score and the reranker were all built from the same word list -- and that
list turned out to be 2/22 useful (docs/evidence/marker_discrimination.md).
The dialect classifier is a stale 21-class model predicted to fail its own
calibration gate.

This metric is different in three ways that matter:
  1. It uses NO word list and NO classifier.
  2. It scores WHOLE STRINGS, so it cannot be gamed by sprinkling markers --
     the exact failure mode Gate C exists to prevent.
  3. It is computed on text that neither the author nor Claude wrote.

Length normalisation is per-token, so longer texts don't dominate. The Kuwaiti
and MSA sides are additionally length-matched by default, because raw perplexity
correlates with length and that would otherwise confound the contrast.

Standalone:
  python eval/likelihood_contrast.py \
      --kuwaiti data/raw/qadi_kuwait.jsonl \
      --msa data/raw/msa_sample.txt \
      --model humain-ai/ALLaM-7B-Instruct-preview \
      --adapter runs/sft/adapter          # optional; omit for base-only
"""
import argparse, json, math, os, random, statistics as st
from pathlib import Path

SEED = 42


def read_lines(path, limit=None):
    out = []
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if p.suffix == ".jsonl":
                try:
                    o = json.loads(line)
                    line = o.get("text") or o.get("reply") or ""
                except json.JSONDecodeError:
                    pass
            line = " ".join(line.split())
            if line:
                out.append(line)
    if limit:
        random.Random(SEED).shuffle(out)
        out = out[:limit]
    return out


def length_match(a, b, tol=0.15):
    """Trim both lists so their word-length distributions are comparable."""
    ma = st.median(len(x.split()) for x in a)
    lo, hi = ma * (1 - tol) - 2, ma * (1 + tol) + 2
    a2 = [x for x in a if lo <= len(x.split()) <= hi]
    b2 = [x for x in b if lo <= len(x.split()) <= hi]
    n = min(len(a2), len(b2))
    return a2[:n], b2[:n]


def score_texts(model, tok, texts, batch_size=8, device="cuda"):
    """Mean per-token log-probability of each text (higher = more expected)."""
    import torch
    out = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits
        # shift for next-token prediction
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        mask = attn[:, 1:].bool()
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for row, m in zip(tok_lp, mask):
            vals = row[m]
            if vals.numel():
                out.append(vals.mean().item())
    return out


def summarise(name, kw, msa):
    d = st.mean(kw) - st.mean(msa)
    se = math.sqrt(st.pstdev(kw) ** 2 / len(kw) + st.pstdev(msa) ** 2 / len(msa))
    print(f"  {name:12s} logP/tok  KW {st.mean(kw):+.4f}   MSA {st.mean(msa):+.4f}"
          f"   D = {d:+.4f}  (+/- {1.96*se:.4f})")
    return {"model": name, "kuwaiti_mean": st.mean(kw), "msa_mean": st.mean(msa),
            "D": d, "ci95": 1.96 * se, "n_kw": len(kw), "n_msa": len(msa)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kuwaiti", required=True)
    ap.add_argument("--msa", required=True)
    ap.add_argument("--model", default="humain-ai/ALLaM-7B-Instruct-preview")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="eval/raw/likelihood_contrast.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    kw = read_lines(args.kuwaiti, limit=args.n * 3)
    msa = read_lines(args.msa, limit=args.n * 3)
    kw, msa = length_match(kw, msa)
    kw, msa = kw[:args.n], msa[:args.n]
    print(f"length-matched: {len(kw)} Kuwaiti / {len(msa)} MSA, "
          f"median words {st.median(len(x.split()) for x in kw):.0f}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token          # NOT eos -- see Gate A evidence
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="cuda:0", dtype=torch.bfloat16)

    print("\nlikelihood contrast  (higher D = more Kuwaiti-leaning)")
    results = [summarise("base", score_texts(model, tok, kw), score_texts(model, tok, msa))]

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        results.append(summarise("tuned", score_texts(model, tok, kw),
                                 score_texts(model, tok, msa)))
        delta = results[1]["D"] - results[0]["D"]
        print(f"\n  DELTA D (tuned - base) = {delta:+.4f}")
        print("  positive = fine-tune shifted the model toward Kuwaiti")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "model": args.model, "adapter": args.adapter,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
