#!/usr/bin/env python3
"""
Baseline probe (PLAN.md P0.3) — does prompting alone already produce acceptable Kuwaiti?

Runs three arms on the same prompts, with NO training of any kind:
  A1  zero-shot, no system prompt
  A2  Kuwaiti system prompt, no examples
  A3  Kuwaiti system prompt + 5 verbatim real examples, best-of-N

Decoding is pinned per docs/PREREGISTRATION.md and recorded in the output.

Standalone:  python probe/run_probe.py --seeds probe/seeds.jsonl --out eval/raw/probe.jsonl
"""
import argparse, json, os, random, sys, time
from pathlib import Path

# Pinned by pre-registration. Changing these invalidates the probe.
TEMPERATURE = 0.8
TOP_P = 0.9
SEED = 42
MAX_NEW_TOKENS = 120
BON_N = 8

MODEL = "humain-ai/ALLaM-7B-Instruct-preview"

SYSTEM_KW = (
    "انت مساعد كويتي. رد دائماً باللهجة الكويتية العامية مثل ما يتكلمون الكويتيين "
    "في المحادثات اليومية. لا تستخدم الفصحى. خل ردودك قصيرة وطبيعية."
)

# MSA function words. Deliberately NOT the Kuwaiti marker list -- that list is
# UNVALIDATED (see docs/evidence/marker_discrimination.md, only 2 of 22 markers
# actually discriminate Kuwaiti). These are MSA-side signals, which is a
# different and safer claim.
MSA_MARKERS = ["سوف", "لقد", "الذي", "التي", "ليس", "إنّ", "إن ", "قد ", "لم ", "لن ",
               "بصفتي", "لا أستطيع", "يمكنني أن", "وذلك", "حيث أن"]


def bon_score(text, target_lo=3, target_hi=40):
    """Heuristic best-of-N scorer. No Kuwaiti marker list (unvalidated).
    Penalises MSA function words and length outside the real-corpus band."""
    n_words = len(text.split())
    msa_hits = sum(1 for m in MSA_MARKERS if m in text)
    score = -2.0 * msa_hits
    if n_words < target_lo:
        score -= 3.0 * (target_lo - n_words)
    elif n_words > target_hi:
        score -= 0.15 * (n_words - target_hi)
    # mild penalty for degenerate repetition
    toks = text.split()
    if len(toks) >= 6:
        tri = [" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
        rep = 1.0 - (len(set(tri)) / max(len(tri), 1))
        score -= 4.0 * rep
    return score


def load_seeds(path):
    """seeds.jsonl: {"id":..., "user":"<kuwaiti>", "reply":"<kuwaiti>"}"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True, help="JSONL of author-written Kuwaiti seed items")
    ap.add_argument("--out", default="eval/raw/probe.jsonl")
    ap.add_argument("--n-shots", type=int, default=5)
    ap.add_argument("--n-prompts", type=int, default=40)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    random.seed(SEED)
    torch.manual_seed(SEED)

    seeds = load_seeds(args.seeds)
    if len(seeds) < args.n_shots + 5:
        sys.exit(f"need at least {args.n_shots + 5} seed items, got {len(seeds)}")

    idx = list(range(len(seeds)))
    random.shuffle(idx)
    shot_idx = idx[:args.n_shots]
    prompt_idx = idx[args.n_shots:args.n_shots + args.n_prompts]
    shots = [seeds[i] for i in shot_idx]
    prompts = [seeds[i] for i in prompt_idx]
    print(f"{len(seeds)} seeds -> {len(shots)} few-shot exemplars, {len(prompts)} eval prompts")

    tok = AutoTokenizer.from_pretrained(args.model)
    # CRITICAL: ALLaM ships pad_token=None. Use unk, NOT eos -- pad==eos would
    # mask the real EOS under response-only loss masking.
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
        print(f"pad_token was None -> set to unk ({tok.pad_token!r}, id={tok.pad_token_id})")
    tok.padding_side = "left"  # required for batched generation

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("loading model in 4-bit NF4 ...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="cuda:0", dtype=torch.bfloat16,
    )
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s, "
          f"VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    def build(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def gen(texts, n=1):
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=n,
                pad_token_id=tok.pad_token_id,
            )
        gen_only = out[:, enc["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip() for g in gen_only]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")
    meta = dict(kind="probe_meta", model=args.model, temperature=TEMPERATURE, top_p=TOP_P,
                seed=SEED, max_new_tokens=MAX_NEW_TOKENS, bon_n=BON_N,
                n_shots=len(shots), n_prompts=len(prompts),
                shot_ids=[s.get("id") for s in shots])
    fout.write(json.dumps(meta, ensure_ascii=False) + "\n")

    for i, p in enumerate(prompts):
        user = p["user"]
        rec = {"kind": "probe_item", "id": p.get("id"), "user": user}

        # A1 zero-shot
        rec["A1"] = gen([build([{"role": "user", "content": user}])])[0]

        # A2 system prompt only
        rec["A2"] = gen([build([{"role": "system", "content": SYSTEM_KW},
                                {"role": "user", "content": user}])])[0]

        # A3 system + 5 real shots, best-of-N
        msgs = [{"role": "system", "content": SYSTEM_KW}]
        for s in shots:
            msgs.append({"role": "user", "content": s["user"]})
            msgs.append({"role": "assistant", "content": s["reply"]})
        msgs.append({"role": "user", "content": user})
        cands = gen([build(msgs)], n=BON_N)
        scored = sorted(((bon_score(c), c) for c in cands), key=lambda x: -x[0])
        rec["A3"] = scored[0][1]
        rec["A3_candidates"] = [{"text": c, "score": round(s, 3)} for s, c in scored]

        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{i+1}/{len(prompts)}] {user[:40]}...")

    fout.close()
    print(f"\nwrote {args.out}")
    print(f"peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    print("\nNext: python probe/rate.py --probe %s --anchors ..." % args.out)


if __name__ == "__main__":
    main()
