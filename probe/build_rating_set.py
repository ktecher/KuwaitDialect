#!/usr/bin/env python3
"""
Assemble the blind discrimination rating set (PLAN.md P2.2, docs/PREREGISTRATION.md).

60 items, shuffled, unlabelled:
   30  model outputs (arm A3)          <- the thing being measured
   15  real human Kuwaiti (QADI)       <- rater-validity anchors + ceiling
    8  marker-stuffed                  <- caricature detection check
    7  MSA / other-Gulf                <- floor anchors

Real anchors come from QADI, NOT the author's own writing -- the author would
recognise their own text and the anchor would be void.

Standalone:
  python probe/build_rating_set.py --probe eval/raw/probe.jsonl \
      --qadi data/raw/qadi_kuwait.jsonl --out eval/raw/rating_set.jsonl
"""
import argparse, json, random, re, hashlib
from pathlib import Path

ORDER_SEED = 20260818  # recorded; changing it changes the presentation order

# Over-stuffed caricature templates. Density is the point, not correctness --
# these deliberately pack far more dialect markers than the real corpus rate
# (2.28 per 100 words, see docs/evidence/marker_gate_validation.md).
CARICATURE = [
    "شلونك عساك طيب؟ وايد اشتقتلك چذي والله زين انك ييت هالحين ترا صدق وايد زين",
    "عيل شنو رايك؟ وايد حلو چذي زين ترا شلون ما قلتلي هالحين صدق وايد",
    "ترا وايد زين شلون چذي صارت السالفة عيل قلي شنو صار وايد اشتقتلك زين",
    "هلا والله شلونك وشلون الاهل؟ وايد زين چذي ترا عيل شنو اخبارك هالحين",
    "زين زين وايد چذي شلون ترا عيل شنو هالحين صدق وايد زين والله",
    "شنو هالسالفة؟ وايد چذي ترا شلون زين عيل هالحين قلي وايد صدق",
    "وايد وايد زين شلونك عيل ترا چذي هالحين شنو صار والله زين",
    "عساك بخير شلونك وايد زين چذي ترا عيل شنو هالحين صدق وايد اشتقتلك",
]

MSA_FLOOR = [
    "أعتقد أن هذا الموضوع يحتاج إلى مزيد من الدراسة والتحليل قبل اتخاذ أي قرار.",
    "لقد كان الطقس اليوم حاراً جداً، ولذلك فضلت البقاء في المنزل.",
    "سوف أتصل بك لاحقاً بعد أن أنتهي من عملي في المكتب.",
    "ما رأيك في الفيلم الذي شاهدناه أمس؟ لم يعجبني كثيراً.",
]
OTHER_GULF = [
    "شحالك؟ شخبارك؟ وايد زمان ما شفتك يا الغالي",          # Emirati
    "وش رايك نروح نتعشى؟ ابغى اجرب المطعم الجديد",          # Saudi
    "شلونك؟ شكو ماكو؟ اكو شي جديد عدكم؟",                    # Iraqi
]


def norm(s):
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"@\w+", "", s)
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--qadi", required=True)
    ap.add_argument("--out", default="eval/raw/rating_set.jsonl")
    ap.add_argument("--arm", default="A3")
    ap.add_argument("--n-model", type=int, default=30)
    ap.add_argument("--n-real", type=int, default=15)
    args = ap.parse_args()

    rng = random.Random(ORDER_SEED)
    items = []

    # model outputs
    model_rows = []
    with open(args.probe, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "probe_item" and r.get(args.arm):
                model_rows.append(r)
    rng.shuffle(model_rows)
    for r in model_rows[:args.n_model]:
        items.append({"type": "model", "arm": args.arm, "src_id": r.get("id"),
                      "text": norm(r[args.arm])})

    # real human anchors, length-matched to the model outputs
    mlens = [len(i["text"].split()) for i in items] or [13]
    lo, hi = max(3, min(mlens)), max(mlens) + 5
    qadi = []
    with open(args.qadi, encoding="utf-8") as f:
        for line in f:
            t = norm(json.loads(line)["text"])
            if lo <= len(t.split()) <= hi:
                qadi.append(t)
    rng.shuffle(qadi)
    for t in qadi[:args.n_real]:
        items.append({"type": "real", "text": t})

    for t in CARICATURE:
        items.append({"type": "caricature", "text": t})
    floor = MSA_FLOOR + OTHER_GULF
    for t in floor:
        items.append({"type": "floor", "text": t})

    rng.shuffle(items)
    for n, it in enumerate(items, 1):
        it["item_id"] = n
        it["hash"] = hashlib.sha1(it["text"].encode()).hexdigest()[:10]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "rating_meta", "order_seed": ORDER_SEED,
                            "counts": {k: sum(1 for i in items if i["type"] == k)
                                       for k in ("model", "real", "caricature", "floor")},
                            "arm": args.arm}, ensure_ascii=False) + "\n")
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"wrote {args.out}: {len(items)} items")
    for k in ("model", "real", "caricature", "floor"):
        print(f"  {k:11s} {sum(1 for i in items if i['type']==k)}")
    print("\nLabels are in the file but NOT shown during rating.")
    print("Next: python probe/rate.py --set", args.out)


if __name__ == "__main__":
    main()
