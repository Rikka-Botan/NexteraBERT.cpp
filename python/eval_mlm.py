#!/usr/bin/env python3
"""Masked-LM accuracy of GGUF files against the PyTorch fp32 model, on a large sample.

    python python/eval_mlm.py --gguf models/*-Q8_0.gguf models/*-Q4_K_M.gguf --json results/verify_mlm.json

verify.py's MLM check shares its few dozen documents with the hidden-state comparison, which
leaves about a point of sampling noise on the accuracy. This script masks 15% of the tokens of
many more documents (AG News, cut to fixed lengths) and reports, per file: accuracy against the
original token, how often the top prediction equals PyTorch's, and standard errors. The
accuracy difference to fp32 is measured on the same positions, so its standard error (paired)
is much smaller than that of either accuracy.

The files are evaluated one after another: each runtime keeps its own spinning thread pool, so
several loaded at once would fight over the cores.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexterabert_cpp import NexteraBERT  # noqa: E402
from verify import DEFAULT_REFERENCE, Reference, build_docs, mask_positions  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE, help="Hub repo id or local directory")
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--per-length", type=int, default=250, help="documents per length")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    ref = Reference(args.reference, args.threads or None)
    docs = build_docs(ref.tokenizer, args.lengths, args.per_length)
    rng = np.random.default_rng(args.seed)
    print(f"{len(docs)} documents ({args.per_length} each at T = {args.lengths})", flush=True)

    # pass 1: PyTorch. Keep only what the comparison needs (the masked input, the original
    # tokens and PyTorch's top prediction), not the 50k-wide logits.
    items, t0 = [], time.time()
    for ids in docs:
        pos = mask_positions(len(ids), rng)
        masked = list(ids)
        for p in pos:
            masked[p] = ref.mask_id
        items.append((masked, pos, np.asarray(ids)[pos], ref.mlm_logits(masked, pos).argmax(-1)))
    gold = np.concatenate([it[2] for it in items])
    top_r = np.concatenate([it[3] for it in items])
    r_ok = (top_r == gold).astype(np.float64)
    n, acc_ref = len(gold), float((top_r == gold).mean())
    print(f"PyTorch fp32: {100 * acc_ref:.2f}% on {n:,} positions ({time.time() - t0:.0f} s)", flush=True)
    del ref

    # pass 2: one GGUF at a time
    results = []
    for path in args.gguf:
        t0 = time.time()
        m = NexteraBERT(path, n_threads=args.threads)
        top_q = np.concatenate([m.mlm_logits(masked, pos).argmax(-1) for masked, pos, _, _ in items])
        ok = (top_q == gold).astype(np.float64)
        diff = ok - r_ok
        results.append({
            "file": Path(path).name, "ftype": m.ftype, "size_mib": Path(path).stat().st_size / 2**20,
            "n_masked": n, "n_docs": len(docs), "lengths": args.lengths,
            "accuracy_ref": acc_ref, "accuracy": float(ok.mean()),
            "accuracy_se": float(np.sqrt(ok.mean() * (1 - ok.mean()) / n)),
            "accuracy_diff": float(diff.mean()), "accuracy_diff_se": float(diff.std(ddof=1) / np.sqrt(n)),
            "top1_agreement": float((top_q == top_r).mean()),
        })
        print(f"{m.ftype}: {100 * ok.mean():.2f}% ({time.time() - t0:.0f} s)", flush=True)
        m.close()
        if args.json:                # after every file: the measurements are the expensive part
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    se_ref = np.sqrt(acc_ref * (1 - acc_ref) / n)
    print(f"\n## Masked-LM, 15% random masks: {n:,} positions in {len(docs):,} documents; "
          f"PyTorch fp32 accuracy {100 * acc_ref:.2f}% (s.e. {100 * se_ref:.2f})\n")
    print("| file type | accuracy | difference to fp32 (paired s.e.) | top-1 = PyTorch |")
    print("|---|---:|---:|---:|")
    for r in results:
        print(f"| {r['ftype']} | {100 * r['accuracy']:.2f}% | {100 * r['accuracy_diff']:+.2f} ({100 * r['accuracy_diff_se']:.2f}) | "
              f"{100 * r['top1_agreement']:.2f}% |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
