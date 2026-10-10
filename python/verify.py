#!/usr/bin/env python3
"""Check GGUF files against the PyTorch model they were converted from.

    python python/verify.py --gguf models/*-f32.gguf models/*-Q8_0.gguf models/*-Q4_K_M.gguf

Three measurements, each against the fp32 PyTorch reference run one sequence at
a time (no padding):

  hidden   cosine similarity of every token's last_hidden_state and of the
           mean-pooled sentence embedding, on news text cut to fixed lengths
  mlm      masked-LM agreement at 15% random [MASK] positions: same top-1 as the
           reference, accuracy against the original token, KL(reference || gguf)
  sts      STS-B test: Spearman of embedding cosine vs the gold scores, and how
           closely the GGUF's similarities track the reference's

An f32 GGUF should match to float rounding (cosine 1.000000); a quantized one
should stay within what its bit width allows -- that is what this script shows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexterabert_cpp import NexteraBERT  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_REFERENCE = "RikkaBotan/NexteraBERT-Mezzoforte-220M-en"


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build_docs(tokenizer, lengths: list[int], per_length: int) -> list[list[int]]:
    """Token-id sequences of exactly the requested lengths, made by concatenating
    AG News articles and truncating (special tokens included in the count)."""
    from datasets import load_dataset

    news = load_dataset("sentence-transformers/agnews", split="train")
    docs, row = [], 0
    for target in lengths:
        for _ in range(per_length):
            text, ids = "", []
            while len(ids) < target:
                # One article at a time; far from a long target, about as many as are
                # still missing (~50 tokens each) so the text is not re-tokenized per article.
                missing = target - len(ids)
                for _ in range(missing // 50 if missing > 4096 else 1):
                    r = news[row]
                    row += 1
                    text = (text + " " + r["title"] + ". " + r["description"]).strip()
                ids = tokenizer(text, truncation=True, max_length=target)["input_ids"]
            docs.append(ids)
    return docs


def mask_positions(n_tokens: int, rng: np.random.Generator, ratio: float = 0.15) -> np.ndarray:
    inner = np.arange(1, n_tokens - 1)                  # never [CLS] / [SEP]
    k = max(1, int(round(ratio * len(inner))))
    return np.sort(rng.choice(inner, size=min(k, len(inner)), replace=False))


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------

class Reference:
    def __init__(self, repo: str, threads: int | None = None):
        import torch
        from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()
        self.mlm = AutoModelForMaskedLM.from_pretrained(repo, trust_remote_code=True).eval()
        self.mask_id = self.tokenizer.mask_token_id

    def hidden(self, ids: list[int]) -> np.ndarray:
        t = self.torch
        with t.inference_mode():
            x = t.tensor([ids])
            return self.model(input_ids=x, attention_mask=t.ones_like(x)).last_hidden_state[0].numpy()

    def mlm_logits(self, ids: list[int], positions: np.ndarray) -> np.ndarray:
        t = self.torch
        with t.inference_mode():
            x = t.tensor([ids])
            logits = self.mlm(input_ids=x, attention_mask=t.ones_like(x)).logits[0]
            return logits[t.as_tensor(positions)].numpy()


def reference_outputs(ref: Reference, docs, sts, mlm_max_len: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {"hidden": [], "mlm": [], "sts": None}
    for ids in docs:
        out["hidden"].append(ref.hidden(ids))
    for ids in docs:
        if len(ids) > mlm_max_len or len(ids) < 8:
            out["mlm"].append(None)
            continue
        pos = mask_positions(len(ids), rng)
        masked = list(ids)
        for p in pos:
            masked[p] = ref.mask_id
        out["mlm"].append({"masked": masked, "pos": pos, "gold": np.asarray(ids)[pos],
                           "logits": ref.mlm_logits(masked, pos)})
    if sts is not None:
        emb = []
        for s in sts["sentences"]:
            h = ref.hidden(ref.tokenizer(s)["input_ids"]).mean(0)
            emb.append(h / np.linalg.norm(h))
        out["sts"] = np.stack(emb)
    return out


def load_sts(n: int | None):
    from datasets import load_dataset

    d = load_dataset("mteb/stsbenchmark-sts", split="test")
    if n:
        d = d.select(range(min(n, len(d))))
    s1, s2 = list(d["sentence1"]), list(d["sentence2"])
    return {"sentences": s1 + s2, "n": len(s1), "gold": np.asarray(d["score"], dtype=np.float64)}


# ---------------------------------------------------------------------------
# GGUF evaluation
# ---------------------------------------------------------------------------

def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def evaluate(path: str, docs, ref_out: dict, sts, ref_tokenizer, threads: int = 0, int8: bool = True) -> dict:
    from scipy.stats import spearmanr

    m = NexteraBERT(path, n_threads=threads, int8=int8)
    res = {"file": Path(path).name, "ftype": m.ftype, "size_mib": Path(path).stat().st_size / 2**20,
           "system": m.system_info}

    # tokenizer embedded in the GGUF == the Hugging Face one
    probe = ["Hello world!", "NexteraBERT: convolution + attention, 8192 tokens.", "  spaces\tand\nnewlines ",
             "naive café – “quotes” 日本語 🙂", "def f(x):\n    return x ** 2"]
    res["tokenizer_ok"] = m.tokenize(probe) == [ref_tokenizer(p)["input_ids"] for p in probe]

    by_len: dict[int, dict[str, list]] = {}
    tok_cos, pool_cos, rel_err = [], [], []
    for ids, h_ref in zip(docs, ref_out["hidden"]):
        _, (h,) = m.encode_ids([ids], normalize=False, return_hidden=True)
        tc = cosine_rows(h_ref, h)
        pc = float(cosine_rows(h_ref.mean(0, keepdims=True), h.mean(0, keepdims=True))[0])
        re = float(np.linalg.norm(h - h_ref) / np.linalg.norm(h_ref))
        tok_cos.append(tc)
        pool_cos.append(pc)
        rel_err.append(re)
        b = by_len.setdefault(len(ids), {"tok": [], "pool": []})
        b["tok"].append(tc)
        b["pool"].append(pc)
    allc = np.concatenate(tok_cos)
    res["hidden"] = {
        "token_cos_mean": float(allc.mean()), "token_cos_min": float(allc.min()),
        "token_cos_p01": float(np.quantile(allc, 0.01)),
        "pooled_cos_mean": float(np.mean(pool_cos)), "pooled_cos_min": float(np.min(pool_cos)),
        "rel_l2_err_mean": float(np.mean(rel_err)),
        "by_length": {str(k): {"token_cos_mean": float(np.concatenate(v["tok"]).mean()),
                               "pooled_cos_mean": float(np.mean(v["pool"]))}
                      for k, v in sorted(by_len.items())},
    }

    if m.has_mlm_head:
        agree = n = acc_ref = acc_q = 0
        kl = []
        for item in ref_out["mlm"]:
            if item is None:
                continue
            lq = m.mlm_logits(item["masked"], item["pos"])
            lr = item["logits"]
            tq, tr = lq.argmax(-1), lr.argmax(-1)
            agree += int((tq == tr).sum())
            acc_ref += int((tr == item["gold"]).sum())
            acc_q += int((tq == item["gold"]).sum())
            n += len(tq)
            pr, pq = log_softmax(lr), log_softmax(lq)
            kl.append((np.exp(pr) * (pr - pq)).sum(-1))
        if n:
            res["mlm"] = {"n_masked": n, "top1_agreement": agree / n, "accuracy_ref": acc_ref / n,
                          "accuracy": acc_q / n, "kl_mean": float(np.concatenate(kl).mean())}

    if sts is not None:
        emb = m.encode(sts["sentences"], batch_size=16)
        k = sts["n"]
        sim_q = (emb[:k] * emb[k:]).sum(-1).astype(np.float64)
        e_ref = ref_out["sts"]
        sim_r = (e_ref[:k] * e_ref[k:]).sum(-1).astype(np.float64)
        res["sts"] = {
            "n_pairs": k,
            "spearman_gold_ref": float(spearmanr(sim_r, sts["gold"]).statistic),
            "spearman_gold": float(spearmanr(sim_q, sts["gold"]).statistic),
            "spearman_vs_ref": float(spearmanr(sim_q, sim_r).statistic),
            "mean_abs_sim_diff": float(np.abs(sim_q - sim_r).mean()),
            "embedding_cos_mean": float(cosine_rows(e_ref, emb).mean()),
            "embedding_cos_min": float(cosine_rows(e_ref, emb).min()),
        }
    m.close()
    return res


def print_report(results: list[dict]) -> None:
    print("\n## Hidden states vs PyTorch fp32 (cosine similarity)\n")
    print("| file type | size MiB | token cos mean | token cos p1 | token cos min | pooled cos mean | pooled cos min | rel. L2 err |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        h = r["hidden"]
        print(f"| {r['ftype']} | {r['size_mib']:.0f} | {h['token_cos_mean']:.6f} | {h['token_cos_p01']:.6f} | "
              f"{h['token_cos_min']:.6f} | {h['pooled_cos_mean']:.6f} | {h['pooled_cos_min']:.6f} | "
              f"{h['rel_l2_err_mean']:.2e} |")
    lens = list(results[0]["hidden"]["by_length"])
    print("\nToken cosine by sequence length:\n")
    print("| file type | " + " | ".join(f"T={k}" for k in lens) + " |")
    print("|---|" + "---:|" * len(lens))
    for r in results:
        print(f"| {r['ftype']} | " + " | ".join(
            f"{r['hidden']['by_length'][k]['token_cos_mean']:.6f}" for k in lens) + " |")

    if all("mlm" in r for r in results):
        ref_acc = results[0]["mlm"]["accuracy_ref"]
        print(f"\n## Masked-LM, 15% random masks ({results[0]['mlm']['n_masked']} positions; "
              f"PyTorch fp32 accuracy {100 * ref_acc:.2f}%)\n")
        print("| file type | top-1 = reference | accuracy | KL(ref ‖ gguf) |")
        print("|---|---:|---:|---:|")
        for r in results:
            q = r["mlm"]
            print(f"| {r['ftype']} | {100 * q['top1_agreement']:.2f}% | {100 * q['accuracy']:.2f}% | {q['kl_mean']:.5f} |")

    if all("sts" in r for r in results):
        s0 = results[0]["sts"]
        print(f"\n## STS-B test, mean-pooled embeddings ({s0['n_pairs']} pairs; "
              f"PyTorch fp32 Spearman {100 * s0['spearman_gold_ref']:.2f})\n")
        print("| file type | Spearman vs gold | Spearman vs reference sims | mean \\|Δ cos sim\\| | embedding cos mean | embedding cos min |")
        print("|---|---:|---:|---:|---:|---:|")
        for r in results:
            s = r["sts"]
            print(f"| {r['ftype']} | {100 * s['spearman_gold']:.2f} | {s['spearman_vs_ref']:.5f} | "
                  f"{s['mean_abs_sim_diff']:.5f} | {s['embedding_cos_mean']:.6f} | {s['embedding_cos_min']:.6f} |")
    bad = [r["file"] for r in results if not r["tokenizer_ok"]]
    print("\nTokenizer embedded in the GGUF matches the Hugging Face tokenizer: "
          + ("yes" if not bad else f"NO ({', '.join(bad)})"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE, help="Hub repo id or local directory")
    ap.add_argument("--lengths", type=int, nargs="+", default=[16, 64, 128, 512, 2048])
    ap.add_argument("--per-length", type=int, default=12, help="documents per length")
    ap.add_argument("--mlm-max-len", type=int, default=512)
    ap.add_argument("--sts", type=int, default=0, help="STS-B test pairs to use (0 = all 1379, -1 = skip)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-int8", action="store_true",
                    help="run quantized files on ggml's own kernels instead of the int8 GEMM")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write the raw numbers here")
    args = ap.parse_args()

    ref = Reference(args.reference, args.threads or None)
    docs = build_docs(ref.tokenizer, args.lengths, args.per_length)
    sts = None if args.sts < 0 else load_sts(args.sts or None)
    print(f"reference: {args.reference} | {len(docs)} documents "
          f"({args.per_length} each at T = {args.lengths})"
          + (f" | {sts['n']} STS-B pairs" if sts else ""), flush=True)
    ref_out = reference_outputs(ref, docs, sts, args.mlm_max_len, args.seed)

    results = []
    for path in args.gguf:
        print(f"evaluating {path} ...", flush=True)
        results.append(evaluate(path, docs, ref_out, sts, ref.tokenizer, args.threads, not args.no_int8))
    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
