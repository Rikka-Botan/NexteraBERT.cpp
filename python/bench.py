#!/usr/bin/env python3
"""Latency of GGUF files vs PyTorch fp32 encoders, on this machine's CPU.

    python python/bench.py --gguf models/*-Q4_K_M.gguf models/*-Q8_0.gguf \
        --baselines google-bert/bert-base-uncased answerdotai/ModernBERT-base

Each cell is the median wall time of one forward pass over real text (AG News,
cut to exactly T tokens by the model's own tokenizer), tokenization excluded,
after warm-up. The PyTorch models -- NexteraBERT itself and any ``--baselines``
-- run in fp32 under ``torch.inference_mode()``; each is timed at every
``--torch-threads`` setting and its best result is reported, so a speed-up is
not an artefact of a bad thread count.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nexterabert_cpp import NexteraBERT  # noqa: E402
from verify import DEFAULT_REFERENCE, build_docs  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def time_call(fn, min_reps: int, budget_s: float) -> float:
    """Median milliseconds per call: 3 warm-up calls, then >= min_reps timed ones.

    Calls that take seconds get fewer of both (1 warm-up + 3 timed; past 20 s,
    1 + 2): at that size run-to-run noise is far below the differences measured.
    """
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > 20.0:
        min_reps, budget_s = 2, 0.0
    elif first > 2.0:
        min_reps, budget_s = 3, 0.0
    else:
        fn()
        fn()
    times, t_end = [], time.perf_counter() + budget_s
    while len(times) < min_reps or (time.perf_counter() < t_end and len(times) < 200):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def total_ram_gib() -> float:
    try:
        if sys.platform == "win32":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("total", "avail", "tp", "ap", "tv", "av", "ext")]

            st = MemoryStatus()
            st.length = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.total / 2**30
        import os

        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except Exception:
        return 16.0


def torch_skip_reason(model, T: int, B: int) -> str | None:
    """Why a PyTorch encoder cannot run B sequences of T tokens here, if it cannot.

    On CPU these models materialise the (heads, T, T) attention scores, so memory
    grows with T^2; trying a size that does not fit would only swap the machine
    to a halt, hence the estimate instead of an attempt.
    """
    cfg = model.config
    max_pos = getattr(cfg, "max_position_embeddings", None)
    if max_pos and T > max_pos:
        return f"position limit {max_pos}"
    heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None) or 12
    need = 2.0 * B * heads * T * T * 4 / 2**30          # scores + the softmax's copy
    if need > 0.6 * total_ram_gib():
        return f"out of memory (~{need:.0f} GiB of attention scores)"
    return None


def load_torch_encoder(repo: str):
    """``AutoModel`` in eval mode, plus its parameter count in millions.

    Some checkpoints are saved from a task class whose base-model prefix the bare
    AutoModel does not share (LiquidAI/LFM2.5-Encoder: keys are ``lfm2.*``), which
    silently leaves the encoder randomly initialised; those are opened through
    AutoModelForMaskedLM and its backbone is used instead.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    if hasattr(config, "reference_compile"):
        config.reference_compile = False        # ModernBERT: no torch.compile on the CPU path
    model, info = AutoModel.from_pretrained(repo, config=config, trust_remote_code=True,
                                            output_loading_info=True)
    if len(info["missing_keys"]) > 0.5 * len(list(model.state_dict())):
        mlm = AutoModelForMaskedLM.from_pretrained(repo, config=config, trust_remote_code=True)
        model = getattr(mlm, "lfm2", None) or mlm.base_model
    model.eval()
    return model, sum(p.numel() for p in model.parameters()) / 1e6


def extend_position_table(model, length: int) -> int | None:
    """Tile a learned absolute position table (BERT's 512 rows) up to ``length``, as Longformer
    does and as NexteraBERT's model_speedbench does: the compute at a given length is the same as
    with a natively longer table, so the timing is fair. Returns the original size if it grew."""
    import torch

    emb = getattr(model, "embeddings", None)
    table = getattr(emb, "position_embeddings", None)
    if table is None or table.weight.shape[0] >= length:
        return None
    old = table.weight.shape[0]
    new = torch.nn.Embedding(length, table.weight.shape[1])
    with torch.no_grad():
        new.weight.copy_(table.weight.repeat(-(-length // old), 1)[:length])
    emb.position_embeddings = new
    emb.register_buffer("position_ids", torch.arange(length).unsqueeze(0), persistent=False)
    if hasattr(emb, "token_type_ids"):
        emb.register_buffer("token_type_ids", torch.zeros((1, length), dtype=torch.long), persistent=False)
    model.config.max_position_embeddings = length
    return old


def time_torch(name: str, repo: str, args, cells) -> dict:
    """Best-over-thread-settings median latency of one PyTorch encoder."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model, params_m = load_torch_encoder(repo)
    extended_from = extend_position_table(model, max(args.lengths)) if args.extend_positions else None
    skipped = {f"{T}x{B}": why for T, B in cells if (why := torch_skip_reason(model, T, B))}
    runnable = sorted({T for T, B in cells if f"{T}x{B}" not in skipped})
    docs = {T: build_docs(tokenizer, [T], max(args.batch)) for T in runnable}
    for cell, why in skipped.items():
        print(f"{name:28s} {cell}: skipped, {why}", flush=True)
    default_threads = torch.get_num_threads()
    best: dict = {}
    for nt in args.torch_threads:
        nt = nt or default_threads
        torch.set_num_threads(nt)
        for T, B in cells:
            if f"{T}x{B}" in skipped:
                continue
            ids = torch.tensor(docs[T][:B])
            mask = torch.ones_like(ids)

            def run():
                with torch.inference_mode():
                    model(input_ids=ids, attention_mask=mask).last_hidden_state

            ms = time_call(run, args.min_reps, args.budget)
            print(f"{name:28s} threads={nt:2d}  T={T:4d} B={B}: {ms:8.2f} ms", flush=True)
            if (T, B) not in best or ms < best[(T, B)][0]:
                best[(T, B)] = (ms, nt)
    torch.set_num_threads(default_threads)
    return {"ms": {f"{T}x{B}": v[0] for (T, B), v in best.items()},
            "threads": {f"{T}x{B}": v[1] for (T, B), v in best.items()},
            "skipped": skipped, "torch": torch.__version__, "repo": repo, "params_m": params_m,
            "position_table_extended_from": extended_from}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", nargs="+", required=True)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE, help="Hub repo id or local directory")
    ap.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8], help="sequences per forward pass")
    ap.add_argument("--threads", type=int, default=0, help="threads for the GGUF runtime (0 = its default)")
    ap.add_argument("--torch-threads", type=int, nargs="+", default=[0],
                    help="thread counts to try for PyTorch (0 = its default); the best is the baseline")
    ap.add_argument("--min-reps", type=int, default=7)
    ap.add_argument("--budget", type=float, default=2.0, help="seconds of timing per cell")
    ap.add_argument("--baselines", nargs="*", default=[],
                    help="other Hugging Face encoders to time in PyTorch fp32 (repo ids)")
    ap.add_argument("--extend-positions", action="store_true",
                    help="tile learned position tables (BERT: 512) so those models are timed at every length")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--ggml-kernels", action="store_true",
                    help="also time each quantized file on ggml's own kernels (int8 GEMM off)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.reference)
    max_batch = max(args.batch)
    docs = {T: build_docs(tokenizer, [T], max_batch) for T in args.lengths}
    cells = [(T, B) for B in args.batch for T in args.lengths]
    results: dict[str, dict] = {}

    if not args.no_torch:
        results["PyTorch fp32"] = time_torch("PyTorch fp32", args.reference, args, cells)
        for repo in args.baselines:
            name = f"PyTorch {repo.split('/')[-1]}"
            results[name] = time_torch(name, repo, args, cells)

    runs = [(path, True) for path in args.gguf]
    if args.ggml_kernels:
        runs += [(path, False) for path in args.gguf]
    on_ggml = set()                  # files whose default run already used ggml's kernels
    for path, int8 in runs:
        m = NexteraBERT(path, n_threads=args.threads, int8=int8)
        uses_ggml = " 0 int8" in m.system_info
        if int8 and uses_ggml:
            on_ggml.add(path)
        if not int8 and (path in on_ggml or not uses_ggml):
            m.close()                # float file, or a build without the int8 GEMM: nothing new
            continue
        name = f"GGUF {m.ftype}" + ("" if int8 else " (ggml kernels)")
        row = {}
        for T, B in cells:
            batch = docs[T][:B]
            ms = time_call(lambda: m.encode_ids(batch), args.min_reps, args.budget)
            row[f"{T}x{B}"] = ms
            print(f"{name:28s} threads={m.n_threads:2d}  T={T:4d} B={B}: {ms:8.2f} ms", flush=True)
        results[name] = {"ms": row, "threads": m.n_threads, "system": m.system_info,
                         "size_mib": Path(path).stat().st_size / 2**20}
        m.close()

    if args.json:                    # before the report: the measurements are the expensive part
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    base = results.get("PyTorch fp32")
    for B in args.batch:
        print(f"\n## batch = {B}: milliseconds per forward pass" + (" (speed-up vs PyTorch fp32)" if base else "") + "\n")
        print("| engine | " + " | ".join(f"T={T}" for T in args.lengths) + " |")
        print("|---|" + "---:|" * len(args.lengths))
        for name, r in results.items():
            cols = []
            for T in args.lengths:
                ms = r["ms"].get(f"{T}x{B}")
                if ms is None:
                    cols.append("-")
                    continue
                ref = base["ms"].get(f"{T}x{B}") if base else None
                if ref and name != "PyTorch fp32":
                    cols.append(f"{ms:.1f} ({ref / ms:.2f}x)")
                else:
                    cols.append(f"{ms:.1f}")
            print(f"| {name} | " + " | ".join(cols) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
