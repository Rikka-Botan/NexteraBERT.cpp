#!/usr/bin/env python3
"""Check the pooler and the classifier heads of a GGUF against PyTorch.

    python python/verify_heads.py --types f32 Q8_0 Q4_K_M --json results/verify_heads.json

For each task class of the Hugging Face model

    AutoModelForSequenceClassification    AutoModelForMultipleChoice
    AutoModelForTokenClassification       AutoModelForQuestionAnswering

a checkpoint is built from the pretrained backbone, saved with ``save_pretrained``,
loaded back, converted with ``convert_hf_to_gguf.py`` (and quantized), and its
logits are compared with the PyTorch model's on single sentences, sentence pairs
with segment ids and a long document, all in one unpadded batch.

No fine-tuned NexteraBERT is published, so the heads are stand-ins: classifier and
transform keep their random initialisation and the segment embeddings are drawn at
random. That says nothing about task accuracy, but it exercises exactly the weights
and the arithmetic a fine-tuned checkpoint uses. An f32 file has to reproduce
PyTorch to float rounding; the script exits non-zero if one does not.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from nexterabert_cpp import NexteraBERT  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_REFERENCE = "RikkaBotan/NexteraBERT-Mezzoforte-220M-en"
F32_TOLERANCE = 1e-3            # max |logit difference| allowed for an f32 file

# task -> (Auto class, labels of the stand-in head; None: the class fixes them)
TASKS = {
    "sequence-classification": ("AutoModelForSequenceClassification", ["negative", "neutral", "positive"]),
    "multiple-choice": ("AutoModelForMultipleChoice", None),
    "token-classification": ("AutoModelForTokenClassification", ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]),
    "question-answering": ("AutoModelForQuestionAnswering", None),
}
PER_TOKEN = ("token-classification", "question-answering")

SINGLES = [
    "Hello world!",
    "The capital of France is Paris.",
    "NexteraBERT mixes convolution, sliding-window attention and full attention in one encoder.",
    "A much longer passage about something else entirely, with enough words that the hierarchical "
    "blocks and the sliding window both have something to do in this test of the task heads.",
]
PAIRS = [
    ("A man is playing a guitar.", "Someone plays an instrument."),
    ("Where is the Eiffel Tower?", "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."),
    ("The weather is cold today.", "It is the hottest day of the year."),
]
LONG = " ".join(SINGLES) + " "


def build_inputs(tokenizer) -> tuple[list[list[int]], list[list[int] | None]]:
    """Token ids and segment ids (None: the sequence is run without any)."""
    ids, types = [], []
    for text in SINGLES + [LONG * 12]:
        ids.append(tokenizer(text)["input_ids"])
        types.append(None)
    for a, b in PAIRS:
        enc = tokenizer(a, b)
        # BERT convention: [CLS] A [SEP] -> 0, B [SEP] -> 1
        first_sep = enc["input_ids"].index(tokenizer.sep_token_id)
        ids.append(enc["input_ids"])
        types.append([0] * (first_sep + 1) + [1] * (len(enc["input_ids"]) - first_sep - 1))
    return ids, types


def make_checkpoint(task: str, reference: str, out: Path, seed: int):
    """Build the stand-in fine-tuned model, save it the Hugging Face way, load it back."""
    import torch
    import transformers

    auto = getattr(transformers, TASKS[task][0])
    labels = TASKS[task][1]
    kwargs = {}
    if labels:
        kwargs = {"num_labels": len(labels), "id2label": dict(enumerate(labels)),
                  "label2id": {n: i for i, n in enumerate(labels)}}
    torch.manual_seed(seed)
    model = auto.from_pretrained(reference, trust_remote_code=True, **kwargs)
    with torch.no_grad():
        model.encoder.token_type_embeddings.weight.normal_(0.0, 0.02)
    model.save_pretrained(out)
    transformers.AutoTokenizer.from_pretrained(reference).save_pretrained(out)
    if Path(reference).is_dir():        # a local reference keeps its code next to the weights
        for f in Path(reference).glob("*.py"):
            shutil.copy(f, out / f.name)
    # Loading back is part of the test: the saved config must restore the head's shape.
    return auto.from_pretrained(out, trust_remote_code=True).eval()


def reference_logits(model, task: str, ids, types) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """(logits, pooler_output) per sequence, each run alone with an all-ones mask."""
    import torch

    logits, pooled = [], []
    with torch.inference_mode():
        for seq, tt in zip(ids, types):
            x = torch.tensor([seq])
            t = None if tt is None else torch.tensor([tt])
            mask = torch.ones_like(x)
            pooled.append(model.encoder(x, None, mask, t)[1][0].numpy())
            if task == "multiple-choice":       # (batch, choices, length)
                out = model(input_ids=x[None], attention_mask=mask[None],
                            token_type_ids=None if t is None else t[None])
                logits.append(out.logits.reshape(1).numpy())
            elif task == "question-answering":
                out = model(input_ids=x, attention_mask=mask, token_type_ids=t)
                logits.append(torch.stack([out.start_logits[0], out.end_logits[0]], -1).numpy())
            else:
                logits.append(model(input_ids=x, attention_mask=mask, token_type_ids=t).logits[0].numpy())
    return logits, pooled


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_gguf(model: NexteraBERT, ids, types) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Logits and pooler_output from two unpadded batches: with and without segment ids."""
    logits: list = [None] * len(ids)
    pooled = np.zeros((len(ids), model.n_embd), dtype=np.float32) if model.has_pooler else None
    for with_types in (False, True):
        idx = [i for i, t in enumerate(types) if (t is not None) == with_types]
        if not idx:
            continue
        tt = [types[i] for i in idx] if with_types else None
        out = model.logits_ids([ids[i] for i in idx], tt)
        for k, i in enumerate(idx):
            logits[i] = np.atleast_1d(out[k])
        if pooled is not None:
            pooled[idx] = model.pooler_output([ids[i] for i in idx], tt)
    return logits, pooled


def evaluate(path: Path, task: str, ids, types, ref_logits, ref_pooled, threads: int) -> dict:
    m = NexteraBERT(path, n_threads=threads)
    assert m.task == task, (m.task, task)
    logits, pooled = run_gguf(m, ids, types)
    ref = np.concatenate([r.reshape(-1, m.n_labels) for r in ref_logits])
    got = np.concatenate([g.reshape(-1, m.n_labels) for g in logits])
    res = {"file": path.name, "ftype": m.ftype, "task": task, "n_labels": m.n_labels, "labels": m.labels,
           "n_token_types": m.n_token_types, "rows": int(len(ref)),
           "logit_max_abs_diff": float(np.abs(ref - got).max()),
           "logit_cos": cosine(ref, got)}
    if m.n_labels > 1:
        res["argmax_agreement"] = float((ref.argmax(-1) == got.argmax(-1)).mean())
    if pooled is not None:
        res["pooler_cos_min"] = min(cosine(r, p) for r, p in zip(ref_pooled, pooled))
    m.close()
    return res


def print_report(results: list[dict]) -> None:
    print("\n## Classifier heads vs PyTorch fp32\n")
    print("| task | file type | labels | logit rows | max \\|Δ logit\\| | logit cosine | same argmax | pooler cosine (min) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        same = f"{100 * r['argmax_agreement']:.1f}%" if "argmax_agreement" in r else "-"
        pool = f"{r['pooler_cos_min']:.6f}" if "pooler_cos_min" in r else "-"
        print(f"| {r['task']} | {r['ftype']} | {r['n_labels']} | {r['rows']} | {r['logit_max_abs_diff']:.2e} | "
              f"{r['logit_cos']:.6f} | {same} | {pool} |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE, help="Hub repo id or local directory")
    ap.add_argument("--tasks", nargs="+", choices=list(TASKS), default=list(TASKS))
    ap.add_argument("--types", nargs="+", default=["f32"],
                    help="file types to test: f32 and any nexterabert-quantize type")
    ap.add_argument("--quantize", type=Path, default=ROOT / "build" / "bin" / "nexterabert-quantize",
                    help="the nexterabert-quantize executable")
    ap.add_argument("--workdir", type=Path, default=None, help="where checkpoints and GGUFs go (default: temporary)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write the raw numbers here")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tmp = None if args.workdir else tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    work = args.workdir or Path(tmp.name)
    work.mkdir(parents=True, exist_ok=True)
    ids, types = build_inputs(AutoTokenizer.from_pretrained(args.reference))
    print(f"reference: {args.reference} | {len(ids)} sequences of {min(map(len, ids))} to "
          f"{max(map(len, ids))} tokens, {sum(t is not None for t in types)} with segment ids", flush=True)

    results = []
    for task in args.tasks:
        print(f"{task}: building the checkpoint ...", flush=True)
        ckpt = work / task
        model = make_checkpoint(task, args.reference, ckpt, args.seed)
        ref_logits, ref_pooled = reference_logits(model, task, ids, types)
        del model
        f32 = work / f"{task}-f32.gguf"
        subprocess.run([sys.executable, str(ROOT / "python" / "convert_hf_to_gguf.py"), str(ckpt),
                        "--outfile", str(f32)], check=True)
        shutil.rmtree(ckpt, ignore_errors=True)
        for ftype in args.types:
            path = f32
            if ftype.lower() != "f32":
                path = work / f"{task}-{ftype}.gguf"
                subprocess.run([str(args.quantize), str(f32), str(path), ftype], check=True,
                               stdout=subprocess.DEVNULL)
            results.append(evaluate(path, task, ids, types, ref_logits, ref_pooled, args.threads))
            if path != f32:
                path.unlink()
        f32.unlink()

    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    bad = [r for r in results if r["ftype"] == "F32" and r["logit_max_abs_diff"] > F32_TOLERANCE]
    for r in bad:
        print(f"FAILED: {r['task']} f32 differs from PyTorch by {r['logit_max_abs_diff']:.2e}")
    if tmp:
        tmp.cleanup()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
