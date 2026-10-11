#!/usr/bin/env python3
"""Convert a NexteraBERT (v1) Hugging Face checkpoint to GGUF.

    python python/convert_hf_to_gguf.py RikkaBotan/NexteraBERT-Mezzoforte-220M-en \
        --outfile models/nexterabert-mezzoforte-220m-en-f32.gguf

Both checkpoint layouts are read: the pretrained backbone (bare encoder keys, the
masked-LM head in ``mlm_head.safetensors``) and a model fine-tuned and saved with
``save_pretrained`` (``encoder.``-prefixed keys plus its task head). The head of a
fine-tuned ``AutoModelForSequenceClassification`` / ``MultipleChoice`` /
``TokenClassification`` / ``QuestionAnswering`` / ``MaskedLM`` checkpoint is
converted with it; ``--task`` overrides what ``config.json`` says it is.

The output is a float GGUF (f32 or f16). Quantize it afterwards with
``nexterabert-quantize`` (the same split as llama.cpp's convert / llama-quantize).

Only numpy, safetensors and gguf are needed -- no torch. The file is fully
self-contained: hyperparameters, weights, the MLM head and the tokenizer
(``tokenizer.json`` verbatim under ``tokenizer.huggingface.json``) and, for a
classifier, its task and label names.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

import gguf

ARCH = "nexterabert"
SUPPORTED_MIXERS = ("lily", "swa", "attn", "hra")

ENCODER_PREFIX = "encoder."

# classifier task -> (config.json "architectures" marker, tensors its head needs)
POOLED_HEAD = ("pooler.dense.weight", "pooler.norm.weight", "classifier.weight", "classifier.bias")
TOKEN_HEAD = ("head.dense.weight", "head.norm.weight", "classifier.weight", "classifier.bias")
TASKS = {
    "sequence-classification": ("ForSequenceClassification", POOLED_HEAD),
    "multiple-choice": ("ForMultipleChoice", POOLED_HEAD),
    "token-classification": ("ForTokenClassification", TOKEN_HEAD),
    "question-answering": ("ForQuestionAnswering", TOKEN_HEAD),
}


def split_checkpoint(state: dict) -> tuple[dict, dict]:
    """(encoder tensors under their bare names, everything else)."""
    if not any(k.startswith(ENCODER_PREFIX) for k in state):
        return dict(state), {}
    enc = {k[len(ENCODER_PREFIX):]: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}
    return enc, {k: v for k, v in state.items() if not k.startswith(ENCODER_PREFIX)}


def detect_task(cfg: dict, head: dict) -> str | None:
    if "classifier.weight" not in head:
        return None
    arch = " ".join(cfg.get("architectures") or [])
    for task, (marker, _) in TASKS.items():
        if marker in arch:
            return task
    # no architecture recorded: only the per-token heads have a transform of their own
    return "token-classification" if "head.dense.weight" in head else "sequence-classification"


def label_names(cfg: dict, n_labels: int) -> list[str]:
    id2label = cfg.get("id2label") or {}
    names = [str(id2label.get(str(i), id2label.get(i, ""))) for i in range(n_labels)]
    return names if len(id2label) == n_labels and all(names) else []


def resolve_model_dir(model: str, revision: str | None) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(
        model, revision=revision,
        allow_patterns=["config.json", "model.safetensors", "mlm_head.safetensors",
                        "tokenizer.json", "tokenizer_config.json"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="local directory or Hub repo id")
    ap.add_argument("--outfile", required=True, type=Path)
    ap.add_argument("--outtype", choices=("f32", "f16"), default="f32",
                    help="storage type of the 2-D weights (default: f32)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--no-mlm-head", action="store_true",
                    help="leave the masked-LM head out (embedding-only file)")
    ap.add_argument("--task", choices=("auto", "none", *TASKS), default="auto",
                    help="classifier head to convert (default: what the checkpoint holds)")
    ap.add_argument("--pooler", action="store_true",
                    help="keep the pooler without a classifier on it (for pooler_output; "
                         "the pooler of a pretrained backbone is untrained)")
    args = ap.parse_args()

    src = resolve_model_dir(args.model, args.revision)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))

    block_types = list(cfg["block_types"])
    bad = sorted(set(block_types) - set(SUPPORTED_MIXERS))
    if bad:
        raise SystemExit(f"unsupported block_types {bad}; supported: {SUPPORTED_MIXERS}")
    if str(cfg.get("mlp_class", "mlp")).lower() != "mlp":
        raise SystemExit("only mlp_class='mlp' is supported")
    if int(cfg.get("expand", 1)) * cfg["n_embd"] % cfg["n_head"]:
        raise SystemExit("expand * n_embd must be divisible by n_head")

    state, head = split_checkpoint(load_file(str(src / "model.safetensors")))
    if "embedding.weight" not in state:
        raise SystemExit(f"{src / 'model.safetensors'} is not a NexteraBERT checkpoint: "
                         "it has neither 'embedding.weight' nor 'encoder.embedding.weight'")

    task = detect_task(cfg, head) if args.task == "auto" else None if args.task == "none" else args.task
    pooled_task = task is not None and TASKS[task][1] is POOLED_HEAD

    tensors: dict[str, np.ndarray] = {}
    for name, arr in state.items():
        # The pooler is only trained by a task that classifies its output.
        if name.startswith("pooler.") and not (pooled_task or args.pooler):
            continue
        # Segment embeddings are zero until a pair task trains them, and add nothing then.
        if name == "token_type_embeddings.weight" and not np.any(arr != 0):
            continue
        # Conv1d depthwise weights are (C, 1, k): drop the group axis.
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
        tensors[name] = arr
    if args.pooler and "pooler.dense.weight" not in tensors:
        raise SystemExit("--pooler: the checkpoint has no pooler")

    labels: list[str] = []
    if task is not None:
        missing = [n for n in TASKS[task][1] if n not in head and n not in tensors]
        if missing:
            raise SystemExit(f"task '{task}' needs {missing}, which the checkpoint does not have")
        for name in TASKS[task][1]:
            if name in head:
                tensors[name] = head.pop(name)
        labels = label_names(cfg, tensors["classifier.bias"].shape[0])

    # The masked-LM head: next to the backbone, or inside a saved AutoModelForMaskedLM.
    has_mlm = False
    mlm = None
    if task is None and "lm_head.bias" in head:
        mlm = {k: head.pop(k) for k in list(head) if k.startswith(("head.", "lm_head."))}
    elif (src / "mlm_head.safetensors").exists():
        mlm = load_file(str(src / "mlm_head.safetensors"))
    if mlm is not None and not args.no_mlm_head:
        tensors["mlm.dense.weight"] = mlm["head.dense.weight"]
        tensors["mlm.norm.weight"] = mlm["head.norm.weight"]
        tensors["mlm.bias"] = mlm["lm_head.bias"]
        if not cfg.get("tie_word_embeddings", True):      # untied decoder
            if "lm_head.weight" not in mlm:
                raise SystemExit("untied MLM decoder weight is missing from the checkpoint")
            tensors["mlm.decoder.weight"] = mlm["lm_head.weight"]
        has_mlm = True

    if args.task == "none":
        head = {k: v for k, v in head.items() if not k.startswith(("classifier.", "head."))}
    if head:
        raise SystemExit(f"tensors this converter does not know: {sorted(head)}")

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(args.outfile), ARCH)
    name = Path(args.model).name if not Path(args.model).is_dir() else src.name
    w.add_name(name)
    w.add_file_type(gguf.LlamaFileType.ALL_F32 if args.outtype == "f32"
                    else gguf.LlamaFileType.MOSTLY_F16)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)

    n_embd, n_head = int(cfg["n_embd"]), int(cfg["n_head"])
    k = f"{ARCH}."
    w.add_uint32(k + "vocab_size", int(cfg["vocab_size"]))
    w.add_uint32(k + "embedding_length", n_embd)
    w.add_uint32(k + "block_count", int(cfg["n_layer"]))
    w.add_uint32(k + "feed_forward_length", int(cfg["n_inter"]))
    w.add_uint32(k + "attention.head_count", n_head)
    w.add_uint32(k + "attention.head_count_kv", int(cfg["n_kv_head"]))
    w.add_array(k + "block_types", block_types)
    w.add_uint32(k + "lily.low_rank", int(cfg.get("expand", 1)) * n_embd // n_head)
    w.add_uint32(k + "lily.kernel_size", int(cfg["n_kernel"]))
    w.add_uint32(k + "hra.block_size", int(cfg["block_size"]))
    w.add_uint32(k + "swa.window", int(cfg.get("swa_window", 256)))
    w.add_float32(k + "rope.freq_base", float(cfg.get("rope_base", 100000)))
    w.add_float32(k + "swa.rope.freq_base", float(cfg.get("swa_rope_base", 10000)))
    w.add_uint32(k + "rope.ntk_train_length", int(cfg.get("rope_ntk_train_len", 0) or 0))
    w.add_string(k + "ssmax.mode", str(cfg.get("ssmax_mode", "sssmax")).lower())
    w.add_float32(k + "ssmax.base", float(cfg.get("ssmax_base", 0.1)))
    w.add_float32(k + "rms_norm_epsilon", 1e-8)      # RMSNorm default in the model code
    w.add_float32(k + "layer_norm_epsilon", 1e-5)    # nn.LayerNorm default
    w.add_bool(k + "mlm_head", has_mlm)
    if task is not None:
        w.add_string(k + "classifier.task", task)
        if labels:
            w.add_array(k + "classifier.labels", labels)

    w.add_string("tokenizer.huggingface.json",
                 (src / "tokenizer.json").read_text(encoding="utf-8"))
    w.add_uint32("tokenizer.ggml.cls_token_id", int(cfg["cls_token_id"]))
    w.add_uint32("tokenizer.ggml.seperator_token_id", int(cfg["sep_token_id"]))
    w.add_uint32("tokenizer.ggml.padding_token_id", int(cfg["pad_token_id"]))
    w.add_uint32("tokenizer.ggml.mask_token_id", int(cfg["mask_token_id"]))

    n_params = 0
    for tname in sorted(tensors):
        arr = tensors[tname]
        n_params += arr.size
        if arr.ndim == 0:
            arr = arr.reshape(1)
        # Only the GEMM weights are worth shrinking; norms, DyT curves, conv
        # kernels and scalars stay f32 (they are < 0.1% of the file).
        is_matrix = arr.ndim == 2 and tname.endswith(".weight") and min(arr.shape) >= 32
        dtype = np.float16 if (args.outtype == "f16" and is_matrix) else np.float32
        w.add_tensor(tname, np.ascontiguousarray(arr.astype(dtype)))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    size = args.outfile.stat().st_size / 2**20
    print(f"wrote {args.outfile}  ({len(tensors)} tensors, {n_params/1e6:.1f}M params, "
          f"{size:.1f} MiB, {args.outtype}, mlm_head={has_mlm}, classifier={task}"
          + (f" ({tensors['classifier.bias'].shape[0]} labels)" if task else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
