#!/usr/bin/env python3
"""Convert a NexteraBERT (v1) Hugging Face checkpoint to GGUF.

    python python/convert_hf_to_gguf.py RikkaBotan/NexteraBERT-Mezzoforte-220M-en \
        --outfile models/nexterabert-mezzoforte-220m-en-f32.gguf

The output is a float GGUF (f32 or f16). Quantize it afterwards with
``nexterabert-quantize`` (the same split as llama.cpp's convert / llama-quantize).

Only numpy, safetensors and gguf are needed -- no torch. The file is fully
self-contained: hyperparameters, weights, the MLM head and the tokenizer
(``tokenizer.json`` verbatim under ``tokenizer.huggingface.json``).
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

# Never used at inference: the pooler is not trained by MLM pretraining (it only
# exists for sequence-classification fine-tunes) and the segment embeddings are
# zero unless a pair task trained them.
SKIP_PREFIXES = ("pooler.",)


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

    state = load_file(str(src / "model.safetensors"))
    tte = state.pop("token_type_embeddings.weight", None)
    if tte is not None and np.any(tte != 0):
        print("warning: token_type_embeddings are non-zero; the runtime ignores "
              "segment ids (single-segment inputs only)", file=sys.stderr)

    tensors: dict[str, np.ndarray] = {}
    for name, arr in state.items():
        if name.startswith(SKIP_PREFIXES):
            continue
        # Conv1d depthwise weights are (C, 1, k): drop the group axis.
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr[:, 0, :]
        tensors[name] = arr

    has_mlm = False
    mlm_path = src / "mlm_head.safetensors"
    if not args.no_mlm_head and mlm_path.exists():
        head = load_file(str(mlm_path))
        tensors["mlm.dense.weight"] = head["head.dense.weight"]
        tensors["mlm.norm.weight"] = head["head.norm.weight"]
        tensors["mlm.bias"] = head["lm_head.bias"]
        if "lm_head.weight" in head:          # untied decoder
            tensors["mlm.decoder.weight"] = head["lm_head.weight"]
        elif not cfg.get("tie_word_embeddings", True):
            raise SystemExit("untied MLM decoder weight is missing from mlm_head.safetensors")
        has_mlm = True

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
          f"{size:.1f} MiB, {args.outtype}, mlm_head={has_mlm})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
