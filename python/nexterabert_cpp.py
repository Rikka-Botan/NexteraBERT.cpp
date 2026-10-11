"""Python binding for nexterabert.cpp (ctypes over the C API in src/nexterabert.h).

    from nexterabert_cpp import NexteraBERT

    model = NexteraBERT("models/nexterabert-mezzoforte-220m-en-Q4_K_M.gguf")
    emb = model.encode(["first text", "second text"])        # (2, 1024), L2-normalised
    print(model.fill_mask("The capital of France is [MASK]."))

    clf = NexteraBERT("models/my-finetuned-classifier-Q4_K_M.gguf")
    print(clf.classify(["a great movie", "a dull one"]))     # [(label, probability), ...]

The GGUF carries its own tokenizer.json and label names, so the file is all that is needed.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class _Params(ctypes.Structure):
    _fields_ = [("n_threads", ctypes.c_int32),
                ("use_int8", ctypes.c_int32),
                ("use_repack", ctypes.c_int32),
                ("verbosity", ctypes.c_int32)]


def _find_library(lib_path: str | os.PathLike | None) -> Path:
    names = {"win32": ["nexterabert.dll"], "darwin": ["libnexterabert.dylib"]}.get(
        sys.platform, ["libnexterabert.so"])
    candidates = []
    if lib_path:
        candidates.append(Path(lib_path))
    if os.environ.get("NEXTERABERT_LIB"):
        candidates.append(Path(os.environ["NEXTERABERT_LIB"]))
    root = Path(__file__).resolve().parent.parent
    for sub in ("build/bin", "build", "build/Release", "build/bin/Release"):
        candidates += [root / sub / n for n in names]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "nexterabert shared library not found. Build it first (cmake -B build && "
        "cmake --build build --config Release) or set NEXTERABERT_LIB.")


def _load_library(lib_path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_library(lib_path)))
    c_ctx, i32 = ctypes.c_void_p, ctypes.c_int32
    p_i32, p_f32 = ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float)
    sigs = {
        "nextera_default_params": ([], _Params),
        "nextera_load": ([ctypes.c_char_p, _Params], c_ctx),
        "nextera_free": ([c_ctx], None),
        "nextera_n_embd": ([c_ctx], i32),
        "nextera_n_vocab": ([c_ctx], i32),
        "nextera_n_layer": ([c_ctx], i32),
        "nextera_has_mlm_head": ([c_ctx], i32),
        "nextera_has_pooler": ([c_ctx], i32),
        "nextera_n_token_types": ([c_ctx], i32),
        "nextera_classifier_task": ([c_ctx], ctypes.c_char_p),
        "nextera_n_labels": ([c_ctx], i32),
        "nextera_label": ([c_ctx, i32], ctypes.c_char_p),
        "nextera_token_cls": ([c_ctx], i32),
        "nextera_token_sep": ([c_ctx], i32),
        "nextera_token_pad": ([c_ctx], i32),
        "nextera_token_mask": ([c_ctx], i32),
        "nextera_tokenizer_json": ([c_ctx], ctypes.c_char_p),
        "nextera_ftype": ([c_ctx], ctypes.c_char_p),
        "nextera_system_info": ([c_ctx], ctypes.c_char_p),
        "nextera_set_n_threads": ([c_ctx, i32], None),
        "nextera_get_n_threads": ([c_ctx], i32),
        "nextera_encode": ([c_ctx, p_i32, p_i32, i32, p_f32, p_f32, i32], i32),
        "nextera_encode_segments": ([c_ctx, p_i32, p_i32, p_i32, i32, p_f32, p_f32, i32], i32),
        "nextera_pooler_output": ([c_ctx, p_i32, p_i32, p_i32, i32, p_f32], i32),
        "nextera_classify": ([c_ctx, p_i32, p_i32, p_i32, i32, p_f32], i32),
        "nextera_mlm_logits": ([c_ctx, p_i32, i32, p_i32, i32, p_f32], i32),
        "nextera_last_ms": ([c_ctx], ctypes.c_double),
        "nextera_debug_set_layers": ([c_ctx, i32], None),
    }
    for name, (argtypes, restype) in sigs.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argtypes, restype
    return lib


def _as_i32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.int32)


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# classifier tasks whose logits are one row per token rather than per sequence
_PER_TOKEN_TASKS = ("token-classification", "question-answering")


class NexteraBERT:
    """A NexteraBERT GGUF loaded for CPU inference."""

    def __init__(self, model_path: str | os.PathLike, n_threads: int = 0, int8: bool = True,
                 use_repack: bool = True, verbose: bool = False, lib_path=None):
        self._lib = _load_library(lib_path)
        params = self._lib.nextera_default_params()
        params.n_threads = int(n_threads)
        params.use_int8 = int(bool(int8))
        params.use_repack = int(bool(use_repack))
        params.verbosity = int(bool(verbose))
        self._ctx = self._lib.nextera_load(os.fsencode(str(model_path)), params)
        if not self._ctx:
            raise RuntimeError(f"failed to load {model_path}")
        self.n_embd = self._lib.nextera_n_embd(self._ctx)
        self.n_vocab = self._lib.nextera_n_vocab(self._ctx)
        self.n_layer = self._lib.nextera_n_layer(self._ctx)
        self.has_mlm_head = bool(self._lib.nextera_has_mlm_head(self._ctx))
        self.mask_token_id = self._lib.nextera_token_mask(self._ctx)
        self.has_pooler = bool(self._lib.nextera_has_pooler(self._ctx))
        self.n_token_types = self._lib.nextera_n_token_types(self._ctx)
        #: "" or "sequence-classification" / "multiple-choice" / "token-classification" / "question-answering"
        self.task = self._lib.nextera_classifier_task(self._ctx).decode()
        self.n_labels = self._lib.nextera_n_labels(self._ctx)
        names = [self._lib.nextera_label(self._ctx, i).decode("utf-8") for i in range(self.n_labels)]
        #: label names of the classifier (``LABEL_i`` when the checkpoint named none)
        self.labels = [n or f"LABEL_{i}" for i, n in enumerate(names)]
        self.ftype = self._lib.nextera_ftype(self._ctx).decode()
        self._tokenizer = None

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.nextera_free(self._ctx)
            self._ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- info --------------------------------------------------------------
    @property
    def system_info(self) -> str:
        return self._lib.nextera_system_info(self._ctx).decode()

    @property
    def n_threads(self) -> int:
        return self._lib.nextera_get_n_threads(self._ctx)

    @n_threads.setter
    def n_threads(self, n: int) -> None:
        self._lib.nextera_set_n_threads(self._ctx, int(n))

    @property
    def last_ms(self) -> float:
        """Wall time of the last forward pass inside the library."""
        return self._lib.nextera_last_ms(self._ctx)

    # -- tokenizer ---------------------------------------------------------
    @property
    def tokenizer(self):
        """The model's Hugging Face tokenizer, rebuilt from the GGUF."""
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            js = self._lib.nextera_tokenizer_json(self._ctx).decode("utf-8")
            if not js:
                raise RuntimeError("this GGUF has no embedded tokenizer")
            tok = Tokenizer.from_str(js)
            tok.no_padding()
            tok.no_truncation()
            self._tokenizer = tok
        return self._tokenizer

    def _encodings(self, texts, text_pairs=None, max_length: int | None = None):
        tok = self.tokenizer
        if max_length:
            tok.enable_truncation(max_length=max_length)
        else:
            tok.no_truncation()
        texts = list(texts)
        if text_pairs is None:
            return tok.encode_batch(texts)
        text_pairs = list(text_pairs)
        if len(text_pairs) != len(texts):
            raise ValueError("texts and text_pairs differ in length")
        return tok.encode_batch(list(zip(texts, text_pairs)))

    def tokenize(self, texts: Iterable[str], max_length: int | None = None,
                 text_pairs: Iterable[str] | None = None) -> list[list[int]]:
        """Token ids of each text, or of each ``(text, pair)`` joined by the tokenizer."""
        return [e.ids for e in self._encodings(texts, text_pairs, max_length)]

    # -- inference ---------------------------------------------------------
    @staticmethod
    def _batch(ids, token_type_ids):
        """(tokens, segment ids or None, lengths) as the C API takes a batch."""
        if len(ids) == 0:
            raise ValueError("empty batch")
        lengths = _as_i32([len(s) for s in ids])
        tokens = _as_i32(np.concatenate([np.asarray(s, dtype=np.int32) for s in ids]))
        types = None
        if token_type_ids is not None:
            if [len(s) for s in token_type_ids] != lengths.tolist():
                raise ValueError("token_type_ids must have the shape of ids")
            types = _as_i32(np.concatenate([np.asarray(s, dtype=np.int32) for s in token_type_ids]))
        return tokens, types, lengths

    def encode_ids(self, ids: Sequence[Sequence[int]], normalize: bool = True,
                   return_hidden: bool = False,
                   token_type_ids: Sequence[Sequence[int]] | None = None):
        """One forward pass over a batch of token-id sequences (no padding needed).

        Returns the pooled embeddings ``(n, n_embd)``; with ``return_hidden`` also a
        list with each sequence's ``(len, n_embd)`` last hidden state.
        ``token_type_ids`` are the segment ids of a model fine-tuned with them.
        """
        tokens, types, lengths = self._batch(ids, token_type_ids)
        pooled = np.empty((len(ids), self.n_embd), dtype=np.float32)
        hidden = np.empty((int(lengths.sum()), self.n_embd), dtype=np.float32) if return_hidden else None
        rc = self._lib.nextera_encode_segments(
            self._ctx, _ptr(tokens, ctypes.c_int32),
            _ptr(types, ctypes.c_int32) if types is not None else None,
            _ptr(lengths, ctypes.c_int32), len(ids),
            _ptr(hidden, ctypes.c_float) if return_hidden else None,
            _ptr(pooled, ctypes.c_float), int(bool(normalize)))
        if rc != 0:
            raise RuntimeError(f"nextera_encode failed ({rc})")
        if not return_hidden:
            return pooled
        offs = np.concatenate([[0], np.cumsum(lengths)])
        return pooled, [hidden[offs[i]:offs[i + 1]] for i in range(len(ids))]

    def encode(self, texts: str | Sequence[str], batch_size: int = 8, normalize: bool = True,
               max_length: int | None = 8192) -> np.ndarray:
        """Mean-pooled sentence embeddings, ``(n, n_embd)`` (or ``(n_embd,)`` for a str)."""
        single = isinstance(texts, str)
        ids = self.tokenize([texts] if single else texts, max_length=max_length)
        # Group similar lengths so a batch's rows are spent on real tokens of
        # comparable sequences; results do not depend on the grouping.
        order = np.argsort([len(s) for s in ids], kind="stable")
        out = np.empty((len(ids), self.n_embd), dtype=np.float32)
        for i in range(0, len(ids), batch_size):
            idx = order[i:i + batch_size]
            out[idx] = self.encode_ids([ids[j] for j in idx], normalize=normalize)
        return out[0] if single else out

    def mlm_logits(self, ids: Sequence[int], positions: Sequence[int]) -> np.ndarray:
        """Masked-LM logits ``(len(positions), n_vocab)`` for one token sequence."""
        tokens, pos = _as_i32(ids), _as_i32(positions)
        logits = np.empty((len(pos), self.n_vocab), dtype=np.float32)
        rc = self._lib.nextera_mlm_logits(
            self._ctx, _ptr(tokens, ctypes.c_int32), len(tokens),
            _ptr(pos, ctypes.c_int32), len(pos), _ptr(logits, ctypes.c_float))
        if rc != 0:
            raise RuntimeError(f"nextera_mlm_logits failed ({rc})")
        return logits

    def fill_mask(self, text: str, top_k: int = 5) -> list[list[tuple[str, float]]]:
        """Top-k (token, probability) candidates for every [MASK] in ``text``."""
        ids = self.tokenize([text])[0]
        positions = [i for i, t in enumerate(ids) if t == self.mask_token_id]
        if not positions:
            raise ValueError("no [MASK] token in the input")
        logits = self.mlm_logits(ids, positions).astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1, keepdims=True)
        out = []
        for row in probs:
            top = np.argsort(-row)[:top_k]
            out.append([(self.tokenizer.decode([int(t)]), float(row[t])) for t in top])
        return out

    # -- pooler and classifier heads ---------------------------------------
    def pooler_output(self, ids: Sequence[Sequence[int]],
                      token_type_ids: Sequence[Sequence[int]] | None = None) -> np.ndarray:
        """``pooler_output`` of the Hugging Face model, ``(n, n_embd)``: the pooler's
        transform of each sequence's mean hidden state."""
        if not self.has_pooler:
            raise RuntimeError("this GGUF has no pooler (convert with --pooler, or a pooled classifier)")
        tokens, types, lengths = self._batch(ids, token_type_ids)
        out = np.empty((len(ids), self.n_embd), dtype=np.float32)
        rc = self._lib.nextera_pooler_output(
            self._ctx, _ptr(tokens, ctypes.c_int32),
            _ptr(types, ctypes.c_int32) if types is not None else None,
            _ptr(lengths, ctypes.c_int32), len(ids), _ptr(out, ctypes.c_float))
        if rc != 0:
            raise RuntimeError(f"nextera_pooler_output failed ({rc})")
        return out

    def logits_ids(self, ids: Sequence[Sequence[int]],
                   token_type_ids: Sequence[Sequence[int]] | None = None):
        """Classifier logits of a batch of token-id sequences, in one forward pass.

        ``(n, n_labels)`` for sequence classification and multiple choice (one score
        per sequence there: a question's choices are the sequences to compare); a
        list of ``(len, n_labels)`` arrays for token classification and question
        answering (columns: span start, span end).
        """
        if not self.task:
            raise RuntimeError("this GGUF has no classifier head")
        tokens, types, lengths = self._batch(ids, token_type_ids)
        per_token = self.task in _PER_TOKEN_TASKS
        out = np.empty((int(lengths.sum()) if per_token else len(ids), self.n_labels), dtype=np.float32)
        rc = self._lib.nextera_classify(
            self._ctx, _ptr(tokens, ctypes.c_int32),
            _ptr(types, ctypes.c_int32) if types is not None else None,
            _ptr(lengths, ctypes.c_int32), len(ids), _ptr(out, ctypes.c_float))
        if rc != 0:
            raise RuntimeError(f"nextera_classify failed ({rc})")
        if not per_token:
            return out
        offs = np.concatenate([[0], np.cumsum(lengths)])
        return [out[offs[i]:offs[i + 1]] for i in range(len(ids))]

    def _require_task(self, *tasks: str) -> None:
        if self.task not in tasks:
            raise RuntimeError(f"this GGUF's head is '{self.task or 'none'}', not {' / '.join(tasks)}")

    def logits(self, texts: str | Sequence[str], text_pairs: str | Sequence[str] | None = None,
               batch_size: int = 8, max_length: int | None = 8192) -> np.ndarray:
        """Sequence-classification logits ``(n, n_labels)`` (``(n_labels,)`` for a str).

        ``text_pairs`` gives each text a second segment (NLI, paraphrase, ...).
        """
        self._require_task("sequence-classification", "multiple-choice")
        single = isinstance(texts, str)
        if single:
            texts, text_pairs = [texts], None if text_pairs is None else [text_pairs]
        ids = self.tokenize(texts, max_length=max_length, text_pairs=text_pairs)
        order = np.argsort([len(s) for s in ids], kind="stable")
        out = np.empty((len(ids), self.n_labels), dtype=np.float32)
        for i in range(0, len(ids), batch_size):
            idx = order[i:i + batch_size]
            out[idx] = self.logits_ids([ids[j] for j in idx])
        return out[0] if single else out

    def classify(self, texts: str | Sequence[str], text_pairs: str | Sequence[str] | None = None,
                 batch_size: int = 8, max_length: int | None = 8192):
        """``(label, probability)`` of the most likely class of each text.

        A single-label head (regression, e.g. STS-B) returns its raw score instead.
        """
        self._require_task("sequence-classification")
        single = isinstance(texts, str)
        logits = np.atleast_2d(self.logits(texts, text_pairs, batch_size, max_length))
        if self.n_labels == 1:
            out = [float(v) for v in logits[:, 0]]
        else:
            probs = _softmax(logits)
            out = [(self.labels[int(i)], float(p[i])) for p, i in zip(probs, probs.argmax(-1))]
        return out[0] if single else out

    def choose(self, prompt: str, choices: Sequence[str], max_length: int | None = 8192) -> np.ndarray:
        """Multiple choice: probabilities of ``choices``, each scored as the pair
        ``(prompt, choice)``."""
        self._require_task("multiple-choice")
        ids = self.tokenize([prompt] * len(choices), max_length=max_length, text_pairs=choices)
        return _softmax(self.logits_ids(ids)[:, 0])

    def classify_tokens(self, text: str, max_length: int | None = 8192) -> list[tuple[str, str, float]]:
        """Token classification: ``(token, label, probability)`` for every token."""
        self._require_task("token-classification")
        enc = self._encodings([text], max_length=max_length)[0]
        probs = _softmax(self.logits_ids([enc.ids])[0])
        return [(t, self.labels[int(i)], float(p[i])) for t, p, i in zip(enc.tokens, probs, probs.argmax(-1))]

    def answer(self, question: str, context: str, max_answer_tokens: int = 64,
               max_length: int | None = 8192) -> tuple[str, float]:
        """Extractive question answering: the best span of ``context`` and its score
        (start logit + end logit)."""
        self._require_task("question-answering")
        enc = self._encodings([question], [context], max_length)[0]
        logits = self.logits_ids([enc.ids])[0].astype(np.float64)
        # spans inside the context: the second sequence of the pair
        ctx = [i for i, s in enumerate(enc.sequence_ids) if s == 1]
        if not ctx:
            return "", float("-inf")
        best, span = float("-inf"), (ctx[0], ctx[0])
        for a in ctx:
            hi = min(ctx[-1], a + max_answer_tokens - 1)
            b = a + int(np.argmax(logits[a:hi + 1, 1]))
            if logits[a, 0] + logits[b, 1] > best:
                best, span = logits[a, 0] + logits[b, 1], (a, b)
        start, end = enc.offsets[span[0]][0], enc.offsets[span[1]][1]
        return context[start:end], float(best)

    # -- debugging ---------------------------------------------------------
    def _debug_set_layers(self, n_layers: int) -> None:
        """Run only the first ``n_layers`` blocks and skip out_norm (-1: full model)."""
        self._lib.nextera_debug_set_layers(self._ctx, int(n_layers))
