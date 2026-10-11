// nexterabert.cpp -- GGUF inference for NexteraBERT on top of ggml.
//
// Plain C API so it can be loaded from Python (ctypes) or any other FFI.
#ifndef NEXTERABERT_H
#define NEXTERABERT_H

#include <stddef.h>
#include <stdint.h>

#if defined(NEXTERA_SHARED)
#  if defined(_WIN32)
#    if defined(NEXTERA_BUILD)
#      define NEXTERA_API __declspec(dllexport)
#    else
#      define NEXTERA_API __declspec(dllimport)
#    endif
#  else
#    define NEXTERA_API __attribute__((visibility("default")))
#  endif
#else
#  define NEXTERA_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct nextera_context;

struct nextera_params {
    int32_t n_threads;   // <= 0: automatic (about 90% of the logical cores)
    int32_t use_int8;    // 1: run quantized linear layers on the built-in int8 GEMM (x86 AVX2 / VNNI);
                         //    0: always use ggml's own kernels for the file's block format
    int32_t use_repack;  // 1: let ggml repack Q4_0 / Q4_K weights into its interleaved GEMM layout
    int32_t verbosity;   // 0 silent, 1 load summary, 2 also ggml's own log
};

NEXTERA_API struct nextera_params nextera_default_params(void);

// Returns NULL on failure (the reason is printed to stderr).
NEXTERA_API struct nextera_context * nextera_load(const char * path, struct nextera_params params);
NEXTERA_API void nextera_free(struct nextera_context * ctx);

NEXTERA_API int32_t nextera_n_embd(const struct nextera_context * ctx);
NEXTERA_API int32_t nextera_n_vocab(const struct nextera_context * ctx);
NEXTERA_API int32_t nextera_n_layer(const struct nextera_context * ctx);
NEXTERA_API int32_t nextera_has_mlm_head(const struct nextera_context * ctx);
// Heads this file carries besides the encoder. nextera_classifier_task is "" or one of
// "sequence-classification", "multiple-choice", "token-classification", "question-answering".
NEXTERA_API int32_t nextera_has_pooler(const struct nextera_context * ctx);
NEXTERA_API int32_t nextera_n_token_types(const struct nextera_context * ctx);   // 0: no segment embeddings
NEXTERA_API const char * nextera_classifier_task(const struct nextera_context * ctx);
NEXTERA_API int32_t nextera_n_labels(const struct nextera_context * ctx);
// Name of label i (id2label of the checkpoint), "" if the file has none.
NEXTERA_API const char * nextera_label(const struct nextera_context * ctx, int32_t i);
NEXTERA_API int32_t nextera_token_cls(const struct nextera_context * ctx);
NEXTERA_API int32_t nextera_token_sep(const struct nextera_context * ctx);
NEXTERA_API int32_t nextera_token_pad(const struct nextera_context * ctx);
NEXTERA_API int32_t nextera_token_mask(const struct nextera_context * ctx);
// tokenizer.json exactly as shipped with the Hugging Face model ("" if absent).
NEXTERA_API const char * nextera_tokenizer_json(const struct nextera_context * ctx);
// e.g. "Q4_K_M"; taken from general.file_type.
NEXTERA_API const char * nextera_ftype(const struct nextera_context * ctx);
NEXTERA_API const char * nextera_system_info(const struct nextera_context * ctx);

NEXTERA_API void    nextera_set_n_threads(struct nextera_context * ctx, int32_t n_threads);
NEXTERA_API int32_t nextera_get_n_threads(const struct nextera_context * ctx);

// Encode `n_seqs` token sequences in one forward pass.
//
//   tokens   all sequences concatenated (sum(lengths) ids, special tokens included)
//   lengths  n_seqs sequence lengths
//   hidden   out, optional: sum(lengths) * n_embd  -- last_hidden_state, sequences concatenated
//   pooled   out, optional: n_seqs * n_embd        -- mean over each sequence's tokens
//   normalize  1: L2-normalise each pooled vector
//
// No padding is involved: each sequence is processed exactly as if it were run
// alone (convolutions and attention stop at sequence edges). A sequence's result
// does not depend on what else is in the batch: bit-identical for float files
// and on the int8 path, within quantization noise on ggml's quantized kernels.
// Returns 0 on success.
NEXTERA_API int32_t nextera_encode(struct nextera_context * ctx,
                                   const int32_t * tokens, const int32_t * lengths, int32_t n_seqs,
                                   float * hidden, float * pooled, int32_t normalize);

// The calls below take the same batch layout as nextera_encode plus
//   token_types  optional (NULL: none): sum(lengths) segment ids, added through the
//                segment embeddings of a fine-tuned file. Files converted from a
//                checkpoint whose segment embeddings are zero carry none; ids are
//                then ignored, as they have no effect in the original model either.

// nextera_encode with segment ids.
NEXTERA_API int32_t nextera_encode_segments(struct nextera_context * ctx,
                                            const int32_t * tokens, const int32_t * token_types,
                                            const int32_t * lengths, int32_t n_seqs,
                                            float * hidden, float * pooled, int32_t normalize);

// pooler_output: dense -> GELU -> LayerNorm of each sequence's mean hidden state.
//   out  n_seqs * n_embd
NEXTERA_API int32_t nextera_pooler_output(struct nextera_context * ctx,
                                          const int32_t * tokens, const int32_t * token_types,
                                          const int32_t * lengths, int32_t n_seqs, float * out);

// Classifier logits.
//   sequence-classification, multiple-choice    logits: n_seqs * n_labels
//       (multiple-choice has one label: pass every choice as a sequence and
//        compare the scores of the choices of one question)
//   token-classification, question-answering    logits: sum(lengths) * n_labels
//       (question-answering: label 0 is the span start, label 1 the span end)
NEXTERA_API int32_t nextera_classify(struct nextera_context * ctx,
                                     const int32_t * tokens, const int32_t * token_types,
                                     const int32_t * lengths, int32_t n_seqs, float * logits);

// Masked-LM logits for selected positions of ONE sequence.
//   positions  n_pos indices into the sequence
//   logits     out: n_pos * n_vocab
NEXTERA_API int32_t nextera_mlm_logits(struct nextera_context * ctx,
                                       const int32_t * tokens, int32_t n_tokens,
                                       const int32_t * positions, int32_t n_pos,
                                       float * logits);

// Wall time of the last nextera_encode / nextera_mlm_logits call, in milliseconds.
NEXTERA_API double nextera_last_ms(const struct nextera_context * ctx);

#ifdef __cplusplus
}
#endif

#endif // NEXTERABERT_H
