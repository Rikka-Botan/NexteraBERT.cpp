// nexterabert.cpp -- GGUF inference for NexteraBERT on top of ggml.
//
// ggml supplies the GGUF reader, the quantized block formats, the graph
// allocator and the thread pool. Everything specific to this architecture is a
// custom op in this file:
//
//   * row ops    RMSNorm, relu^2, the sigmoid gate, depthwise convolutions, the
//                SnowLily state update, DyT + RoPE, HRA block pooling. Each one
//                produces a whole activation row in a single pass and can write
//                it either as f32 or already quantized for the next linear layer.
//   * attention  grouped-query, optionally banded (swa), per document.
//   * linear     quantized files run their nn.Linear layers on an int8 GEMM
//                (AVX-VNNI / AVX2); float files, and builds without that kernel,
//                use ggml's mul_mat on the file's own block format.
//
// A forward pass takes any number of sequences and runs them *unpadded*: the
// rows of all sequences are concatenated into one (n_embd, N) activation and
// the position-mixing ops stop at sequence boundaries. This is exactly the
// model applied to each sequence alone.

#include "nexterabert.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#define NEXTERA_AVX2 1
#endif

// int8 GEMM (see "int8 linear layers" below). u8 activations x s8 weights:
// one vpdpbusd per 32 products with VNNI; without it, vpmaddubsw would
// saturate on 8-bit activations, so they are quantized to 7 bits instead.
#if defined(NEXTERA_AVX2)
#  define NEXTERA_INT8 1
#  if defined(__AVXVNNI__)
#    define NEXTERA_DPBUSD(acc, x, w) _mm256_dpbusd_avx_epi32(acc, x, w)
#    define NEXTERA_INT8_NAME "AVX-VNNI"
#    define NEXTERA_ACT_MAX 127
#  elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
#    define NEXTERA_DPBUSD(acc, x, w) _mm256_dpbusd_epi32(acc, x, w)
#    define NEXTERA_INT8_NAME "AVX512-VNNI"
#    define NEXTERA_ACT_MAX 127
#  else
#    define NEXTERA_DPBUSD(acc, x, w) \
        _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), _mm256_set1_epi16(1)))
#    define NEXTERA_INT8_NAME "AVX2"
#    define NEXTERA_ACT_MAX 63
#  endif
#  define NEXTERA_ACT_OFF (NEXTERA_ACT_MAX + 1)
#endif

namespace {

// ---------------------------------------------------------------------------
// small SIMD kernels
// ---------------------------------------------------------------------------

#if defined(NEXTERA_AVX2)

// exp(x), Cephes-style: ~1e-7 relative error over the clamped range.
inline __m256 exp256(__m256 x) {
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.3f));
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, z, _mm256_add_ps(x, _mm256_set1_ps(1.0f)));
    const __m256i e = _mm256_slli_epi32(
        _mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

// tanh(x): (1 - e^-2|x|) / (1 + e^-2|x|) with the sign restored, and the odd
// Taylor series near 0 where that quotient would lose relative precision.
inline __m256 tanh256(__m256 x) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    const __m256 ax   = _mm256_andnot_ps(sign, x);
    const __m256 t    = exp256(_mm256_mul_ps(ax, _mm256_set1_ps(-2.0f)));
    const __m256 one  = _mm256_set1_ps(1.0f);
    __m256 big = _mm256_div_ps(_mm256_sub_ps(one, t), _mm256_add_ps(one, t));
    big = _mm256_or_ps(big, _mm256_and_ps(sign, x));
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 p = _mm256_set1_ps(-17.0f / 315.0f);
    p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(2.0f / 15.0f));
    p = _mm256_fmadd_ps(p, x2, _mm256_set1_ps(-1.0f / 3.0f));
    p = _mm256_fmadd_ps(p, x2, one);
    const __m256 small = _mm256_mul_ps(p, x);
    return _mm256_blendv_ps(big, small, _mm256_cmp_ps(ax, _mm256_set1_ps(0.1f), _CMP_LT_OQ));
}

inline __m256 sigmoid256(__m256 x) {
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, exp256(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

#endif // NEXTERA_AVX2

inline float dot_f32(const float * a, const float * b, int n) {
    int i = 0;
    float s = 0.0f;
#if defined(NEXTERA_AVX2)
    __m256 s0 = _mm256_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    for (; i + 8 <= n; i += 8) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    }
    s = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
#endif
    for (; i < n; ++i) s += a[i] * b[i];
    return s;
}

// y[i] += w * x[i]
inline void axpy_f32(float * y, float w, const float * x, int n) {
    int i = 0;
#if defined(NEXTERA_AVX2)
    const __m256 vw = _mm256_set1_ps(w);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(vw, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    }
#endif
    for (; i < n; ++i) y[i] += w * x[i];
}

// y[i] += a[i] * b[i]
inline void addmul_f32(float * y, const float * a, const float * b, int n) {
    int i = 0;
#if defined(NEXTERA_AVX2)
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i),
                                                _mm256_loadu_ps(y + i)));
    }
#endif
    for (; i < n; ++i) y[i] += a[i] * b[i];
}

// In-place softmax weights (un-normalised): s[i] = exp(s[i] - max); returns the sum.
inline float softmax_exp(float * s, int n) {
    float m = s[0];
    for (int i = 1; i < n; ++i) m = std::max(m, s[i]);
    int i = 0;
    float sum = 0.0f;
#if defined(NEXTERA_AVX2)
    const __m256 vm = _mm256_set1_ps(m);
    __m256 vs = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        const __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(s + i), vm));
        _mm256_storeu_ps(s + i, e);
        vs = _mm256_add_ps(vs, e);
    }
    sum = hsum256(vs);
#endif
    for (; i < n; ++i) {
        s[i] = std::exp(s[i] - m);
        sum += s[i];
    }
    return sum;
}

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

enum mixer_type { MIX_LILY, MIX_SWA, MIX_ATTN, MIX_HRA };
enum ssmax_mode { SSMAX_SSSMAX, SSMAX_SSMAX, SSMAX_NONE };

struct hparams {
    int32_t n_vocab = 0, n_embd = 0, n_layer = 0, n_inter = 0;
    int32_t n_head = 0, n_kv_head = 0, head_dim = 0;
    int32_t lily_rank = 0, lily_kernel = 0, hra_block = 0, swa_window = 0;
    int32_t ntk_train_len = 0;
    float   rope_base = 100000.0f, swa_rope_base = 10000.0f;
    float   ssmax_base = 0.1f, rms_eps = 1e-8f, ln_eps = 1e-5f;
    ssmax_mode ssmax = SSMAX_SSSMAX;
    std::vector<mixer_type> block_types;
    int32_t cls_id = -1, sep_id = -1, pad_id = -1, mask_id = -1;
    bool    has_mlm = false;
};

// Row layout of a batch of unpadded sequences ("documents").
struct layout {
    std::vector<int32_t> start, len;   // per document
    std::vector<int32_t> doc;          // per row
    int32_t total = 0;

    void build(const int32_t * lengths, int32_t n) {
        start.resize(n);
        len.assign(lengths, lengths + n);
        total = 0;
        for (int i = 0; i < n; ++i) { start[i] = total; total += len[i]; }
        doc.resize(total);
        for (int i = 0; i < n; ++i) std::fill(doc.begin() + start[i], doc.begin() + start[i] + len[i], i);
    }
};

struct rope_table {
    int32_t n_pos = 0;
    float   base  = 0.0f;
    std::vector<float> cos, sin;       // n_pos * head_dim/2
};

// A linear layer's weight re-laid out for the int8 GEMM: s8 values with one
// float scale per (output row, block of I8_BLOCK inputs).
const int I8_BLOCK = 256;

struct i8_weight {
    int32_t n_in = 0, n_out = 0, n_blk = 0;
    std::vector<int8_t>  w;       // n_out * n_in, row-major
    std::vector<float>   scale;   // n_blk * n_out
    std::vector<int32_t> off;     // n_blk * n_out: -ACT_OFF * sum of the block's weights
};

// Wall time per kind of custom op (measured on thread 0), for --profile.
enum prof_kind { PROF_LINEAR, PROF_ATTN, PROF_ROWS, PROF_COUNT };
const char * const PROF_NAMES[PROF_COUNT] = {
    "int8 linear", "attention", "norm / conv / gate / quantize",
};

struct profiler {
    bool    enabled = false;
    int64_t us[PROF_COUNT] = {};
    int64_t total_us = 0, calls = 0;
};

struct prof_scope {
    profiler * pr;
    int        kind;
    int64_t    t0;
    prof_scope(profiler * p, int k, int ith) : pr(p && p->enabled && ith == 0 ? p : nullptr), kind(k), t0(0) {
        if (pr) t0 = ggml_time_us();
    }
    ~prof_scope() { if (pr) pr->us[kind] += ggml_time_us() - t0; }
};

// Everything a custom op needs besides its input tensors. One per graph node,
// rebuilt with the graph on every forward pass.
struct op_params {
    profiler *     prof = nullptr;
    const layout * lay  = nullptr;     // rows of the op's input
    const layout * blay = nullptr;     // HRA: rows of the block-level tensors
    // Work cursor shared by the op's threads: each takes the next few rows, so
    // fast and slow cores both stay busy (a static split waits for the slowest).
    std::atomic<int32_t> next{0};
    // row ops: write u8 + scales for an int8 linear instead of f32
    bool    quant = false;
    int32_t n_blk = 0;
    // RMSNorm
    const float * gain = nullptr;
    float   eps = 0.0f;
    // depthwise conv
    const float * taps = nullptr;      // k * C, tap-major
    int32_t k = 0;
    // DyT + RoPE
    const float * alpha = nullptr, * beta = nullptr;
    const rope_table * rope = nullptr;
    int32_t head_dim = 0;
    bool    head_major = false;        // write (head, row, dim) instead of (row, head, dim)
    // attention
    int32_t n_head = 0, n_kv_head = 0, half = -1;
    std::vector<float> scale;          // per document
    // HRA
    int32_t block = 0;
    // int8 linear
    const i8_weight * i8 = nullptr;
    int32_t n_rows = 0;
    bool    residual = false;
};

} // namespace

struct nextera_context {
    hparams hp;
    std::string tokenizer_json, ftype, sysinfo;

    ggml_backend_t        backend    = nullptr;
    ggml_threadpool *     threadpool = nullptr;
    int32_t               n_threads  = 1;

    ggml_context *        ctx_plain  = nullptr;   // weights read in place
    ggml_context *        ctx_repack = nullptr;   // weights ggml re-lays out for its GEMM kernels
    ggml_backend_buffer_t buf_plain  = nullptr;
    ggml_backend_buffer_t buf_repack = nullptr;
    int32_t               n_repacked = 0;
    std::unordered_map<std::string, ggml_tensor *> tensors;
    std::unordered_map<std::string, std::vector<float>> taps;   // conv kernels, tap-major
    std::unordered_map<std::string, i8_weight> i8;              // weights on the int8 GEMM path

    ggml_gallocr_t        galloc = nullptr;
    std::vector<uint8_t>  graph_mem;

    // per-forward state the custom ops point into
    layout                lay, blay;
    rope_table            rope[2];                // 0: attn / hra, 1: swa
    std::deque<op_params> ops;

    int32_t               debug_n_layers = -1;    // run only the first n blocks (and skip out_norm)
    profiler              prof;
    double                last_ms = 0.0;

    ggml_tensor * get(const std::string & name) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            fprintf(stderr, "nexterabert: missing tensor '%s'\n", name.c_str());
            abort();
        }
        return it->second;
    }
};

namespace {

// ---------------------------------------------------------------------------
// activation quantization (input side of the int8 linear layers)
// ---------------------------------------------------------------------------

#if defined(NEXTERA_INT8)

// Bytes of a quantized activation: T rows of n_in u8, then the scales laid out
// [token / 4][block][token % 4] so four tokens' scales load as one vector.
inline size_t qact_scale_offset(int T, int n_in) { return (size_t) T * n_in; }
inline size_t qact_bytes(int T, int n_in, int n_blk) {
    return qact_scale_offset(T, n_in) + (size_t) ((T + 3) / 4) * n_blk * 4 * sizeof(float);
}

// One row: each block of I8_BLOCK values is scaled by its own max |x|.
inline void quantize_row(const float * xr, uint8_t * qr, float * sx, int t, int n_in, int n_blk) {
    const __m256  sign = _mm256_set1_ps(-0.0f);
    const __m256i voff = _mm256_set1_epi32(NEXTERA_ACT_OFF);
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (int B = 0; B < n_blk; ++B) {
        const int i0 = B * I8_BLOCK, i1 = std::min(n_in, i0 + I8_BLOCK);
        __m256 vmax = _mm256_setzero_ps();
        for (int i = i0; i < i1; i += 8) {
            vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign, _mm256_loadu_ps(xr + i)));
        }
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_movehdup_ps(m));
        const float amax = _mm_cvtss_f32(m);
        sx[((size_t) (t / 4) * n_blk + B) * 4 + (t % 4)] = amax / (float) NEXTERA_ACT_MAX;
        const __m256 inv = _mm256_set1_ps(amax > 0.0f ? (float) NEXTERA_ACT_MAX / amax : 0.0f);
        for (int i = i0; i < i1; i += 32) {
#define NEXTERA_Q8(o) _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(xr + i + (o)), inv)), voff)
            const __m256i ab = _mm256_packs_epi32(NEXTERA_Q8(0), NEXTERA_Q8(8));
            const __m256i cd = _mm256_packs_epi32(NEXTERA_Q8(16), NEXTERA_Q8(24));
#undef NEXTERA_Q8
            _mm256_storeu_si256((__m256i *) (qr + i),
                _mm256_permutevar8x32_epi32(_mm256_packus_epi16(ab, cd), perm));
        }
    }
}

#endif // NEXTERA_INT8

// ---------------------------------------------------------------------------
// row ops. Activations are (channels, rows): row t starts at data + t * C.
//
// for_rows() runs `fn(t, out)` for every output row, handing it either the
// row's place in the f32 result or a scratch row that is then quantized.
// ---------------------------------------------------------------------------

template <typename F>
inline void for_rows(ggml_tensor * dst, op_params & p, int C, int T, int nth, F && fn) {
    const int grain = std::max(1, std::min(16, T / (4 * std::max(1, nth))));
    float * dd = (float *) dst->data;
#if defined(NEXTERA_INT8)
    static thread_local std::vector<float> scratch;
    uint8_t * qb = (uint8_t *) dst->data;
    float * sx = (float *) (qb + qact_scale_offset(T, C));
    if (p.quant && (int) scratch.size() < C) scratch.resize(C);
#endif
    for (;;) {
        const int t0 = p.next.fetch_add(grain, std::memory_order_relaxed);
        if (t0 >= T) break;
        const int t1 = std::min(T, t0 + grain);
        for (int t = t0; t < t1; ++t) {
#if defined(NEXTERA_INT8)
            if (p.quant) {
                fn(t, scratch.data());
                quantize_row(scratch.data(), qb + (size_t) t * C, sx, t, C, p.n_blk);
                continue;
            }
#endif
            fn(t, dd + (size_t) t * C);
        }
    }
}

inline const float * rowf(const ggml_tensor * x, int t) {
    return (const float *) x->data + (size_t) t * x->ne[0];
}

// The rows [lo, hi) of row t's document.
inline void doc_range(const layout * lay, int t, int & lo, int & hi) {
    const int d = lay->doc[t];
    lo = lay->start[d];
    hi = lo + lay->len[d];
}

// x as is (f32 -> quantized)
void op_copy(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * x = dst->src[0];
    const int C = (int) x->ne[0];
    for_rows(dst, p, C, (int) x->ne[1], nth, [&](int t, float * out) {
        std::memcpy(out, rowf(x, t), sizeof(float) * C);
    });
}

// RMSNorm with a per-channel gain: x * rsqrt(mean(x^2) + eps) * g
void op_rms_norm(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * x = dst->src[0];
    const int C = (int) x->ne[0];
    for_rows(dst, p, C, (int) x->ne[1], nth, [&](int t, float * out) {
        const float * xr = rowf(x, t);
        const float inv = 1.0f / std::sqrt(dot_f32(xr, xr, C) / (float) C + p.eps);
        int i = 0;
#if defined(NEXTERA_AVX2)
        const __m256 vi = _mm256_set1_ps(inv);
        for (; i + 8 <= C; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i), vi),
                                                    _mm256_loadu_ps(p.gain + i)));
        }
#endif
        for (; i < C; ++i) out[i] = xr[i] * inv * p.gain[i];
    });
}

// relu(x)^2
void op_relu_sqr(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * x = dst->src[0];
    const int C = (int) x->ne[0];
    for_rows(dst, p, C, (int) x->ne[1], nth, [&](int t, float * out) {
        const float * xr = rowf(x, t);
        int i = 0;
#if defined(NEXTERA_AVX2)
        const __m256 zero = _mm256_setzero_ps();
        for (; i + 8 <= C; i += 8) {
            const __m256 r = _mm256_max_ps(_mm256_loadu_ps(xr + i), zero);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(r, r));
        }
#endif
        for (; i < C; ++i) {
            const float r = xr[i] > 0.0f ? xr[i] : 0.0f;
            out[i] = r * r;
        }
    });
}

// sigmoid(g) * y
void op_gate(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * g = dst->src[0], * y = dst->src[1];
    const int C = (int) g->ne[0];
    for_rows(dst, p, C, (int) g->ne[1], nth, [&](int t, float * out) {
        const float * gr = rowf(g, t), * yr = rowf(y, t);
        int i = 0;
#if defined(NEXTERA_AVX2)
        for (; i + 8 <= C; i += 8) {
            _mm256_storeu_ps(out + i, _mm256_mul_ps(sigmoid256(_mm256_loadu_ps(gr + i)), _mm256_loadu_ps(yr + i)));
        }
#endif
        for (; i < C; ++i) out[i] = yr[i] / (1.0f + std::exp(-gr[i]));
    });
}

// Depthwise 1-D convolution over the rows of each document (zero padding at
// document edges), plus an optional second input:
//   out[t, c] = sum_j w[c, j] * x[t + j - k/2, c]  (+ r[t, c])
void op_dwconv(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * x = dst->src[0], * r = dst->src[1];
    const int C = (int) x->ne[0];
    for_rows(dst, p, C, (int) x->ne[1], nth, [&](int t, float * out) {
        int lo, hi;
        doc_range(p.lay, t, lo, hi);
        if (r) std::memcpy(out, rowf(r, t), sizeof(float) * C);
        else   std::memset(out, 0, sizeof(float) * C);
        for (int j = 0; j < p.k; ++j) {
            const int tt = t + j - p.k / 2;
            if (tt < lo || tt >= hi) continue;
            addmul_f32(out, p.taps + (size_t) j * C, rowf(x, tt), C);
        }
    });
}

// SnowLily state update: the depthwise convolution of the input-dependent
// product a * x, gated by b:   out[t] = b[t] * sum_j w[j] * a[t + j - k/2] * x[t + j - k/2]
void op_lily_state(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * a = dst->src[0], * x = dst->src[1], * b = dst->src[2];
    const int C = (int) x->ne[0];
    for_rows(dst, p, C, (int) x->ne[1], nth, [&](int t, float * out) {
        int lo, hi;
        doc_range(p.lay, t, lo, hi);
        const int j0 = std::max(0, lo - (t - p.k / 2)), j1 = std::min(p.k, hi - (t - p.k / 2));
        const float * br = rowf(b, t);
        int i = 0;
#if defined(NEXTERA_AVX2)
        for (; i + 8 <= C; i += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int j = j0; j < j1; ++j) {
                const int tt = t + j - p.k / 2;
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(p.taps + (size_t) j * C + i),
                    _mm256_mul_ps(_mm256_loadu_ps(rowf(a, tt) + i), _mm256_loadu_ps(rowf(x, tt) + i)), acc);
            }
            _mm256_storeu_ps(out + i, _mm256_mul_ps(acc, _mm256_loadu_ps(br + i)));
        }
#endif
        for (; i < C; ++i) {
            float acc = 0.0f;
            for (int j = j0; j < j1; ++j) {
                const int tt = t + j - p.k / 2;
                acc += p.taps[(size_t) j * C + i] * rowf(a, tt)[i] * rowf(x, tt)[i];
            }
            out[i] = acc * br[i];
        }
    });
}

// Attention reads one head's keys / values for a run of consecutive positions.
// Stored row-major they are a whole row apart; "head-major" (head, row, dim)
// makes that run contiguous. Element (t, h) of a head-major (D * H, T) tensor:
inline float * head_major_at(float * base, int t, int h, int T, int D) {
    return base + ((size_t) h * T + t) * D;
}

// Separable DyT (beta * tanh(alpha * x), per head channel) followed by RoPE,
// on a (head_dim * n_head, rows) projection.
void op_qk_prep(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * x = dst->src[0];
    const int CH = (int) x->ne[0], T = (int) x->ne[1];
    const int D = p.head_dim, half = D / 2, H = CH / D;
    for_rows(dst, p, CH, T, nth, [&](int t, float * out) {
        const int pos = t - p.lay->start[p.lay->doc[t]];
        const float * c = p.rope->cos.data() + (size_t) pos * half;
        const float * s = p.rope->sin.data() + (size_t) pos * half;
        const float * xr = rowf(x, t);
        for (int h = 0; h < H; ++h) {
            const float * x1 = xr + (size_t) h * D, * x2 = x1 + half;
            float * y1 = p.head_major ? head_major_at((float *) dst->data, t, h, T, D) : out + (size_t) h * D;
            float * y2 = y1 + half;
            int i = 0;
#if defined(NEXTERA_AVX2)
            for (; i + 8 <= half; i += 8) {
                const __m256 a = _mm256_mul_ps(_mm256_loadu_ps(p.beta + i),
                    tanh256(_mm256_mul_ps(_mm256_loadu_ps(p.alpha + i), _mm256_loadu_ps(x1 + i))));
                const __m256 b = _mm256_mul_ps(_mm256_loadu_ps(p.beta + half + i),
                    tanh256(_mm256_mul_ps(_mm256_loadu_ps(p.alpha + half + i), _mm256_loadu_ps(x2 + i))));
                const __m256 vc = _mm256_loadu_ps(c + i), vs = _mm256_loadu_ps(s + i);
                _mm256_storeu_ps(y1 + i, _mm256_fmadd_ps(a, vc, _mm256_mul_ps(b, vs)));
                _mm256_storeu_ps(y2 + i, _mm256_fnmadd_ps(a, vs, _mm256_mul_ps(b, vc)));
            }
#endif
            for (; i < half; ++i) {
                const float a = p.beta[i] * std::tanh(p.alpha[i] * x1[i]);
                const float b = p.beta[half + i] * std::tanh(p.alpha[half + i] * x2[i]);
                y1[i] = a * c[i] + b * s[i];
                y2[i] = b * c[i] - a * s[i];
            }
        }
    });
}

// (D * H, rows) row-major -> head-major
void op_head_major(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * x = dst->src[0];
    const int CH = (int) x->ne[0], T = (int) x->ne[1], D = p.head_dim, H = CH / D;
    for_rows(dst, p, CH, T, nth, [&](int t, float *) {
        const float * xr = rowf(x, t);
        for (int h = 0; h < H; ++h) {
            std::memcpy(head_major_at((float *) dst->data, t, h, T, D), xr + (size_t) h * D, sizeof(float) * D);
        }
    });
}

// HRA: mean over each block's (real) tokens of h (* g). (C, rows) -> (C, blocks)
void op_block_mean(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * h = dst->src[0], * g = dst->src[1];
    const int C = (int) h->ne[0];
    for_rows(dst, p, C, p.blay->total, nth, [&](int b, float * out) {
        const int d  = p.blay->doc[b];
        const int bi = b - p.blay->start[d];
        const int ts = p.lay->start[d] + bi * p.block;
        const int n  = std::min(p.block, p.lay->len[d] - bi * p.block);
        std::memset(out, 0, sizeof(float) * C);
        for (int i = 0; i < n; ++i) {
            if (g) addmul_f32(out, rowf(h, ts + i), rowf(g, ts + i), C);
            else   axpy_f32(out, 1.0f, rowf(h, ts + i), C);
        }
        const float inv = 1.0f / (float) n;
        for (int c = 0; c < C; ++c) out[c] *= inv;
    });
}

// HRA: broadcast each block's output to its tokens and add the token-wise path.
// y: (C, blocks), r: (C, rows) -> (C, rows)
void op_upsample_add(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ROWS, ith);
    const ggml_tensor * y = dst->src[0], * r = dst->src[1];
    const int C = (int) r->ne[0];
    for_rows(dst, p, C, (int) r->ne[1], nth, [&](int t, float * out) {
        const int d = p.lay->doc[t];
        const float * yb = rowf(y, p.blay->start[d] + (t - p.lay->start[d]) / p.block);
        const float * rr = rowf(r, t);
        for (int c = 0; c < C; ++c) out[c] = yb[c] + rr[c];
    });
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

#if defined(NEXTERA_AVX2)

// head_dim 64, one (query, head). Keys / values of consecutive positions are
// `stride` floats apart.
//
// Scores of keys [j0, j1): the query sits in 8 registers and 8 keys are scored
// at a time. j0 must be a multiple of 8, so that splitting a key range into
// blocks does not change which keys share a group (results stay bit-identical).
inline void scores_d64(const float * q, const float * k, size_t stride, int j0, int j1, float scale, float * sc) {
    const __m256 q0 = _mm256_loadu_ps(q),      q1 = _mm256_loadu_ps(q + 8);
    const __m256 q2 = _mm256_loadu_ps(q + 16), q3 = _mm256_loadu_ps(q + 24);
    const __m256 q4 = _mm256_loadu_ps(q + 32), q5 = _mm256_loadu_ps(q + 40);
    const __m256 q6 = _mm256_loadu_ps(q + 48), q7 = _mm256_loadu_ps(q + 56);
    const __m256 vscale = _mm256_set1_ps(scale);
#define NEXTERA_DOT64(kp) \
    _mm256_fmadd_ps(q7, _mm256_loadu_ps((kp) + 56), _mm256_fmadd_ps(q6, _mm256_loadu_ps((kp) + 48), \
    _mm256_fmadd_ps(q5, _mm256_loadu_ps((kp) + 40), _mm256_fmadd_ps(q4, _mm256_loadu_ps((kp) + 32), \
    _mm256_fmadd_ps(q3, _mm256_loadu_ps((kp) + 24), _mm256_fmadd_ps(q2, _mm256_loadu_ps((kp) + 16), \
    _mm256_fmadd_ps(q1, _mm256_loadu_ps((kp) + 8),  _mm256_mul_ps(q0, _mm256_loadu_ps(kp)))))))))
    int j = j0;
    for (; j + 8 <= j1; j += 8) {
        const float * kp = k + (size_t) j * stride;
        const __m256 a0 = NEXTERA_DOT64(kp),              a1 = NEXTERA_DOT64(kp + stride);
        const __m256 a2 = NEXTERA_DOT64(kp + 2 * stride), a3 = NEXTERA_DOT64(kp + 3 * stride);
        const __m256 a4 = NEXTERA_DOT64(kp + 4 * stride), a5 = NEXTERA_DOT64(kp + 5 * stride);
        const __m256 a6 = NEXTERA_DOT64(kp + 6 * stride), a7 = NEXTERA_DOT64(kp + 7 * stride);
        // eight 8-lane sums -> one vector of eight totals
        const __m256 h0 = _mm256_hadd_ps(_mm256_hadd_ps(a0, a1), _mm256_hadd_ps(a2, a3));
        const __m256 h1 = _mm256_hadd_ps(_mm256_hadd_ps(a4, a5), _mm256_hadd_ps(a6, a7));
        const __m256 t = _mm256_add_ps(_mm256_permute2f128_ps(h0, h1, 0x20), _mm256_permute2f128_ps(h0, h1, 0x31));
        _mm256_storeu_ps(sc + j, _mm256_mul_ps(t, vscale));
    }
    for (; j < j1; ++j) sc[j] = scale * hsum256(NEXTERA_DOT64(k + (size_t) j * stride));
#undef NEXTERA_DOT64
}

// acc (8 registers' worth, 64 floats) += sum over keys [j0, j1) of w[j] * inv * v[j]
inline void values_d64(const float * w, float inv, const float * v, size_t stride, int j0, int j1, float * acc) {
    __m256 o0 = _mm256_loadu_ps(acc),      o1 = _mm256_loadu_ps(acc + 8);
    __m256 o2 = _mm256_loadu_ps(acc + 16), o3 = _mm256_loadu_ps(acc + 24);
    __m256 o4 = _mm256_loadu_ps(acc + 32), o5 = _mm256_loadu_ps(acc + 40);
    __m256 o6 = _mm256_loadu_ps(acc + 48), o7 = _mm256_loadu_ps(acc + 56);
    for (int j = j0; j < j1; ++j) {
        const float * vp = v + (size_t) j * stride;
        const __m256 c = _mm256_set1_ps(w[j] * inv);
        o0 = _mm256_fmadd_ps(c, _mm256_loadu_ps(vp),      o0);
        o1 = _mm256_fmadd_ps(c, _mm256_loadu_ps(vp + 8),  o1);
        o2 = _mm256_fmadd_ps(c, _mm256_loadu_ps(vp + 16), o2);
        o3 = _mm256_fmadd_ps(c, _mm256_loadu_ps(vp + 24), o3);
        o4 = _mm256_fmadd_ps(c, _mm256_loadu_ps(vp + 32), o4);
        o5 = _mm256_fmadd_ps(c, _mm256_loadu_ps(vp + 40), o5);
        o6 = _mm256_fmadd_ps(c, _mm256_loadu_ps(vp + 48), o6);
        o7 = _mm256_fmadd_ps(c, _mm256_loadu_ps(vp + 56), o7);
    }
    _mm256_storeu_ps(acc,      o0); _mm256_storeu_ps(acc + 8,  o1);
    _mm256_storeu_ps(acc + 16, o2); _mm256_storeu_ps(acc + 24, o3);
    _mm256_storeu_ps(acc + 32, o4); _mm256_storeu_ps(acc + 40, o5);
    _mm256_storeu_ps(acc + 48, o6); _mm256_storeu_ps(acc + 56, o7);
}

#endif // NEXTERA_AVX2

// One (query, head): scores against n keys, softmax, weighted sum of the values.
inline void attend(const float * q, const float * k, const float * v, size_t stride, int n, int D,
                   float scale, float * sc, float * out) {
#if defined(NEXTERA_AVX2)
    if (D == 64) {
        scores_d64(q, k, stride, 0, n, scale, sc);
        const float inv = 1.0f / softmax_exp(sc, n);
        std::memset(out, 0, sizeof(float) * 64);
        values_d64(sc, inv, v, stride, 0, n, out);
        return;
    }
#endif
    for (int j = 0; j < n; ++j) sc[j] = scale * dot_f32(q, k + (size_t) j * stride, D);
    const float inv = 1.0f / softmax_exp(sc, n);
    std::memset(out, 0, sizeof(float) * D);
    for (int j = 0; j < n; ++j) axpy_f32(out, sc[j] * inv, v + (size_t) j * stride, D);
}

// Past this many keys, one head's keys no longer fit in cache and attending one
// query at a time would stream them from memory once per query. Queries that
// share a key range are then processed together, a block of keys at a time.
const int ATTN_BLOCK_KEYS = 1024;       // multiple of 8 (see scores_d64)
const int ATTN_BLOCK_MIN  = 2048;

// Grouped-query attention inside each document, optionally restricted to the
// band |i - j| <= half. q: (D * n_head, rows) row-major; k / v: (D * n_kv_head,
// rows) head-major.
void op_attention(ggml_tensor * dst, int ith, int nth, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_ATTN, ith);
    const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2];
    const int D = p.head_dim, H = p.n_head, G = p.n_kv_head, rep = H / G;
    const int CQ = D * H, T = (int) q->ne[1];
    const float * kd = (const float *) k->data;
    const float * vd = (const float *) v->data;
    float * dd = (float *) dst->data;
    static thread_local std::vector<float> scores, inv;
    const int max_grain = p.half < 0 && T >= ATTN_BLOCK_MIN ? 32 : 8;
    const int grain = std::max(1, std::min(max_grain, T / (4 * std::max(1, nth))));
    for (;;) {
        const int t0 = p.next.fetch_add(grain, std::memory_order_relaxed);
        if (t0 >= T) break;
        const int t1 = std::min(T, t0 + grain);
        // head outermost: neighbouring queries read the same stretch of one
        // head's keys and values, which then stays in cache
        for (int h = 0; h < H; ++h) {
            const size_t go = (size_t) (h / rep) * T * D;
            for (int t = t0; t < t1;) {
                int lo, hi;
                doc_range(p.lay, t, lo, hi);
                const float scale = p.scale[p.lay->doc[t]];
#if defined(NEXTERA_AVX2)
                if (D == 64 && p.half < 0 && hi - lo >= ATTN_BLOCK_MIN) {
                    // every query of this document in the chunk sees keys [lo, hi)
                    const int te = std::min(t1, hi), nq = te - t, n = hi - lo;
                    const size_t row = (size_t) n + 8;
                    if (scores.size() < row * nq) scores.resize(row * nq);
                    if ((int) inv.size() < nq) inv.resize(nq);
                    const float * kh = kd + go + (size_t) lo * D, * vh = vd + go + (size_t) lo * D;
                    for (int j0 = 0; j0 < n; j0 += ATTN_BLOCK_KEYS) {
                        const int j1 = std::min(n, j0 + ATTN_BLOCK_KEYS);
                        for (int i = 0; i < nq; ++i) {
                            scores_d64(rowf(q, t + i) + (size_t) h * D, kh, (size_t) D, j0, j1, scale,
                                       scores.data() + row * i);
                        }
                    }
                    for (int i = 0; i < nq; ++i) {
                        inv[i] = 1.0f / softmax_exp(scores.data() + row * i, n);
                        std::memset(dd + (size_t) (t + i) * CQ + (size_t) h * D, 0, sizeof(float) * 64);
                    }
                    for (int j0 = 0; j0 < n; j0 += ATTN_BLOCK_KEYS) {
                        const int j1 = std::min(n, j0 + ATTN_BLOCK_KEYS);
                        for (int i = 0; i < nq; ++i) {
                            values_d64(scores.data() + row * i, inv[i], vh, (size_t) D, j0, j1,
                                       dd + (size_t) (t + i) * CQ + (size_t) h * D);
                        }
                    }
                    t = te;
                    continue;
                }
#endif
                if (p.half >= 0) {
                    lo = std::max(lo, t - p.half);
                    hi = std::min(hi, t + p.half + 1);
                }
                const int n = hi - lo;
                if ((int) scores.size() < n + 8) scores.resize((size_t) n + 8);
                attend(rowf(q, t) + (size_t) h * D, kd + go + (size_t) lo * D, vd + go + (size_t) lo * D,
                       (size_t) D, n, D, scale, scores.data(),
                       dd + (size_t) t * CQ + (size_t) h * D);
                ++t;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// int8 linear layers
//
// ggml evaluates a quantized mul_mat by unpacking the 4-bit weights inside the
// inner loop. Here the weights of a quantized file are instead expanded once,
// at load time, to s8 with a float scale per (row, 256 inputs), the activations
// are quantized to u8 with a scale per (token, 256 inputs), and the product is
//
//   y[t, r] = sum_B  sw[B, r] * sx[t, B] * ( sum_i w8[r, i] * x8[t, i]  -  OFF * sum_i w8[r, i] )
//
// whose inner sum is pure integer work: 12 vpdpbusd (3 rows x 4 tokens) per 32
// inputs. The 8-bit grid is 8-16x finer than a 4-bit block's, so re-gridding a
// Q4 weight costs ~1% of the error it already has (see README, "Accuracy").
// ---------------------------------------------------------------------------

#if defined(NEXTERA_INT8)

// Sums the eight lanes of four accumulators: lane k of the result = sum(a_k).
inline __m128i reduce4_epi32(__m256i a0, __m256i a1, __m256i a2, __m256i a3) {
    const __m256i s = _mm256_hadd_epi32(_mm256_hadd_epi32(a0, a1), _mm256_hadd_epi32(a2, a3));
    return _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
}

#define NEXTERA_W(p, i) _mm256_loadu_si256((const __m256i *) ((p) + (i)))

// 3 weight rows (r, r+1, r+2) x 4 tokens (t .. t+3). `out` / `res` point at
// token t's output row / residual row (res may be NULL).
inline void i8_kernel_3x4(const i8_weight & w, const uint8_t * xq, const float * sxq,
                          int r, float * out, const float * res) {
    const int n_in = w.n_in, n_out = w.n_out;
    const int8_t * w0 = w.w.data() + (size_t) r * n_in, * w1 = w0 + n_in, * w2 = w1 + n_in;
    const uint8_t * x0 = xq, * x1 = x0 + n_in, * x2 = x1 + n_in, * x3 = x2 + n_in;
    __m128 f0 = _mm_setzero_ps(), f1 = f0, f2 = f0;
    for (int B = 0; B < w.n_blk; ++B) {
        const int i0 = B * I8_BLOCK, i1 = std::min(n_in, i0 + I8_BLOCK);
        __m256i a00 = _mm256_setzero_si256(), a01 = a00, a02 = a00, a03 = a00;
        __m256i a10 = a00, a11 = a00, a12 = a00, a13 = a00;
        __m256i a20 = a00, a21 = a00, a22 = a00, a23 = a00;
        for (int i = i0; i < i1; i += 32) {
            const __m256i v0 = NEXTERA_W(x0, i), v1 = NEXTERA_W(x1, i);
            const __m256i v2 = NEXTERA_W(x2, i), v3 = NEXTERA_W(x3, i);
            a00 = NEXTERA_DPBUSD(a00, v0, NEXTERA_W(w0, i));
            a01 = NEXTERA_DPBUSD(a01, v1, NEXTERA_W(w0, i));
            a02 = NEXTERA_DPBUSD(a02, v2, NEXTERA_W(w0, i));
            a03 = NEXTERA_DPBUSD(a03, v3, NEXTERA_W(w0, i));
            a10 = NEXTERA_DPBUSD(a10, v0, NEXTERA_W(w1, i));
            a11 = NEXTERA_DPBUSD(a11, v1, NEXTERA_W(w1, i));
            a12 = NEXTERA_DPBUSD(a12, v2, NEXTERA_W(w1, i));
            a13 = NEXTERA_DPBUSD(a13, v3, NEXTERA_W(w1, i));
            a20 = NEXTERA_DPBUSD(a20, v0, NEXTERA_W(w2, i));
            a21 = NEXTERA_DPBUSD(a21, v1, NEXTERA_W(w2, i));
            a22 = NEXTERA_DPBUSD(a22, v2, NEXTERA_W(w2, i));
            a23 = NEXTERA_DPBUSD(a23, v3, NEXTERA_W(w2, i));
        }
        const __m128    sx = _mm_loadu_ps(sxq + (size_t) B * 4);
        const float *   sw = w.scale.data() + (size_t) B * n_out + r;
        const int32_t * of = w.off.data() + (size_t) B * n_out + r;
        __m128i s;
        s  = _mm_add_epi32(reduce4_epi32(a00, a01, a02, a03), _mm_set1_epi32(of[0]));
        f0 = _mm_fmadd_ps(_mm_mul_ps(_mm_cvtepi32_ps(s), sx), _mm_set1_ps(sw[0]), f0);
        s  = _mm_add_epi32(reduce4_epi32(a10, a11, a12, a13), _mm_set1_epi32(of[1]));
        f1 = _mm_fmadd_ps(_mm_mul_ps(_mm_cvtepi32_ps(s), sx), _mm_set1_ps(sw[1]), f1);
        s  = _mm_add_epi32(reduce4_epi32(a20, a21, a22, a23), _mm_set1_epi32(of[2]));
        f2 = _mm_fmadd_ps(_mm_mul_ps(_mm_cvtepi32_ps(s), sx), _mm_set1_ps(sw[2]), f2);
    }
    // rows (f0, f1, f2) x tokens (lanes) -> three values per token
    float y[3][4];
    _mm_storeu_ps(y[0], f0);
    _mm_storeu_ps(y[1], f1);
    _mm_storeu_ps(y[2], f2);
    for (int k = 0; k < 4; ++k) {
        float * o = out + (size_t) k * n_out + r;
        if (res) {
            const float * rr = res + (size_t) k * n_out + r;
            o[0] = rr[0] + y[0][k];
            o[1] = rr[1] + y[1][k];
            o[2] = rr[2] + y[2][k];
        } else {
            o[0] = y[0][k];
            o[1] = y[1][k];
            o[2] = y[2][k];
        }
    }
}

// 4 weight rows (r .. r+3) x 1 token: the edges the 3x4 kernel leaves over.
inline void i8_kernel_4x1(const i8_weight & w, const uint8_t * x, const float * sxq, int k,
                          int r, float * out, const float * res) {
    const int n_in = w.n_in, n_out = w.n_out;
    const int8_t * w0 = w.w.data() + (size_t) r * n_in, * w1 = w0 + n_in, * w2 = w1 + n_in, * w3 = w2 + n_in;
    __m128 f = _mm_setzero_ps();
    for (int B = 0; B < w.n_blk; ++B) {
        const int i0 = B * I8_BLOCK, i1 = std::min(n_in, i0 + I8_BLOCK);
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        for (int i = i0; i < i1; i += 32) {
            const __m256i v = NEXTERA_W(x, i);
            a0 = NEXTERA_DPBUSD(a0, v, NEXTERA_W(w0, i));
            a1 = NEXTERA_DPBUSD(a1, v, NEXTERA_W(w1, i));
            a2 = NEXTERA_DPBUSD(a2, v, NEXTERA_W(w2, i));
            a3 = NEXTERA_DPBUSD(a3, v, NEXTERA_W(w3, i));
        }
        const __m128i s = _mm_add_epi32(reduce4_epi32(a0, a1, a2, a3),
            _mm_loadu_si128((const __m128i *) (w.off.data() + (size_t) B * n_out + r)));
        // same operation order as the 3x4 kernel, so a token's result does not
        // depend on which kernel its position in the batch selects
        f = _mm_fmadd_ps(_mm_mul_ps(_mm_cvtepi32_ps(s), _mm_set1_ps(sxq[(size_t) B * 4 + k])),
                         _mm_loadu_ps(w.scale.data() + (size_t) B * n_out + r), f);
    }
    if (res) f = _mm_add_ps(_mm_loadu_ps(res + r), f);
    _mm_storeu_ps(out + r, f);
}

#undef NEXTERA_W

// (quantized activations [, residual]) -> f32 (n_out, T)
void op_linear_i8(ggml_tensor * dst, int ith, int, void * ud) {
    op_params & p = *(op_params *) ud;
    prof_scope ps(p.prof, PROF_LINEAR, ith);
    const i8_weight & w = *p.i8;
    const int n_in = w.n_in, n_out = w.n_out, n_blk = w.n_blk, T = p.n_rows;
    const uint8_t * qb = (const uint8_t *) dst->src[0]->data;
    const float * sx = (const float *) (qb + qact_scale_offset(T, n_in));
    const float * res = p.residual ? (const float *) dst->src[1]->data : nullptr;
    float * out = (float *) dst->data;

    // Work items = (token block, row chunk). A token block's activations stay in
    // L1 while the weight rows stream past. Items are kept small (a few
    // microseconds): the op ends when its slowest thread finishes its last item.
    const int tok_blk = std::max(4, std::min(64, (32 * 1024 / n_in) & ~3));
    const int row_chk = 12;                                 // multiple of 12
    const int n_tb = (T + tok_blk - 1) / tok_blk, n_rc = (n_out + row_chk - 1) / row_chk;
    const int main_rows = n_out - n_out % 12;               // rows the 3-row kernel covers
    for (;;) {
        const int item = p.next.fetch_add(1, std::memory_order_relaxed);
        if (item >= n_tb * n_rc) break;
        const int t0 = (item / n_rc) * tok_blk, t1 = std::min(T, t0 + tok_blk);
        const int r0 = (item % n_rc) * row_chk, r1 = std::min(n_out, r0 + row_chk);
        const int rm = std::min(r1, main_rows), tq = t0 + ((t1 - t0) & ~3);
        for (int r = r0; r < rm; r += 3) {
            for (int t = t0; t < tq; t += 4) {
                i8_kernel_3x4(w, qb + (size_t) t * n_in, sx + (size_t) (t / 4) * n_blk * 4, r,
                              out + (size_t) t * n_out, res ? res + (size_t) t * n_out : nullptr);
            }
        }
        for (int t = t0; t < t1; ++t) {
            // leftover tokens take every row; leftover rows take every token
            const int rs = t < tq ? rm : r0;
            for (int r = rs; r < r1; r += 4) {
                i8_kernel_4x1(w, qb + (size_t) t * n_in, sx + (size_t) (t / 4) * n_blk * 4, t % 4, r,
                              out + (size_t) t * n_out, res ? res + (size_t) t * n_out : nullptr);
            }
        }
    }
}

// Expands one weight matrix (any ggml type) to the int8 layout.
void build_i8_weight(i8_weight & w, ggml_type type, const void * data, int n_in, int n_out, int n_threads) {
    w.n_in  = n_in;
    w.n_out = n_out;
    w.n_blk = (n_in + I8_BLOCK - 1) / I8_BLOCK;
    w.w.resize((size_t) n_in * n_out);
    w.scale.resize((size_t) w.n_blk * n_out);
    w.off.resize((size_t) w.n_blk * n_out);
    const size_t row_bytes = ggml_row_size(type, n_in);
    const ggml_type_traits * tt = ggml_get_type_traits(type);
    std::atomic<int> next(0);
    auto work = [&]() {
        std::vector<float> row(n_in);
        for (;;) {
            const int r = next.fetch_add(1);
            if (r >= n_out) break;
            const void * src = (const uint8_t *) data + (size_t) r * row_bytes;
            if (type == GGML_TYPE_F32)      std::memcpy(row.data(), src, sizeof(float) * n_in);
            else if (type == GGML_TYPE_F16) ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, row.data(), n_in);
            else                            tt->to_float(src, row.data(), n_in);
            int8_t * q = w.w.data() + (size_t) r * n_in;
            for (int B = 0; B < w.n_blk; ++B) {
                const int i0 = B * I8_BLOCK, i1 = std::min(n_in, i0 + I8_BLOCK);
                float amax = 0.0f;
                for (int i = i0; i < i1; ++i) amax = std::max(amax, std::fabs(row[i]));
                const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
                int32_t sum = 0;
                for (int i = i0; i < i1; ++i) {
                    q[i] = (int8_t) std::lround(row[i] * inv);
                    sum += q[i];
                }
                w.scale[(size_t) B * n_out + r] = amax / 127.0f;
                w.off[(size_t) B * n_out + r]   = -NEXTERA_ACT_OFF * sum;
            }
        }
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < n_threads; ++i) pool.emplace_back(work);
    work();
    for (std::thread & th : pool) th.join();
}

#endif // NEXTERA_INT8

// ---------------------------------------------------------------------------
// graph construction
// ---------------------------------------------------------------------------

// An activation as a linear layer can consume it: f32, quantized, or both.
struct act {
    ggml_tensor * f = nullptr;   // (C, T) f32
    ggml_tensor * q = nullptr;   // quantized (see qact_bytes)
    int64_t C = 0, T = 0;
};

struct builder {
    nextera_context & c;
    ggml_context *    g;
    // quantized copies of f32 activations, shared by the int8 linears reading them
    mutable std::unordered_map<ggml_tensor *, ggml_tensor *> qcache;

    ggml_tensor * w(const std::string & name) const { return c.get(name); }

    op_params * params() const {
        c.ops.emplace_back();
        c.ops.back().prof = &c.prof;
        return &c.ops.back();
    }

    // True if every one of these weights runs on the int8 GEMM, i.e. their
    // shared input is only ever needed quantized.
    bool int8_only(std::initializer_list<std::string> names) const {
        for (const std::string & n : names) {
            if (!c.i8.count(n)) return false;
        }
        return true;
    }

    // A row op node producing (C, T), as f32 or quantized.
    act row_op(ggml_custom_op_t fun, std::initializer_list<ggml_tensor *> args, int64_t C, int64_t T,
               op_params * p, bool quant) const {
        std::vector<ggml_tensor *> a;
        for (ggml_tensor * t : args) {
            if (t) a.push_back(t);
        }
        act r;
        r.C = C;
        r.T = T;
#if defined(NEXTERA_INT8)
        if (quant) {
            p->quant = true;
            p->n_blk = (int32_t) ((C + I8_BLOCK - 1) / I8_BLOCK);
            const int64_t n = (int64_t) ((qact_bytes((int) T, (int) C, p->n_blk) + 3) / 4);
            r.q = ggml_custom_4d(g, GGML_TYPE_F32, n, 1, 1, 1, a.data(), (int) a.size(), fun, GGML_N_TASKS_MAX, p);
            return r;
        }
#else
        (void) quant;
#endif
        r.f = ggml_custom_4d(g, GGML_TYPE_F32, C, T, 1, 1, a.data(), (int) a.size(), fun, GGML_N_TASKS_MAX, p);
        return r;
    }

    act of(ggml_tensor * f) const {
        act r;
        r.f = f;
        r.C = f->ne[0];
        r.T = f->ne[1];
        return r;
    }

    // y = W x (+ residual)
    ggml_tensor * linear(const std::string & name, const act & x, ggml_tensor * residual = nullptr) const {
#if defined(NEXTERA_INT8)
        auto it = c.i8.find(name);
        if (it != c.i8.end()) {
            ggml_tensor * q = x.q;
            if (!q) {
                ggml_tensor *& cached = qcache[x.f];
                if (!cached) cached = row_op(op_copy, {x.f}, x.C, x.T, params(), true).q;
                q = cached;
            }
            op_params * p = params();
            p->i8       = &it->second;
            p->n_rows   = (int32_t) x.T;
            p->residual = residual != nullptr;
            ggml_tensor * a[2] = { q, residual };
            return ggml_custom_4d(g, GGML_TYPE_F32, it->second.n_out, x.T, 1, 1, a, residual ? 2 : 1,
                                  op_linear_i8, GGML_N_TASKS_MAX, p);
        }
#endif
        GGML_ASSERT(x.f != nullptr);
        ggml_tensor * y = ggml_mul_mat(g, w(name), x.f);
        return residual ? ggml_add(g, residual, y) : y;
    }

    act rms_norm(const std::string & name, ggml_tensor * x, bool quant) const {
        op_params * p = params();
        p->gain = (const float *) w(name)->data;
        p->eps  = c.hp.rms_eps;
        return row_op(op_rms_norm, {x}, x->ne[0], x->ne[1], p, quant);
    }

    op_params * conv_params(const std::string & name, int64_t C) const {
        op_params * p = params();
        const std::vector<float> & taps = c.taps.at(name);
        p->lay  = &c.lay;
        p->taps = taps.data();
        p->k    = (int32_t) (taps.size() / (size_t) C);
        return p;
    }

    ggml_tensor * qk_prep(const std::string & norm, ggml_tensor * x, const layout * lay,
                          const rope_table * rope, bool head_major) const {
        op_params * p = params();
        p->head_major = head_major;
        p->lay      = lay;
        p->alpha    = (const float *) w(norm + ".alpha")->data;
        p->beta     = (const float *) w(norm + ".beta")->data;
        p->rope     = rope;
        p->head_dim = c.hp.head_dim;
        return row_op(op_qk_prep, {x}, x->ne[0], x->ne[1], p, false).f;
    }

    // Length-dependent query scale (Scalable-Softmax) for `n` attended keys.
    float ssmax_scale(const std::string & pre, int n) const {
        if (c.hp.ssmax == SSMAX_NONE) return 1.0f;
        const float s  = *(const float *) w(pre + "scalable_factor.s")->data;
        const float ln = std::log((float) std::max(n, 1));
        if (c.hp.ssmax == SSMAX_SSMAX) return s * ln;
        const float b = *(const float *) w(pre + "scalable_factor.b")->data;
        return s * ln + c.hp.ssmax_base + std::max(b, 0.0f);
    }

    // q / k / v projections of `h`, then attention within each document of `lay`.
    ggml_tensor * attention(const std::string & pre, const act & h, const layout * lay,
                            const rope_table * rope, int half, bool ssmax) const {
        ggml_tensor * q = qk_prep(pre + "q_norm", linear(pre + "c_q.weight", h), lay, rope, false);
        ggml_tensor * k = qk_prep(pre + "k_norm", linear(pre + "c_k.weight", h), lay, rope, true);
        ggml_tensor * v = linear(pre + "c_v.weight", h);
        op_params * pv = params();
        pv->head_dim = c.hp.head_dim;
        v = row_op(op_head_major, {v}, v->ne[0], v->ne[1], pv, false).f;
        op_params * p = params();
        p->lay       = lay;
        p->head_dim  = c.hp.head_dim;
        p->n_head    = c.hp.n_head;
        p->n_kv_head = c.hp.n_kv_head;
        p->half      = half;
        p->scale.resize(lay->len.size());
        const float base = 1.0f / std::sqrt((float) c.hp.head_dim);
        for (size_t d = 0; d < lay->len.size(); ++d) {
            p->scale[d] = base * (ssmax ? ssmax_scale(pre, lay->len[d]) : 1.0f);
        }
        ggml_tensor * a[3] = { q, k, v };
        return ggml_custom_4d(g, GGML_TYPE_F32, q->ne[0], q->ne[1], 1, 1, a, 3, op_attention,
                              GGML_N_TASKS_MAX, p);
    }

    // x + c_proj(sigmoid(c_gate(y)) * z)
    ggml_tensor * gated_out(const std::string & pre, ggml_tensor * y, ggml_tensor * z, ggml_tensor * x) const {
        ggml_tensor * gl = linear(pre + "c_gate.weight", of(y));
        const act gz = row_op(op_gate, {gl, z}, z->ne[0], z->ne[1], params(),
                              int8_only({pre + "c_proj.weight"}));
        return linear(pre + "c_proj.weight", gz, x);
    }

    // full ("attn") and sliding-window ("swa") gated attention
    ggml_tensor * mix_attn(const std::string & pre, ggml_tensor * x, bool swa) const {
        const act h = rms_norm(norm_name(pre), x,
                               int8_only({pre + "c_q.weight", pre + "c_k.weight", pre + "c_v.weight"}));
        const int half = swa ? std::max(1, c.hp.swa_window / 2) : -1;
        ggml_tensor * y = attention(pre, h, &c.lay, &c.rope[swa ? 1 : 0], half, !swa);
        return gated_out(pre, y, y, x);
    }

    // SnowLily: liquid convolution
    ggml_tensor * mix_lily(const std::string & pre, ggml_tensor * x) const {
        const act h = rms_norm(norm_name(pre), x,
                               int8_only({pre + "ltc_conv.in_proj.weight", pre + "ltc_conv.A.weight",
                                          pre + "x_proj.weight", pre + "B_proj.weight"}));
        // low-rank state operator: out_proj(conv(in_proj h) + A h)
        ggml_tensor * z = linear(pre + "ltc_conv.in_proj.weight", h);
        ggml_tensor * r = linear(pre + "ltc_conv.A.weight", h);
        const act lr = row_op(op_dwconv, {z, r}, z->ne[0], z->ne[1],
                              conv_params(pre + "ltc_conv.kernel", z->ne[0]),
                              int8_only({pre + "ltc_conv.out_proj.weight"}));
        ggml_tensor * a  = linear(pre + "ltc_conv.out_proj.weight", lr);
        ggml_tensor * xp = linear(pre + "x_proj.weight", h);
        ggml_tensor * b  = linear(pre + "B_proj.weight", h);
        const act st = row_op(op_lily_state, {a, xp, b}, xp->ne[0], xp->ne[1],
                              conv_params(pre + "state_conv.depthwise.weight", xp->ne[0]),
                              int8_only({pre + "c_proj.weight"}));
        return linear(pre + "c_proj.weight", st, x);
    }

    // Hierarchical recurrent attention: attention over block means
    ggml_tensor * mix_hra(const std::string & pre, ggml_tensor * x) const {
        const int64_t C = x->ne[0], T = x->ne[1];
        const act h = rms_norm(norm_name(pre), x,
                               int8_only({pre + "qkv_in.weight", pre + "qkv_gate.weight", pre + "c_C.weight"}));
        ggml_tensor * i  = linear(pre + "qkv_in.weight", h);
        ggml_tensor * ci = row_op(op_dwconv, {i}, C, T, conv_params(pre + "qkv_conv.depthwise.weight", C), false).f;
        ggml_tensor * gt = linear(pre + "qkv_gate.weight", h);
        op_params * pm = params();
        pm->lay = &c.lay; pm->blay = &c.blay; pm->block = c.hp.hra_block;
        const act hb = row_op(op_block_mean, {ci, gt}, C, c.blay.total, pm,
                              int8_only({pre + "c_q.weight", pre + "c_k.weight", pre + "c_v.weight"}));
        ggml_tensor * yb = attention(pre, hb, &c.blay, &c.rope[0], -1, true);

        op_params * pu = params();
        pu->lay = &c.lay; pu->blay = &c.blay; pu->block = c.hp.hra_block;
        ggml_tensor * y = row_op(op_upsample_add, {yb, linear(pre + "c_C.weight", h)}, C, T, pu, false).f;
        ggml_tensor * co = row_op(op_dwconv, {y}, C, T, conv_params(pre + "c_o.depthwise.weight", C), false).f;
        return gated_out(pre, y, co, x);
    }

    // "blocks.3.attn." -> "blocks.3.attn_norm.weight"
    static std::string norm_name(const std::string & pre) {
        return pre.substr(0, pre.size() - 1) + "_norm.weight";
    }

    ggml_tensor * encoder(ggml_tensor * tokens) const {
        ggml_tensor * x = ggml_get_rows(g, w("embedding.weight"), tokens);
        const int n_layer = c.debug_n_layers >= 0 ? std::min(c.debug_n_layers, c.hp.n_layer) : c.hp.n_layer;
        for (int l = 0; l < n_layer; ++l) {
            const std::string blk = "blocks." + std::to_string(l) + ".";
            const std::string pre = blk + "attn.";
            switch (c.hp.block_types[l]) {
                case MIX_LILY: x = mix_lily(pre, x);        break;
                case MIX_SWA:  x = mix_attn(pre, x, true);  break;
                case MIX_ATTN: x = mix_attn(pre, x, false); break;
                case MIX_HRA:  x = mix_hra(pre, x);         break;
            }
            // x + c_proj(relu(c_fc(norm(x)))^2)
            const act h = rms_norm(blk + "mlp_norm.weight", x, int8_only({blk + "mlp.c_fc.weight"}));
            ggml_tensor * u = linear(blk + "mlp.c_fc.weight", h);
            const act r = row_op(op_relu_sqr, {u}, u->ne[0], u->ne[1], params(),
                                 int8_only({blk + "mlp.c_proj.weight"}));
            x = linear(blk + "mlp.c_proj.weight", r, x);
        }
        if (c.debug_n_layers >= 0) return x;
        x = ggml_norm(g, x, c.hp.ln_eps);
        return ggml_add(g, ggml_mul(g, x, w("out_norm.weight")), w("out_norm.bias"));
    }
};

// cos / sin of position * 1 / base^(2i / head_dim), computed in fp32 the way the
// PyTorch model builds its table.
void build_rope(rope_table & r, int n_pos, float base, int head_dim) {
    if (r.n_pos >= n_pos && r.base == base) return;
    const int half = head_dim / 2;
    r.n_pos = n_pos;
    r.base  = base;
    r.cos.resize((size_t) n_pos * half);
    r.sin.resize((size_t) n_pos * half);
    std::vector<float> inv(half);
    for (int i = 0; i < half; ++i) inv[i] = 1.0f / std::pow(base, (float) (2 * i) / (float) head_dim);
    for (int t = 0; t < n_pos; ++t) {
        for (int i = 0; i < half; ++i) {
            const float a = (float) t * inv[i];
            r.cos[(size_t) t * half + i] = std::cos(a);
            r.sin[(size_t) t * half + i] = std::sin(a);
        }
    }
}

// Lays the batch out and (re)builds whatever the custom ops read from the context.
bool prepare_batch(nextera_context & c, const int32_t * lengths, int32_t n_seqs) {
    if (n_seqs <= 0) return false;
    int max_len = 0;
    for (int i = 0; i < n_seqs; ++i) {
        if (lengths[i] <= 0) {
            fprintf(stderr, "nexterabert: empty sequence in batch\n");
            return false;
        }
        max_len = std::max(max_len, lengths[i]);
    }
    c.lay.build(lengths, n_seqs);
    std::vector<int32_t> nb(n_seqs);
    for (int i = 0; i < n_seqs; ++i) nb[i] = (lengths[i] + c.hp.hra_block - 1) / c.hp.hra_block;
    c.blay.build(nb.data(), n_seqs);

    float base = c.hp.rope_base;
    if (c.hp.ntk_train_len > 0 && max_len > c.hp.ntk_train_len) {
        // Dynamic NTK scaling stretches the base by the sequence length, so one
        // table cannot serve sequences of different lengths.
        if (n_seqs > 1) {
            fprintf(stderr, "nexterabert: sequences beyond rope.ntk_train_length must be encoded one at a time\n");
            return false;
        }
        const double d = c.hp.head_dim;
        base = (float) ((double) base * std::pow((double) max_len / c.hp.ntk_train_len, d / (d - 2.0)));
    }
    build_rope(c.rope[0], max_len, base, c.hp.head_dim);
    build_rope(c.rope[1], max_len, c.hp.swa_rope_base, c.hp.head_dim);
    return true;
}

const size_t GRAPH_NODES = 4096;

ggml_context * new_graph_ctx(nextera_context & c) {
    const size_t need = ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false);
    if (c.graph_mem.size() < need) c.graph_mem.resize(need);
    ggml_init_params ip = { c.graph_mem.size(), c.graph_mem.data(), true };
    c.ops.clear();
    return ggml_init(ip);
}

bool compute(nextera_context & c, ggml_cgraph * gf) {
    if (!ggml_gallocr_alloc_graph(c.galloc, gf)) {
        fprintf(stderr, "nexterabert: failed to allocate the compute buffer\n");
        return false;
    }
    return true;
}

const char * ftype_name(uint32_t ft) {
    switch (ft) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 7:  return "Q8_0";
        case 8:  return "Q5_0";
        case 9:  return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        default: return "unknown";
    }
}

uint32_t kv_u32(const gguf_context * gg, const char * key, uint32_t def, bool required = false) {
    const int64_t id = gguf_find_key(gg, key);
    if (id < 0) {
        if (required) { fprintf(stderr, "nexterabert: missing key '%s'\n", key); abort(); }
        return def;
    }
    return gguf_get_val_u32(gg, id);
}

float kv_f32(const gguf_context * gg, const char * key, float def) {
    const int64_t id = gguf_find_key(gg, key);
    return id < 0 ? def : gguf_get_val_f32(gg, id);
}

std::string kv_str(const gguf_context * gg, const char * key, const char * def) {
    const int64_t id = gguf_find_key(gg, key);
    return id < 0 ? def : gguf_get_val_str(gg, id);
}

// The buffer type ggml's CPU backend uses to re-lay out quantized weights for
// its interleaved GEMM kernels (NULL if this build has none).
ggml_backend_buffer_type_t find_repack_buft(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (!fn) return nullptr;
    ggml_backend_buffer_type_t * bufts = fn(dev);
    return (bufts && *bufts) ? *bufts : nullptr;
}

// Would mul_mat(weight, f32 activations) run on the repacked layout?
bool repack_supported(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft, const ggml_tensor * meta) {
    if (!buft || ggml_n_dims(meta) != 2 || !ggml_is_quantized(meta->type)) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * wt = ggml_new_tensor_2d(ctx, meta->type, meta->ne[0], meta->ne[1]);
    ggml_tensor * xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, meta->ne[0], 16);
    ggml_tensor * op = ggml_mul_mat(ctx, wt, xt);
    wt->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    const bool ok = wt->buffer && ggml_backend_dev_supports_op(dev, op);
    if (wt->buffer) ggml_backend_buffer_free(wt->buffer);
    ggml_free(ctx);
    return ok;
}

bool ends_with(const std::string & s, const char * suf) {
    const size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

int64_t file_seek(FILE * f, uint64_t off) {
#if defined(_WIN32)
    return _fseeki64(f, (__int64) off, SEEK_SET);
#else
    return fseeko(f, (off_t) off, SEEK_SET);
#endif
}

// ggml reports every repacked tensor at INFO level; keep warnings and errors only.
int g_log_verbosity = 0;

void log_callback(enum ggml_log_level level, const char * text, void *) {
    static bool show = true;
    if (level != GGML_LOG_LEVEL_CONT) {     // a continuation follows the line it continues
        show = level >= GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_NONE || g_log_verbosity >= 2;
    }
    if (show) fputs(text, stderr);
}

void reset_threadpool(nextera_context & c, int n_threads) {
    if (n_threads <= 0) {
        // All cores but ~10%: the pool's threads spin between graph nodes, so a
        // core lost to anything else stalls every one of them.
        const int hw = (int) std::max(1u, std::thread::hardware_concurrency());
        n_threads = hw - hw / 10;
    }
    c.n_threads = n_threads;
    ggml_backend_cpu_set_n_threads(c.backend, n_threads);
    ggml_backend_cpu_set_threadpool(c.backend, nullptr);
    if (c.threadpool) ggml_threadpool_free(c.threadpool);
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    c.threadpool = ggml_threadpool_new(&tpp);
    ggml_backend_cpu_set_threadpool(c.backend, c.threadpool);
}

} // namespace

// ---------------------------------------------------------------------------
// C API
// ---------------------------------------------------------------------------

struct nextera_params nextera_default_params(void) {
    nextera_params p;
    p.n_threads  = 0;
    p.use_repack = 1;
    p.use_int8   = 1;
    p.verbosity  = 0;
    return p;
}

struct nextera_context * nextera_load(const char * path, struct nextera_params params) {
    ggml_time_init();
    const int64_t t_start = ggml_time_us();
    g_log_verbosity = params.verbosity;
    ggml_log_set(log_callback, nullptr);

    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * gg = gguf_init_from_file(path, gp);
    if (!gg) {
        fprintf(stderr, "nexterabert: failed to open '%s'\n", path);
        return nullptr;
    }
    if (kv_str(gg, "general.architecture", "") != "nexterabert") {
        fprintf(stderr, "nexterabert: '%s' is not a nexterabert GGUF (general.architecture = '%s')\n",
                path, kv_str(gg, "general.architecture", "").c_str());
        gguf_free(gg);
        ggml_free(meta);
        return nullptr;
    }

    nextera_context * c = new nextera_context();
    hparams & hp = c->hp;
    hp.n_vocab       = kv_u32(gg, "nexterabert.vocab_size", 0, true);
    hp.n_embd        = kv_u32(gg, "nexterabert.embedding_length", 0, true);
    hp.n_layer       = kv_u32(gg, "nexterabert.block_count", 0, true);
    hp.n_inter       = kv_u32(gg, "nexterabert.feed_forward_length", 0, true);
    hp.n_head        = kv_u32(gg, "nexterabert.attention.head_count", 0, true);
    hp.n_kv_head     = kv_u32(gg, "nexterabert.attention.head_count_kv", 0, true);
    hp.head_dim      = hp.n_embd / hp.n_head;
    hp.lily_rank     = kv_u32(gg, "nexterabert.lily.low_rank", hp.head_dim);
    hp.lily_kernel   = kv_u32(gg, "nexterabert.lily.kernel_size", 5);
    hp.hra_block     = kv_u32(gg, "nexterabert.hra.block_size", 4);
    hp.swa_window    = kv_u32(gg, "nexterabert.swa.window", 256);
    hp.ntk_train_len = kv_u32(gg, "nexterabert.rope.ntk_train_length", 0);
    hp.rope_base     = kv_f32(gg, "nexterabert.rope.freq_base", 100000.0f);
    hp.swa_rope_base = kv_f32(gg, "nexterabert.swa.rope.freq_base", 10000.0f);
    hp.ssmax_base    = kv_f32(gg, "nexterabert.ssmax.base", 0.1f);
    hp.rms_eps       = kv_f32(gg, "nexterabert.rms_norm_epsilon", 1e-8f);
    hp.ln_eps        = kv_f32(gg, "nexterabert.layer_norm_epsilon", 1e-5f);
    hp.cls_id        = (int32_t) kv_u32(gg, "tokenizer.ggml.cls_token_id", (uint32_t) -1);
    hp.sep_id        = (int32_t) kv_u32(gg, "tokenizer.ggml.seperator_token_id", (uint32_t) -1);
    hp.pad_id        = (int32_t) kv_u32(gg, "tokenizer.ggml.padding_token_id", (uint32_t) -1);
    hp.mask_id       = (int32_t) kv_u32(gg, "tokenizer.ggml.mask_token_id", (uint32_t) -1);
    {
        const int64_t id = gguf_find_key(gg, "nexterabert.mlm_head");
        hp.has_mlm = id >= 0 && gguf_get_val_bool(gg, id);
    }
    const std::string mode = kv_str(gg, "nexterabert.ssmax.mode", "sssmax");
    hp.ssmax = mode == "none" ? SSMAX_NONE : mode == "ssmax" ? SSMAX_SSMAX : SSMAX_SSSMAX;

    bool ok = true;
    {
        const int64_t id = gguf_find_key(gg, "nexterabert.block_types");
        const size_t n = id < 0 ? 0 : gguf_get_arr_n(gg, id);
        if ((int32_t) n != hp.n_layer) {
            fprintf(stderr, "nexterabert: block_types has %zu entries, expected %d\n", n, hp.n_layer);
            ok = false;
        }
        for (size_t i = 0; ok && i < n; ++i) {
            const std::string t = gguf_get_arr_str(gg, id, i);
            if      (t == "lily") hp.block_types.push_back(MIX_LILY);
            else if (t == "swa")  hp.block_types.push_back(MIX_SWA);
            else if (t == "attn") hp.block_types.push_back(MIX_ATTN);
            else if (t == "hra")  hp.block_types.push_back(MIX_HRA);
            else { fprintf(stderr, "nexterabert: unsupported block type '%s'\n", t.c_str()); ok = false; }
        }
    }
    if (hp.head_dim % 2 != 0 || hp.n_head % hp.n_kv_head != 0) {
        fprintf(stderr, "nexterabert: unsupported head layout\n");
        ok = false;
    }

    c->tokenizer_json = kv_str(gg, "tokenizer.huggingface.json", "");
    c->ftype = ftype_name(kv_u32(gg, "general.file_type", 0));

    FILE * f = ok ? ggml_fopen(path, "rb") : nullptr;
    if (!f) {
        gguf_free(gg);
        ggml_free(meta);
        delete c;
        return nullptr;
    }

    c->backend = ggml_backend_cpu_init();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_buffer_type_t repack_buft = (params.use_repack && dev) ? find_repack_buft(dev) : nullptr;

    const int64_t n_tensors = gguf_get_n_tensors(gg);
    ggml_init_params wp = { ggml_tensor_overhead() * (size_t) (n_tensors + 1), nullptr, true };
    c->ctx_plain  = ggml_init(wp);
    c->ctx_repack = ggml_init(wp);

    // Where each tensor lives: on the int8 GEMM path (quantized linear weights,
    // when this build has the kernel), in ggml's repack buffer, or as is.
    std::vector<bool> repack(n_tensors, false), to_i8(n_tensors, false);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(gg, i);
        ggml_tensor * mt = ggml_get_tensor(meta, name.c_str());
        // The embedding matrix is read row-wise (get_rows) and must keep its plain layout.
        const bool gemm = name != "embedding.weight" && name != "mlm.decoder.weight"
                          && ggml_n_dims(mt) == 2 && ggml_is_quantized(mt->type);
#if defined(NEXTERA_INT8)
        to_i8[i] = params.use_int8 && gemm && mt->ne[0] % 32 == 0 && mt->ne[1] % 4 == 0;
#endif
        if (to_i8[i]) continue;
        repack[i] = gemm && repack_supported(dev, repack_buft, mt);
        ggml_tensor * t = ggml_dup_tensor(repack[i] ? c->ctx_repack : c->ctx_plain, mt);
        ggml_set_name(t, name.c_str());
        c->tensors[name] = t;
        c->n_repacked += repack[i] ? 1 : 0;
    }
    c->buf_plain = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx_plain, ggml_backend_cpu_buffer_type());
    if (c->n_repacked > 0) {
        c->buf_repack = ggml_backend_alloc_ctx_tensors_from_buft(c->ctx_repack, repack_buft);
    }

    std::vector<uint8_t> tmp;
    const uint64_t data_off = gguf_get_data_offset(gg);
    const int load_threads = (int) std::max(1u, std::thread::hardware_concurrency());
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(gg, i);
        ggml_tensor * t = to_i8[i] ? ggml_get_tensor(meta, name.c_str()) : c->tensors[name];
        const size_t n = ggml_nbytes(t);
        tmp.resize(n);
        if (file_seek(f, data_off + gguf_get_tensor_offset(gg, i)) != 0 || fread(tmp.data(), 1, n, f) != n) {
            fprintf(stderr, "nexterabert: failed to read tensor '%s'\n", name.c_str());
            ok = false;
            break;
        }
#if defined(NEXTERA_INT8)
        if (to_i8[i]) {
            build_i8_weight(c->i8[name], t->type, tmp.data(), (int) t->ne[0], (int) t->ne[1], load_threads);
            continue;
        }
#endif
        ggml_backend_tensor_set(t, tmp.data(), 0, n);

        // Conv kernels are stored (k, C) per channel; the op wants one row per tap.
        if (ends_with(name, ".depthwise.weight") || ends_with(name, ".kernel")) {
            const int k = (int) t->ne[0], C = (int) t->ne[1];
            const float * src = (const float *) tmp.data();
            std::vector<float> & taps = c->taps[name];
            taps.resize((size_t) k * C);
            for (int ch = 0; ch < C; ++ch) {
                for (int j = 0; j < k; ++j) taps[(size_t) j * C + ch] = src[(size_t) ch * k + j];
            }
        }
    }
    fclose(f);
    gguf_free(gg);
    ggml_free(meta);
    if (!ok) {
        nextera_free(c);
        return nullptr;
    }

    c->galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    reset_threadpool(*c, params.n_threads);

    char info[512];
    snprintf(info, sizeof(info),
             "AVX2 = %d | AVX_VNNI = %d | FMA = %d | F16C = %d | AVX512 = %d | NEON = %d | "
             "LLAMAFILE = %d | linear layers: %d int8 (%s), %d ggml repack | custom ops: %s",
             ggml_cpu_has_avx2(), ggml_cpu_has_avx_vnni(), ggml_cpu_has_fma(), ggml_cpu_has_f16c(),
             ggml_cpu_has_avx512(), ggml_cpu_has_neon(), ggml_cpu_has_llamafile(), (int) c->i8.size(),
#if defined(NEXTERA_INT8)
             NEXTERA_INT8_NAME,
#else
             "not built",
#endif
             c->n_repacked,
#if defined(NEXTERA_AVX2)
             "AVX2"
#else
             "scalar"
#endif
    );
    c->sysinfo = info;

    if (params.verbosity > 0) {
        size_t bytes = ggml_backend_buffer_get_size(c->buf_plain);
        if (c->buf_repack) bytes += ggml_backend_buffer_get_size(c->buf_repack);
        for (const auto & kv : c->i8) {
            bytes += kv.second.w.size() + 4 * kv.second.scale.size() + 4 * kv.second.off.size();
        }
        fprintf(stderr, "nexterabert: %s | %s | %d layers, n_embd %d | weights %.1f MiB | loaded in %.0f ms\n",
                path, c->ftype.c_str(), hp.n_layer, hp.n_embd, bytes / 1048576.0,
                (ggml_time_us() - t_start) / 1000.0);
        fprintf(stderr, "nexterabert: n_threads = %d | %s\n", c->n_threads, c->sysinfo.c_str());
    }
    return c;
}

void nextera_free(struct nextera_context * c) {
    if (!c) return;
    if (c->galloc) ggml_gallocr_free(c->galloc);
    if (c->backend) ggml_backend_cpu_set_threadpool(c->backend, nullptr);
    if (c->threadpool) ggml_threadpool_free(c->threadpool);
    if (c->buf_plain) ggml_backend_buffer_free(c->buf_plain);
    if (c->buf_repack) ggml_backend_buffer_free(c->buf_repack);
    if (c->ctx_plain) ggml_free(c->ctx_plain);
    if (c->ctx_repack) ggml_free(c->ctx_repack);
    if (c->backend) ggml_backend_free(c->backend);
    delete c;
}

int32_t nextera_n_embd(const struct nextera_context * c)       { return c->hp.n_embd; }
int32_t nextera_n_vocab(const struct nextera_context * c)      { return c->hp.n_vocab; }
int32_t nextera_n_layer(const struct nextera_context * c)      { return c->hp.n_layer; }
int32_t nextera_has_mlm_head(const struct nextera_context * c) { return c->hp.has_mlm ? 1 : 0; }
int32_t nextera_token_cls(const struct nextera_context * c)    { return c->hp.cls_id; }
int32_t nextera_token_sep(const struct nextera_context * c)    { return c->hp.sep_id; }
int32_t nextera_token_pad(const struct nextera_context * c)    { return c->hp.pad_id; }
int32_t nextera_token_mask(const struct nextera_context * c)   { return c->hp.mask_id; }
const char * nextera_tokenizer_json(const struct nextera_context * c) { return c->tokenizer_json.c_str(); }
const char * nextera_ftype(const struct nextera_context * c)          { return c->ftype.c_str(); }
const char * nextera_system_info(const struct nextera_context * c)    { return c->sysinfo.c_str(); }
int32_t nextera_get_n_threads(const struct nextera_context * c)       { return c->n_threads; }
double  nextera_last_ms(const struct nextera_context * c)             { return c->last_ms; }

void nextera_set_n_threads(struct nextera_context * c, int32_t n_threads) {
    reset_threadpool(*c, n_threads);
}

// Not part of the public header: run only the first `n_layers` blocks and return
// the residual stream before out_norm (-1 restores the full model). Used by
// python/verify.py to localise a mismatch against PyTorch.
extern "C" NEXTERA_API void nextera_debug_set_layers(struct nextera_context * c, int32_t n_layers) {
    c->debug_n_layers = n_layers;
}

// Diagnostics (also not in the public header): per-op wall time, as seen by thread 0.
extern "C" NEXTERA_API void nextera_debug_profile(struct nextera_context * c, int32_t enable) {
    c->prof = profiler();
    c->prof.enabled = enable != 0;
}

extern "C" NEXTERA_API void nextera_debug_profile_print(const struct nextera_context * c) {
    const profiler & pr = c->prof;
    if (!pr.calls) return;
    int64_t sum = 0;
    fprintf(stderr, "profile over %lld forward passes (ms per pass):\n", (long long) pr.calls);
    for (int i = 0; i < PROF_COUNT; ++i) {
        sum += pr.us[i];
        fprintf(stderr, "  %-22s %8.3f  %5.1f%%\n", PROF_NAMES[i], pr.us[i] / 1000.0 / pr.calls,
                100.0 * pr.us[i] / pr.total_us);
    }
    fprintf(stderr, "  %-22s %8.3f  %5.1f%%   (ggml ops, graph build, thread sync)\n", "everything else",
            (pr.total_us - sum) / 1000.0 / pr.calls, 100.0 * (pr.total_us - sum) / pr.total_us);
    fprintf(stderr, "  %-22s %8.3f\n", "total", pr.total_us / 1000.0 / pr.calls);
}

static bool check_tokens(const nextera_context & c, const int32_t * tokens, int32_t n) {
    for (int32_t i = 0; i < n; ++i) {
        if (tokens[i] < 0 || tokens[i] >= c.hp.n_vocab) {
            fprintf(stderr, "nexterabert: token id %d out of range\n", tokens[i]);
            return false;
        }
    }
    return true;
}

int32_t nextera_encode(struct nextera_context * c,
                       const int32_t * tokens, const int32_t * lengths, int32_t n_seqs,
                       float * hidden, float * pooled, int32_t normalize) {
    const int64_t t_start = ggml_time_us();
    if (!prepare_batch(*c, lengths, n_seqs)) return 1;
    const int32_t N = c->lay.total, C = c->hp.n_embd;
    if (!check_tokens(*c, tokens, N)) return 1;

    ggml_context * g = new_graph_ctx(*c);
    ggml_cgraph * gf = ggml_new_graph_custom(g, GRAPH_NODES, false);
    ggml_tensor * inp = ggml_new_tensor_1d(g, GGML_TYPE_I32, N);
    ggml_set_input(inp);
    builder b = { *c, g, {} };
    ggml_tensor * out = b.encoder(inp);
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    int32_t rc = 0;
    if (!compute(*c, gf)) {
        rc = 2;
    } else {
        ggml_backend_tensor_set(inp, tokens, 0, sizeof(int32_t) * (size_t) N);
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS) {
            rc = 3;
        } else {
            const float * od = (const float *) out->data;
            if (hidden) std::memcpy(hidden, od, sizeof(float) * (size_t) N * C);
            if (pooled) {
                for (int32_t d = 0; d < n_seqs; ++d) {
                    float * pv = pooled + (size_t) d * C;
                    std::memset(pv, 0, sizeof(float) * C);
                    const int32_t s = c->lay.start[d], n = c->lay.len[d];
                    for (int32_t t = 0; t < n; ++t) axpy_f32(pv, 1.0f, od + (size_t) (s + t) * C, C);
                    float scale = 1.0f / (float) n;
                    if (normalize) {
                        const float nrm = std::sqrt(dot_f32(pv, pv, C));
                        scale = nrm > 0.0f ? 1.0f / nrm : 0.0f;
                    }
                    for (int32_t i = 0; i < C; ++i) pv[i] *= scale;
                }
            }
        }
    }
    ggml_free(g);
    c->last_ms = (ggml_time_us() - t_start) / 1000.0;
    if (c->prof.enabled) {
        c->prof.total_us += ggml_time_us() - t_start;
        c->prof.calls++;
    }
    return rc;
}

int32_t nextera_mlm_logits(struct nextera_context * c,
                           const int32_t * tokens, int32_t n_tokens,
                           const int32_t * positions, int32_t n_pos,
                           float * logits) {
    const int64_t t_start = ggml_time_us();
    if (!c->hp.has_mlm) {
        fprintf(stderr, "nexterabert: this file has no MLM head\n");
        return 1;
    }
    if (n_pos <= 0 || !prepare_batch(*c, &n_tokens, 1) || !check_tokens(*c, tokens, n_tokens)) return 1;
    for (int32_t i = 0; i < n_pos; ++i) {
        if (positions[i] < 0 || positions[i] >= n_tokens) return 1;
    }

    ggml_context * g = new_graph_ctx(*c);
    ggml_cgraph * gf = ggml_new_graph_custom(g, GRAPH_NODES, false);
    ggml_tensor * inp = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, n_pos);
    ggml_set_input(inp);
    ggml_set_input(pos);
    builder b = { *c, g, {} };
    // dense -> GELU -> LayerNorm (no bias) -> tied decoder + bias, on the selected rows only
    ggml_tensor * h = ggml_get_rows(g, b.encoder(inp), pos);
    h = ggml_gelu_erf(g, b.linear("mlm.dense.weight", b.of(h)));
    h = ggml_mul(g, ggml_norm(g, h, c->hp.ln_eps), b.w("mlm.norm.weight"));
    const bool untied = c->tensors.count("mlm.decoder.weight") > 0;
    ggml_tensor * out = ggml_add(g, b.linear(untied ? "mlm.decoder.weight" : "embedding.weight", b.of(h)),
                                 b.w("mlm.bias"));
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    int32_t rc = 0;
    if (!compute(*c, gf)) {
        rc = 2;
    } else {
        ggml_backend_tensor_set(inp, tokens, 0, sizeof(int32_t) * (size_t) n_tokens);
        ggml_backend_tensor_set(pos, positions, 0, sizeof(int32_t) * (size_t) n_pos);
        if (ggml_backend_graph_compute(c->backend, gf) != GGML_STATUS_SUCCESS) {
            rc = 3;
        } else {
            std::memcpy(logits, out->data, sizeof(float) * (size_t) n_pos * c->hp.n_vocab);
        }
    }
    ggml_free(g);
    c->last_ms = (ggml_time_us() - t_start) / 1000.0;
    if (c->prof.enabled) {
        c->prof.total_us += ggml_time_us() - t_start;
        c->prof.calls++;
    }
    return rc;
}
