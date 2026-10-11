// nexterabert-quantize -- re-encode the GEMM weights of a NexteraBERT GGUF with
// one of ggml's block quantization formats (the llama-quantize of this repo).
//
//   nexterabert-quantize in-f32.gguf out-Q4_K_M.gguf Q4_K_M [n_threads]
//       [--tensor-type SUBSTRING=TYPE ...]
//
// Only 2-D weights are quantized. Norm gains, DyT curves, conv kernels, biases
// and the SSSMax scalars stay f32: they are < 0.1% of the file and are read by
// element-wise code, not by a GEMM.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

struct preset {
    const char * name;
    uint32_t     ftype;      // llama.cpp's general.file_type numbering
    ggml_type    base;       // every quantized tensor
    ggml_type    more;       // tensors that get extra bits ("_M")
    const char * desc;
};

const preset PRESETS[] = {
    { "F32",    0,  GGML_TYPE_F32,  GGML_TYPE_F32,  "no quantization" },
    { "F16",    1,  GGML_TYPE_F16,  GGML_TYPE_F16,  "half precision" },
    { "Q8_0",   7,  GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, "8.5 bpw" },
    { "Q6_K",   18, GGML_TYPE_Q6_K, GGML_TYPE_Q6_K, "6.56 bpw" },
    { "Q5_K_M", 17, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, "5.5 bpw, residual-stream writers at Q6_K" },
    { "Q5_K_S", 16, GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, "5.5 bpw" },
    { "Q5_0",   8,  GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, "5.5 bpw, legacy uniform blocks" },
    { "Q4_K_M", 15, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, "4.5 bpw, residual-stream writers at Q6_K (recommended 4-bit)" },
    { "Q4_K_S", 14, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, "4.5 bpw" },
    { "Q4_0",   2,  GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, "4.5 bpw, legacy uniform blocks" },
    { "Q3_K_M", 12, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, "3.44 bpw, residual-stream writers at Q4_K" },
};

const std::pair<const char *, ggml_type> TYPE_NAMES[] = {
    { "F32", GGML_TYPE_F32 },   { "F16", GGML_TYPE_F16 },   { "Q8_0", GGML_TYPE_Q8_0 },
    { "Q6_K", GGML_TYPE_Q6_K }, { "Q5_K", GGML_TYPE_Q5_K }, { "Q5_0", GGML_TYPE_Q5_0 },
    { "Q5_1", GGML_TYPE_Q5_1 }, { "Q4_K", GGML_TYPE_Q4_K }, { "Q4_0", GGML_TYPE_Q4_0 },
    { "Q4_1", GGML_TYPE_Q4_1 }, { "Q3_K", GGML_TYPE_Q3_K }, { "Q2_K", GGML_TYPE_Q2_K },
};

std::string upper(std::string s) {
    for (char & ch : s) ch = (char) toupper((unsigned char) ch);
    return s;
}

bool parse_type(const std::string & s, ggml_type & out) {
    for (const auto & t : TYPE_NAMES) {
        if (upper(s) == t.first) { out = t.second; return true; }
    }
    return false;
}

bool ends_with(const std::string & s, const char * suf) {
    const size_t n = strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// The "_M" presets keep at Q6_K the tensors whose 4-bit error reaches the output
// most directly (measured one group at a time, see README "Choosing a file type"):
//  * every projection that writes into the residual stream -- the mixers'
//    c_proj and the MLP's c_proj;
//  * the attention value projection and the sigmoid output gate;
//  * the dense transform of the MLM head, of the per-token heads and of the pooler.
// Queries / keys, the SnowLily input projections, the MLP up-projection and the
// embedding matrix tolerate 4 bits far better and stay at the base type.
bool wants_more_bits(const std::string & name) {
    return ends_with(name, ".attn.c_proj.weight") || ends_with(name, ".mlp.c_proj.weight")
        || ends_with(name, ".attn.c_v.weight") || ends_with(name, ".attn.c_gate.weight")
        || name == "mlm.dense.weight" || name == "head.dense.weight" || name == "pooler.dense.weight";
}

// K-quants pack 256 weights per super-block; rows that do not divide fall back
// to the nearest legacy format (same rule as llama.cpp).
ggml_type fallback_for_row(ggml_type t, int64_t n_per_row) {
    if (n_per_row % ggml_blck_size(t) == 0) return t;
    switch (t) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K: t = GGML_TYPE_Q4_0; break;
        case GGML_TYPE_Q4_K: t = GGML_TYPE_Q5_0; break;
        case GGML_TYPE_Q5_K: t = GGML_TYPE_Q5_1; break;
        case GGML_TYPE_Q6_K: t = GGML_TYPE_Q8_0; break;
        default: break;
    }
    return n_per_row % ggml_blck_size(t) == 0 ? t : GGML_TYPE_F32;
}

void usage(const char * argv0) {
    fprintf(stderr, "usage: %s input.gguf output.gguf TYPE [n_threads] [--tensor-type SUBSTRING=TYPE ...]\n\n", argv0);
    fprintf(stderr, "file types:\n");
    for (const preset & p : PRESETS) fprintf(stderr, "  %-7s %s\n", p.name, p.desc);
    fprintf(stderr, "\n--tensor-type overrides the type of every tensor whose name contains SUBSTRING\n"
                    "(tensor types: F32 F16 Q8_0 Q6_K Q5_K Q5_0 Q5_1 Q4_K Q4_0 Q4_1 Q3_K Q2_K).\n");
}

} // namespace

int main(int argc, char ** argv) {
    std::vector<std::string> pos;
    std::vector<std::pair<std::string, ggml_type>> overrides;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--tensor-type" && i + 1 < argc) {
            const std::string v = argv[++i];
            const size_t eq = v.rfind('=');
            ggml_type t;
            if (eq == std::string::npos || !parse_type(v.substr(eq + 1), t)) {
                fprintf(stderr, "bad --tensor-type '%s'\n", v.c_str());
                return 1;
            }
            overrides.emplace_back(v.substr(0, eq), t);
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            pos.push_back(a);
        }
    }
    if (pos.size() < 3) {
        usage(argv[0]);
        return 1;
    }
    const preset * ps = nullptr;
    for (const preset & p : PRESETS) {
        if (upper(pos[2]) == p.name) ps = &p;
    }
    if (!ps) {
        fprintf(stderr, "unknown file type '%s'\n\n", pos[2].c_str());
        usage(argv[0]);
        return 1;
    }
    int n_threads = pos.size() > 3 ? atoi(pos[3].c_str()) : 0;
    if (n_threads <= 0) n_threads = (int) std::max(1u, std::thread::hardware_concurrency());

    ggml_context * ctx = nullptr;
    gguf_init_params gp = { false, &ctx };
    gguf_context * in = gguf_init_from_file(pos[0].c_str(), gp);
    if (!in) {
        fprintf(stderr, "failed to read '%s'\n", pos[0].c_str());
        return 1;
    }

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    gguf_set_val_u32(out, "general.file_type", ps->ftype);
    gguf_set_val_u32(out, "general.quantization_version", GGML_QNT_VERSION);

    const int64_t n_tensors = gguf_get_n_tensors(in);
    std::vector<std::vector<uint8_t>> store((size_t) n_tensors);
    std::vector<float> f32;
    size_t bytes_in = 0, bytes_out = 0;

    for (int64_t i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(in, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        const size_t src_bytes = ggml_nbytes(t);
        bytes_in += src_bytes;

        // The classifier is the output layer and a few kilobytes: it stays as it is.
        const bool is_matrix = ggml_n_dims(t) == 2 && ends_with(name, ".weight")
                               && name != "classifier.weight"
                               && t->ne[0] >= 32 && t->ne[1] >= 32
                               && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16);
        ggml_type target = t->type;
        if (is_matrix) {
            target = wants_more_bits(name) ? ps->more : ps->base;
            for (const auto & o : overrides) {
                if (name.find(o.first) != std::string::npos) target = o.second;
            }
            target = fallback_for_row(target, t->ne[0]);
        }

        gguf_add_tensor(out, t);
        if (target == t->type) {
            bytes_out += src_bytes;
            continue;   // data pointer already set by gguf_add_tensor
        }

        const int64_t n_per_row = t->ne[0], nrows = t->ne[1];
        const float * src = (const float *) t->data;
        if (t->type == GGML_TYPE_F16) {
            f32.resize((size_t) (n_per_row * nrows));
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) t->data, f32.data(), n_per_row * nrows);
            src = f32.data();
        }

        std::vector<uint8_t> & dst = store[(size_t) i];
        dst.resize(ggml_row_size(target, n_per_row) * (size_t) nrows);
        ggml_quantize_init(target);
        // Rows are independent, so hand each thread a contiguous slab of them.
        const int nt = (int) std::min<int64_t>(n_threads, nrows);
        std::atomic<int64_t> next(0);
        const int64_t slab = std::max<int64_t>(1, nrows / (nt * 8));
        std::vector<std::thread> workers;
        for (int w = 0; w < nt; ++w) {
            workers.emplace_back([&]() {
                for (;;) {
                    const int64_t r0 = next.fetch_add(slab);
                    if (r0 >= nrows) break;
                    const int64_t n = std::min(slab, nrows - r0);
                    ggml_quantize_chunk(target, src, dst.data(), r0 * n_per_row, n, n_per_row, nullptr);
                }
            });
        }
        for (std::thread & th : workers) th.join();

        gguf_set_tensor_type(out, name.c_str(), target);
        gguf_set_tensor_data(out, name.c_str(), dst.data());
        bytes_out += dst.size();
        printf("%-48s [%5lld x %5lld]  %-4s -> %-5s %8.2f MiB -> %7.2f MiB\n", name.c_str(),
               (long long) n_per_row, (long long) nrows, ggml_type_name(t->type), ggml_type_name(target),
               src_bytes / 1048576.0, dst.size() / 1048576.0);
    }

    if (!gguf_write_to_file(out, pos[1].c_str(), false)) {
        fprintf(stderr, "failed to write '%s'\n", pos[1].c_str());
        return 1;
    }
    printf("\n%s: %.1f MiB -> %.1f MiB  (%s)\n", pos[1].c_str(), bytes_in / 1048576.0,
           bytes_out / 1048576.0, ps->name);

    gguf_free(out);
    gguf_free(in);
    ggml_free(ctx);
    ggml_quantize_free();
    return 0;
}
