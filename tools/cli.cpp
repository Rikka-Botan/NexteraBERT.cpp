// nexterabert-cli -- run a NexteraBERT GGUF from the command line.
//
//   nexterabert-cli -m model.gguf --info
//   nexterabert-cli -m model.gguf --ids 50281,2,3,50282           embedding of one token sequence
//   nexterabert-cli -m model.gguf --bench 128 [--batch 8]         time random sequences of 128 tokens
//
// Tokenization lives in the Python binding (python/nexterabert_cpp.py), which
// reads tokenizer.json out of the same GGUF; the CLI works on token ids.

#include "nexterabert.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" {
// diagnostics exported by the library but kept out of the public header
NEXTERA_API void nextera_debug_profile(struct nextera_context * ctx, int32_t enable);
NEXTERA_API void nextera_debug_profile_print(const struct nextera_context * ctx);
}

namespace {

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf [options]\n\n"
        "  -t, --threads N     threads (default: all cores)\n"
        "  --info              print model / system info\n"
        "  --ids A,B,C         token ids of one sequence (repeat for a batch)\n"
        "  --full              print whole embedding vectors instead of a preview\n"
        "  --no-normalize      do not L2-normalise the pooled embedding\n"
        "  --bench T           time forward passes over random sequences of T tokens\n"
        "  --batch B           sequences per forward pass in --bench (default 1)\n"
        "  --reps R            timed repetitions in --bench (default 20)\n"
        "  --profile           with --bench: print where the time goes\n"
        "  --no-int8           run quantized layers on ggml's kernels instead of the int8 GEMM\n"
        "  --no-repack         (with --no-int8) keep quantized weights in their file layout\n",
        argv0);
}

std::vector<int32_t> parse_ids(const char * s) {
    std::vector<int32_t> ids;
    while (*s) {
        char * end = nullptr;
        const long v = strtol(s, &end, 10);
        if (end == s) break;
        ids.push_back((int32_t) v);
        s = (*end == ',') ? end + 1 : end;
    }
    return ids;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model;
    nextera_params params = nextera_default_params();
    params.verbosity = 1;
    std::vector<std::vector<int32_t>> seqs;
    bool info = false, full = false, normalize = true, profile = false;
    int bench_t = 0, batch = 1, reps = 20;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")   model = next();
        else if (a == "-t" || a == "--threads") params.n_threads = atoi(next());
        else if (a == "--info")                 info = true;
        else if (a == "--ids")                  seqs.push_back(parse_ids(next()));
        else if (a == "--full")                 full = true;
        else if (a == "--no-normalize")         normalize = false;
        else if (a == "--bench")                bench_t = atoi(next());
        else if (a == "--batch")                batch = atoi(next());
        else if (a == "--reps")                 reps = atoi(next());
        else if (a == "--profile")              profile = true;
        else if (a == "--no-int8")              params.use_int8 = 0;
        else if (a == "--no-repack")            params.use_repack = 0;
        else { usage(argv[0]); return a == "-h" || a == "--help" ? 0 : 1; }
    }
    if (model.empty()) {
        usage(argv[0]);
        return 1;
    }

    nextera_context * ctx = nextera_load(model.c_str(), params);
    if (!ctx) return 1;
    const int n_embd = nextera_n_embd(ctx);

    if (info) {
        printf("file type  : %s\n", nextera_ftype(ctx));
        printf("n_layer    : %d\n", nextera_n_layer(ctx));
        printf("n_embd     : %d\n", n_embd);
        printf("n_vocab    : %d\n", nextera_n_vocab(ctx));
        printf("mlm head   : %s\n", nextera_has_mlm_head(ctx) ? "yes" : "no");
        printf("tokenizer  : %zu bytes of tokenizer.json\n", strlen(nextera_tokenizer_json(ctx)));
        printf("n_threads  : %d\n", nextera_get_n_threads(ctx));
        printf("system     : %s\n", nextera_system_info(ctx));
    }

    if (!seqs.empty()) {
        std::vector<int32_t> tokens, lengths;
        for (const auto & s : seqs) {
            tokens.insert(tokens.end(), s.begin(), s.end());
            lengths.push_back((int32_t) s.size());
        }
        std::vector<float> pooled(seqs.size() * (size_t) n_embd);
        if (nextera_encode(ctx, tokens.data(), lengths.data(), (int32_t) seqs.size(),
                           nullptr, pooled.data(), normalize ? 1 : 0) != 0) {
            nextera_free(ctx);
            return 1;
        }
        for (size_t s = 0; s < seqs.size(); ++s) {
            const float * v = pooled.data() + s * n_embd;
            const int n = full ? n_embd : std::min(n_embd, 8);
            printf("[%zu] (%d tokens)", s, lengths[s]);
            for (int i = 0; i < n; ++i) printf(" %.6f", v[i]);
            printf(full ? "\n" : " ...\n");
        }
        fprintf(stderr, "encode: %.2f ms\n", nextera_last_ms(ctx));
    }

    if (bench_t > 0) {
        std::mt19937 rng(0);
        std::uniform_int_distribution<int32_t> dist(1000, std::min(nextera_n_vocab(ctx), 40000) - 1);
        std::vector<int32_t> tokens((size_t) bench_t * batch), lengths(batch, bench_t);
        for (int32_t & t : tokens) t = dist(rng);
        for (int b = 0; b < batch; ++b) {
            tokens[(size_t) b * bench_t] = nextera_token_cls(ctx);
            tokens[(size_t) (b + 1) * bench_t - 1] = nextera_token_sep(ctx);
        }
        std::vector<float> pooled((size_t) batch * n_embd);
        std::vector<double> ms;
        for (int r = 0; r < reps + 3; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            if (nextera_encode(ctx, tokens.data(), lengths.data(), batch, nullptr, pooled.data(), 1) != 0) {
                nextera_free(ctx);
                return 1;
            }
            const auto t1 = std::chrono::steady_clock::now();
            if (r >= 3) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            if (r == 2 && profile) nextera_debug_profile(ctx, 1);
        }
        if (profile) nextera_debug_profile_print(ctx);
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        printf("bench: %s  T=%d  batch=%d  threads=%d  median %.2f ms  min %.2f ms  (%.0f tokens/s)\n",
               nextera_ftype(ctx), bench_t, batch, nextera_get_n_threads(ctx), med, ms.front(),
               1000.0 * bench_t * batch / med);
    }

    nextera_free(ctx);
    return 0;
}
