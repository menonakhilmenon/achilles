// trace-moe: generate with a GGUF MoE model while recording routed-expert IDs.
//
// Captures ffn_moe_topk-<layer> (i32 [k, n_tokens]) every forward pass via the
// backend-scheduler eval callback, plus ffn_moe_probs-<layer> (f32 [E, n_tokens])
// on decode steps (n_tokens==1) for router-margin analysis.
//
// Output (binary, little-endian) to $TRACE_OUT:
//   topk record:  u32 0x544F504B ('TOPK') | i32 step | i32 layer | i32 n_tokens
//                 | i32 k | k*n_tokens i32 expert ids
//   probs record: u32 0x50524F42 ('PROB') | i32 step | i32 layer | i32 n_tokens
//                 | i32 n_expert | n_expert*n_tokens f32
//
// Usage: TRACE_OUT=out.bin trace-moe -m model.gguf -p "prompt" -n 512 [-t 16 ...]
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct trace_ctx {
    FILE * out = nullptr;
    int    step = 0;
    size_t records = 0;
};

static bool cb_trace(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * tc = (trace_ctx *) user_data;
    const bool is_topk  = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const bool is_probs = strncmp(t->name, "ffn_moe_probs-", 14) == 0;
    if (ask) {
        return is_topk || (is_probs && t->ne[1] == 1); // probs only on decode steps
    }
    if (!is_topk && !is_probs) {
        return true;
    }
    const int layer    = atoi(strchr(t->name, '-') + 1);
    const int n        = (int) t->ne[0]; // k or n_expert
    const int n_tokens = (int) t->ne[1];

    const uint32_t magic = is_topk ? 0x544F504Bu : 0x50524F42u;
    const int32_t hdr[4] = {tc->step, layer, n_tokens, n};
    fwrite(&magic, 4, 1, tc->out);
    fwrite(hdr, 4, 4, tc->out);

    // tensors like ffn_moe_topk are strided views (row stride nb[1] spans the
    // full parent row) — copy row by row, writing only ne[0] elements per row
    static std::vector<uint8_t> buf;
    const size_t row_bytes = (size_t) n * ggml_type_size(t->type);
    buf.resize(row_bytes);
    for (int j = 0; j < n_tokens; j++) {
        ggml_backend_tensor_get(t, buf.data(), (size_t) j * t->nb[1], row_bytes);
        fwrite(buf.data(), 1, row_bytes, tc->out);
    }
    tc->records++;
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    const char * trace_path = getenv("TRACE_OUT");
    if (!trace_path) {
        LOG_ERR("TRACE_OUT env var required\n");
        return 1;
    }
    trace_ctx tc;
    tc.out = fopen(trace_path, "wb");
    if (!tc.out) {
        LOG_ERR("cannot open %s\n", trace_path);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = cb_trace;
    params.cb_eval_user_data = &tc;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    if (!model || !lctx) {
        LOG_ERR("failed to init model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens =
        common_tokenize(lctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    LOG_INF("prefill tokens: %zu\n", tokens.size());

    // prefill (step 0)
    if (llama_decode(lctx, llama_batch_get_one(tokens.data(), (int32_t) tokens.size()))) {
        LOG_ERR("prefill failed\n");
        return 1;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    std::string text;
    const int n_gen = params.n_predict > 0 ? params.n_predict : 512;
    int gen = 0;
    for (; gen < n_gen; gen++) {
        llama_token tok = llama_sampler_sample(smpl, lctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) {
            break;
        }
        text += common_token_to_piece(lctx, tok);
        tc.step++;
        if (llama_decode(lctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("decode failed at step %d\n", tc.step);
            break;
        }
    }

    fclose(tc.out);
    llama_sampler_free(smpl);
    LOG_INF("\n=== generated %d tokens, %zu trace records -> %s\n",
            gen, tc.records, trace_path);
    fprintf(stderr, "OUTPUT_PREVIEW: %.300s\n", text.c_str());
    return 0;
}
