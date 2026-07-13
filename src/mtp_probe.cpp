// Cage harness: which step of MTP setup allocates? Run under MemoryMax=8G,
// MemorySwapMax=0. Steps: A model load (mmap) -> B target ctx -> C MTP ctx
// -> D speculative init + one draft. RSS printed after each; any bomb dies
// cleanly at the cage wall and the last line names the culprit.
#include "llama.h"
#include "common.h"
#include "speculative.h"

#include <cstdio>
#include <cstring>

static long rss_mb() {
    FILE * f = fopen("/proc/self/statm", "r");
    long size = 0, res = 0;
    if (f && fscanf(f, "%ld %ld", &size, &res) == 2) { fclose(f); return res * 4096 / (1 << 20); }
    if (f) fclose(f);
    return -1;
}
#define STEP(name) do { fprintf(stderr, "CAGE %-28s rss=%ld MB\n", name, rss_mb()); fflush(stderr); } while (0)

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: mtp_probe <model.gguf> [ngl]\n"); return 1; }
    const int ngl = argc > 2 ? atoi(argv[2]) : 0;

    llama_backend_init();
    STEP("backend_init");

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.use_mmap = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "CAGE FAIL model load\n"); return 1; }
    STEP("A: model load (mmap)");
    fprintf(stderr, "CAGE n_layer_nextn=%d\n", llama_model_n_layer_nextn(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.n_threads = 4;
    cp.n_threads_batch = 4;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "CAGE FAIL target ctx\n"); return 1; }
    STEP("B: target ctx");

    common_params params;
    params.n_ctx = 1024;
    params.n_batch = 128;
    params.n_ubatch = 128;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 3;
    params.model.path = argv[1];
    auto spec_res = common_speculative_init_from_params(params, model, ctx);
    llama_context * ctx_dft = spec_res ? spec_res->context() : nullptr;
    if (!ctx_dft) { fprintf(stderr, "CAGE FAIL mtp ctx\n"); return 1; }
    STEP("C: MTP ctx");

    params.speculative.draft.ctx_tgt = ctx;
    params.speculative.draft.ctx_dft = ctx_dft;
    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (!spec) { fprintf(stderr, "CAGE FAIL spec init\n"); return 1; }
    STEP("D1: speculative init");

    // one tiny target decode (2 tokens) then one draft
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_tokens prompt = common_tokenize(ctx, "The capital of France is", true, false);
    common_speculative_begin(spec, 0, prompt);
    llama_batch b = llama_batch_init((int) prompt.size(), 0, 1);
    for (size_t i = 0; i < prompt.size(); i++) {
        b.token[i] = prompt[i]; b.pos[i] = (llama_pos) i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = true;
    }
    b.n_tokens = (int) prompt.size();
    if (llama_decode(ctx, b)) { fprintf(stderr, "CAGE FAIL target decode\n"); return 1; }
    STEP("D2: target decode (prompt)");
    if (!common_speculative_process(spec, b)) { fprintf(stderr, "CAGE FAIL process\n"); return 1; }
    STEP("D3: speculative process");

    llama_tokens draft;
    llama_tokens ptoks(prompt.begin(), prompt.end() - 1);
    common_speculative_get_draft_params(spec, 0) = {
        true, -1, (llama_pos) prompt.size(), prompt.back(), &ptoks, &draft };
    common_speculative_draft(spec);
    STEP("D4: draft");
    fprintf(stderr, "CAGE draft tokens (%zu):", draft.size());
    for (auto t : draft) fprintf(stderr, " %d'%s'", t, common_token_to_piece(ctx, t).c_str());
    fprintf(stderr, "\nCAGE DONE\n");
    return 0;
}
