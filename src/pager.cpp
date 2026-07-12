// achilles-pager: managed expert paging for llama.cpp MoE inference (Phase 2 v1).
//
// Strategy: llama.cpp mmaps the GGUF, so the OS page cache is the expert cache.
// We steer it instead of letting the kernel guess:
//   - PREFETCH: gate-ahead prediction (router weights applied to layer-l hidden
//     state for layers l+1..l+DELTA) -> posix_fadvise(WILLNEED) on the predicted
//     experts' file ranges, expert-granular, queued by deadline.
//   - EVICT: decayed-LFU over (layer, expert); posix_fadvise(DONTNEED) on the
//     coldest experts when the residency budget is exceeded (replaces kernel LRU).
//   - ACCOUNT: the ffn_moe_topk callback tells us every real expert activation
//     (demand traffic), keeping scores and residency truthful.
//
// All measured deltas from Phase 1 are addressed except fault granularity
// (still 4KiB faults on a miss, but WILLNEED readahead is range-sized); the
// O_DIRECT arena that fixes misses too is Phase 2 v2.
//
// Usage:
//   achilles-pager -m model.gguf -p "prompt" -n 128 -t 10 -ngl 99 -ot exps=CPU \
//       --budget-gib 12 --delta 4 --fetch 12 [--no-pager] [--stats]
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct expert_slice {
    off_t  off;
    size_t len;
};

struct file_map { uintptr_t start, end; size_t foff; };

struct pager {
    int fd = -1;
    int n_layer = 0, n_expert = 0, n_embd = 0, n_used = 0;
    std::vector<file_map> maps;                // mmap regions of the model file
    std::string model_path;
    bool sigmoid_bias = false;                 // GLM-style: sigmoid(logits)+bias ranking
    std::vector<std::vector<std::vector<expert_slice>>> slices;  // [layer][expert][proj]
    std::vector<std::vector<float>> rw;        // router weights [layer] (E x n_embd)
    std::vector<std::vector<float>> rb;        // router bias    [layer] (E) or empty
    size_t slice_bytes_per_expert = 0;

    // state
    std::vector<float> score;                  // decayed LFU, [layer*E]
    std::vector<uint8_t> resident;             // [layer*E]
    size_t budget = 0, resident_bytes = 0;
    std::mutex mu;

    // prefetch queue
    struct req { int layer, expert, prio; };
    std::deque<req> q;
    std::condition_variable cv;
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;

    // stats
    std::atomic<uint64_t> n_prefetch{0}, n_evict{0}, n_demand_hit{0}, n_demand_miss{0};

    int idx(int l, int e) const { return l * n_expert + e; }

    void enqueue(int l, int e, int prio) {
        std::lock_guard<std::mutex> lk(mu);
        if (resident[idx(l, e)]) return;
        q.push_back({l, e, prio});
        std::push_heap(q.begin(), q.end(), [](const req &a, const req &b) { return a.prio > b.prio; });
        cv.notify_one();
    }

    void worker() {
        while (!stop) {
            req r;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !q.empty() || !pending_drops.empty(); });
                if (stop) return;
                if (q.empty()) { lk.unlock(); drain_drops(); continue; }
                std::pop_heap(q.begin(), q.end(), [](const req &a, const req &b) { return a.prio > b.prio; });
                r = q.back();
                q.pop_back();
                if (resident[idx(r.layer, r.expert)]) continue;
                // protect fresh prefetches from immediate LFU eviction
                score[idx(r.layer, r.expert)] = std::max(score[idx(r.layer, r.expert)], 1.5f);
                mark_resident_locked(r.layer, r.expert);
            }
            for (const auto &s : slices[r.layer][r.expert]) {
                posix_fadvise(fd, s.off, s.len, POSIX_FADV_WILLNEED);
            }
            n_prefetch++;
            drain_drops();
        }
    }

    // caller holds mu
    void mark_resident_locked(int l, int e) {
        if (resident[idx(l, e)]) return;
        resident[idx(l, e)] = 1;
        resident_bytes += slice_bytes_per_expert;
        while (resident_bytes > budget) {
            evict_coldest_locked();
        }
    }

    // find llama.cpp's mmap regions for the model file (call once, post-init)
    void init_mappings() {
        FILE * f = fopen("/proc/self/maps", "r");
        if (!f) return;
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (!strstr(line, model_path.c_str())) continue;
            uintptr_t start, end; size_t foff;
            if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR " %*4s %zx", &start, &end, &foff) == 3) {
                maps.push_back({start, end, foff});
            }
        }
        fclose(f);
        LOG_INF("pager: %zu mmap region(s) of model found\n", maps.size());
    }

    // drop an expert's pages: madvise on the mapping (page-aligned inward, so
    // neighbouring experts' boundary pages survive) + fadvise for the cache
    void drop_slice(const expert_slice &s) {
        posix_fadvise(fd, s.off, s.len, POSIX_FADV_DONTNEED);
        const long pg = sysconf(_SC_PAGESIZE);
        for (const auto &m : maps) {
            if ((size_t) s.off < m.foff || (size_t) s.off + s.len > m.foff + (m.end - m.start)) continue;
            uintptr_t a = m.start + (s.off - m.foff);
            uintptr_t lo = (a + pg - 1) & ~(uintptr_t)(pg - 1);
            uintptr_t hi = (a + s.len) & ~(uintptr_t)(pg - 1);
            if (hi > lo) madvise((void *) lo, hi - lo, MADV_DONTNEED);
            break;
        }
    }

    void evict_coldest_locked() {
        float best = 1e30f;
        int bl = -1, be = -1;
        for (int l = 0; l < n_layer; l++) {
            for (int e = 0; e < n_expert; e++) {
                if (resident[idx(l, e)] && score[idx(l, e)] < best) {
                    best = score[idx(l, e)];
                    bl = l; be = e;
                }
            }
        }
        if (bl < 0) return;
        resident[idx(bl, be)] = 0;
        resident_bytes -= slice_bytes_per_expert;
        pending_drops.push_back({bl, be}); // syscalls happen outside the lock
        n_evict++;
    }

    std::vector<std::pair<int,int>> pending_drops; // victims awaiting syscalls

    // real activations for layer l (from ffn_moe_topk). Fires a few graph nodes
    // BEFORE the expert matmul executes, so WILLNEED here turns the matmul's
    // serial 4KiB demand faults into parallel expert-sized readahead.
    void on_topk(int l, const int32_t *ids, int k, int n_tokens) {
        std::vector<int> need;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (l == 0) { // decay once per forward pass
                for (auto &s : score) s *= 0.98f;
            }
            for (int t = 0; t < n_tokens; t++) {
                for (int i = 0; i < k; i++) {
                    int e = ids[t * k + i];
                    if (e < 0 || e >= n_expert) continue;
                    score[idx(l, e)] += 1.0f;
                    if (resident[idx(l, e)]) n_demand_hit++;
                    else {
                        n_demand_miss++;
                        need.push_back(e);
                        mark_resident_locked(l, e);
                    }
                }
            }
        }
        // issue readahead for imminent misses (cheap, async); leave the
        // expensive DONTNEED page-table scans to the worker threads
        for (int e : need) {
            for (const auto &s : slices[l][e]) {
                posix_fadvise(fd, s.off, s.len, POSIX_FADV_WILLNEED);
            }
        }
        cv.notify_one();
    }

    void drain_drops() {
        std::vector<std::pair<int,int>> victims;
        {
            std::lock_guard<std::mutex> lk(mu);
            victims.swap(pending_drops);
        }
        for (auto [l, e] : victims) {
            for (const auto &s : slices[l][e]) {
                drop_slice(s);
            }
        }
    }

    // gate-ahead prediction from layer l's hidden state (decode only).
    // sliding window: each layer predicts ONLY l+delta (earlier offsets were
    // predicted by earlier layers), except layer 0 which warms 1..delta.
    void on_hidden(int l, const float *h, int delta, int fetch) {
        if (fetch <= 0) return;
        const int d_lo = (l == 0) ? 1 : delta;
        for (int d = d_lo; d <= delta && l + d < n_layer; d++) {
            const auto &W = rw[l + d];
            if (W.empty()) continue;
            std::vector<std::pair<float, int>> sc(n_expert);
            for (int e = 0; e < n_expert; e++) {
                float dot = 0;
                const float *w = &W[(size_t) e * n_embd];
                for (int j = 0; j < n_embd; j++) dot += w[j] * h[j];
                if (sigmoid_bias) {
                    dot = 1.0f / (1.0f + expf(-dot));
                    if (!rb[l + d].empty()) dot += rb[l + d][e];
                }
                sc[e] = {dot, e};
            }
            std::partial_sort(sc.begin(), sc.begin() + fetch, sc.end(),
                              [](auto &a, auto &b) { return a.first > b.first; });
            for (int i = 0; i < fetch; i++) {
                enqueue(l + d, sc[i].second, d);
            }
        }
    }
};

static pager g_pager;
static bool g_pager_on = true;
static int g_delta = 4, g_fetch = 12, g_workers = 2;

static bool cb_pager(struct ggml_tensor * t, bool ask, void *) {
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    // router input: "ffn_norm-<l>" (qwen3moe et al) or "post_attn_norm-<l>" (glm4moe)
    const bool is_norm = strncmp(t->name, "ffn_norm-", 9) == 0 ||
                         strncmp(t->name, "post_attn_norm-", 15) == 0;
    if (ask) {
        return g_pager_on && (is_topk || (is_norm && t->ne[1] == 1));
    }
    if (!g_pager_on) return true;
    if (is_topk) {
        const int l = atoi(t->name + 13);
        const int k = (int) t->ne[0], n_tokens = (int) t->ne[1];
        static std::vector<int32_t> ids;
        ids.resize((size_t) k * n_tokens);
        for (int j = 0; j < n_tokens; j++) {
            ggml_backend_tensor_get(t, ids.data() + (size_t) j * k, j * t->nb[1], k * sizeof(int32_t));
        }
        g_pager.on_topk(l, ids.data(), k, n_tokens);
    } else if (is_norm && t->ne[1] == 1 && t->type == GGML_TYPE_F32) {
        const int l = atoi(strchr(t->name, '-') + 1);
        static std::vector<float> h;
        h.resize(t->ne[0]);
        ggml_backend_tensor_get(t, h.data(), 0, t->ne[0] * sizeof(float));
        g_pager.on_hidden(l, h.data(), g_delta, g_fetch);
    }
    return true;
}

static bool load_expert_map(const char * path, size_t budget_bytes) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*no_alloc*/ true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) { LOG_ERR("gguf parse failed\n"); return false; }

    const int i_arch = gguf_find_key(g, "general.architecture");
    const std::string arch = gguf_get_val_str(g, i_arch);
    auto kv_int = [&](const char * suffix) -> int {
        const int i = gguf_find_key(g, (arch + "." + suffix).c_str());
        return i < 0 ? 0 : (int) gguf_get_val_u32(g, i);
    };
    g_pager.n_layer  = kv_int("block_count");
    g_pager.n_expert = kv_int("expert_count");
    g_pager.n_embd   = kv_int("embedding_length");
    g_pager.n_used   = kv_int("expert_used_count");

    const size_t data_off = gguf_get_data_offset(g);
    g_pager.slices.assign(g_pager.n_layer, {});
    g_pager.rw.assign(g_pager.n_layer, {});
    g_pager.rb.assign(g_pager.n_layer, {});
    for (auto &v : g_pager.slices) v.assign(g_pager.n_expert, {});

    g_pager.fd = open(path, O_RDONLY);

    const char * projs[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    size_t per_expert_total = 0;  // from the first MoE layer (early layers may be dense)
    for (int l = 0; l < g_pager.n_layer; l++) {
        size_t layer_expert_bytes = 0;
        for (const char * pj : projs) {
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, pj);
            const int64_t ti = gguf_find_tensor(g, name);
            if (ti < 0) continue;
            ggml_tensor * t = ggml_get_tensor(meta, name);
            const size_t nbytes = ggml_nbytes(t);
            const size_t slice = nbytes / g_pager.n_expert;
            const off_t base = (off_t) (data_off + gguf_get_tensor_offset(g, ti));
            for (int e = 0; e < g_pager.n_expert; e++) {
                g_pager.slices[l][e].push_back({base + (off_t) (slice * e), slice});
            }
            layer_expert_bytes += slice;
        }
        if (per_expert_total == 0 && layer_expert_bytes > 0) {
            per_expert_total = layer_expert_bytes;
        }
        // router weights (+ optional bias) for gate-ahead
        char rname[128];
        snprintf(rname, sizeof(rname), "blk.%d.ffn_gate_inp.weight", l);
        if (gguf_find_tensor(g, rname) >= 0) {
            ggml_tensor * rt = ggml_get_tensor(meta, rname);
            const size_t n = ggml_nelements(rt);
            std::vector<float> w(n);
            const off_t roff = (off_t) (data_off + gguf_get_tensor_offset(g, gguf_find_tensor(g, rname)));
            if (rt->type == GGML_TYPE_F32) {
                if (pread(g_pager.fd, w.data(), n * 4, roff) != (ssize_t) (n * 4)) w.clear();
            } else if (rt->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(n);
                if (pread(g_pager.fd, tmp.data(), n * 2, roff) == (ssize_t) (n * 2)) {
                    for (size_t i = 0; i < n; i++) w[i] = ggml_fp16_to_fp32(tmp[i]);
                } else w.clear();
            } else w.clear();
            g_pager.rw[l] = std::move(w);
        }
        snprintf(rname, sizeof(rname), "blk.%d.exp_probs_b.bias", l);
        const int64_t bi = gguf_find_tensor(g, rname);
        if (bi >= 0) {
            g_pager.sigmoid_bias = true;
            ggml_tensor * bt = ggml_get_tensor(meta, rname);
            const size_t n = ggml_nelements(bt);
            std::vector<float> b(n);
            const off_t boff = (off_t) (data_off + gguf_get_tensor_offset(g, bi));
            if (bt->type == GGML_TYPE_F32 && pread(g_pager.fd, b.data(), n * 4, boff) == (ssize_t) (n * 4)) {
                g_pager.rb[l] = std::move(b);
            }
        }
    }
    g_pager.slice_bytes_per_expert = per_expert_total;
    g_pager.score.assign((size_t) g_pager.n_layer * g_pager.n_expert, 0.f);
    g_pager.resident.assign((size_t) g_pager.n_layer * g_pager.n_expert, 0);
    g_pager.budget = budget_bytes;

    int with_router = 0;
    for (auto &w : g_pager.rw) with_router += !w.empty();
    LOG_INF("pager: %d layers x %d experts, %.1f MiB/expert, budget %.1f GiB "
            "(%zu experts), routers loaded %d/%d, sigmoid_bias=%d\n",
            g_pager.n_layer, g_pager.n_expert, per_expert_total / 1048576.0,
            budget_bytes / (double) (1 << 30), budget_bytes / per_expert_total,
            with_router, g_pager.n_layer, (int) g_pager.sigmoid_bias);

    gguf_free(g);
    ggml_free(meta);
    return per_expert_total > 0;
}

int main(int argc, char ** argv) {
    // strip our custom flags before llama arg parsing
    double budget_gib = 12.0;
    bool stats = false;
    std::vector<char *> args;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--budget-gib") { budget_gib = atof(argv[++i]); }
        else if (a == "--delta") { g_delta = atoi(argv[++i]); }
        else if (a == "--fetch") { g_fetch = atoi(argv[++i]); }
        else if (a == "--no-pager") { g_pager_on = false; }
        else if (a == "--workers") { g_workers = atoi(argv[++i]); }
        else if (a == "--stats") { stats = true; }
        else args.push_back(argv[i]);
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    if (g_pager_on) {
        if (!load_expert_map(params.model.path.c_str(), (size_t) (budget_gib * (1 << 30)))) {
            return 1;
        }
        for (int i = 0; i < g_workers; i++) {
            g_pager.workers.emplace_back([] { g_pager.worker(); });
        }
        params.cb_eval = cb_pager;
        params.warmup = false;
    }

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    if (!model || !lctx) { LOG_ERR("init failed\n"); return 1; }
    if (g_pager_on) {
        g_pager.model_path = params.model.path;
        g_pager.init_mappings();
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> toks =
        common_tokenize(lctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    LOG_INF("prefill %zu tokens\n", toks.size());

    const double t_p0 = ggml_time_us() / 1e6;
    if (llama_decode(lctx, llama_batch_get_one(toks.data(), (int32_t) toks.size()))) {
        LOG_ERR("prefill failed\n");
        return 1;
    }
    const double t_p1 = ggml_time_us() / 1e6;

    auto sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    std::string text;
    const int n_gen = params.n_predict > 0 ? params.n_predict : 64;
    int gen = 0;
    const double t_g0 = ggml_time_us() / 1e6;
    for (; gen < n_gen; gen++) {
        llama_token tok = llama_sampler_sample(smpl, lctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;
        text += common_token_to_piece(lctx, tok);
        if (llama_decode(lctx, llama_batch_get_one(&tok, 1))) break;
    }
    const double t_g1 = ggml_time_us() / 1e6;

    g_pager.stop = true;
    g_pager.cv.notify_all();
    for (auto &w : g_pager.workers) w.join();

    fprintf(stderr, "\nACHILLES prefill: %zu tok in %.1fs = %.2f tok/s\n",
            toks.size(), t_p1 - t_p0, toks.size() / (t_p1 - t_p0));
    fprintf(stderr, "ACHILLES decode:  %d tok in %.1fs = %.3f tok/s\n",
            gen, t_g1 - t_g0, gen / (t_g1 - t_g0));
    if (g_pager_on && stats) {
        const uint64_t hit = g_pager.n_demand_hit, miss = g_pager.n_demand_miss;
        fprintf(stderr, "ACHILLES pager: prefetch=%" PRIu64 " evict=%" PRIu64
                " demand_hit=%" PRIu64 " demand_miss=%" PRIu64 " (hit rate %.3f)\n",
                g_pager.n_prefetch.load(), g_pager.n_evict.load(), hit, miss,
                hit + miss ? (double) hit / (hit + miss) : 0.0);
    }
    fprintf(stderr, "OUTPUT: %.200s\n", text.c_str());
    llama_sampler_free(smpl);
    return 0;
}
