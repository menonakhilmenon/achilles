// achilles-arena: Phase 2 v2 spike — owned expert memory instead of page-cache
// steering.
//
// After llama.cpp mmaps the GGUF, we replace each expert tensor's page-interior
// address range with anonymous memory (mmap MAP_FIXED, PROT_READ|WRITE). Tensor
// data pointers are unchanged and llama.cpp is none the wiser, but expert bytes
// are now ours:
//   - LOAD:  pread() the expert's slice from the GGUF into its own addresses
//            (synchronously on a demand miss — the topk callback fires before
//            the expert matmul, so validity is guaranteed; asynchronously from
//            worker threads for gate-ahead prefetch). Completion is knowable
//            because we performed the read.
//   - EVICT: madvise(MADV_DONTNEED) on anonymous memory — always works, no
//            mapcount fights with the kernel.
//   - Tensor-edge pages that share a page with neighbouring tensors stay
//     file-backed and are served by kernel paging as before; loads/evicts are
//     trimmed to the anonymous interior.
//
// Correctness invariant: an expert's bytes are only read by the matmul after
// on_topk() has either found it valid or synchronously loaded it. Experts used
// in the current or previous forward pass are never evicted.
//
// Usage: achilles-arena -m model.gguf -p "..." -n 64 -t 10 -ngl 99 -ot exps=CPU \
//        --budget-gib 15 --delta 3 --fetch 10 --workers 4 [--no-pager] [--stats]
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

struct load_range {          // one contiguous piece of an expert, trimmed to arena
    uint8_t * addr;
    off_t     foff;
    size_t    len;
};

struct file_map { uintptr_t start, end; size_t foff; };

struct arena {
    int fd = -1;
    int n_layer = 0, n_expert = 0, n_embd = 0;
    bool sigmoid_bias = false;
    std::string model_path;
    std::vector<file_map> maps;

    // per (layer, expert): load ranges (up to 3 projections), page-trimmed
    std::vector<std::vector<std::vector<load_range>>> ranges;
    // per (layer, tensor): raw slice info kept for size accounting
    size_t bytes_per_expert = 0;
    std::vector<std::vector<float>> rw, rb;    // router weights/bias for gate-ahead

    // state
    std::vector<float> score;
    std::vector<uint8_t> valid, inflight;
    std::vector<uint8_t> pageable;             // per layer: arena installed?
    std::vector<uint32_t> last_pass;           // forward-pass id of last use
    uint32_t pass_id = 0;
    int cur_layer = 0;
    size_t budget = 0, resident_bytes = 0;
    std::mutex mu;
    std::condition_variable cv;

    struct req { int layer, expert, prio; };
    std::deque<req> q;
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;

    std::atomic<uint64_t> n_prefetch{0}, n_evict{0}, n_hit{0}, n_miss{0}, n_sync_load{0};

    int idx(int l, int e) const { return l * n_expert + e; }

    void do_load(int l, int e) { // no lock held; writes are idempotent
        for (const auto &r : ranges[l][e]) {
            if (!r.addr || r.len == 0) continue;
            size_t done = 0;
            while (done < r.len) {
                ssize_t n = pread(fd, r.addr + done, r.len - done, r.foff + (off_t) done);
                if (n <= 0) { LOG_ERR("pread failed l=%d e=%d\n", l, e); abort(); }
                done += (size_t) n;
            }
        }
    }

    void drop(int l, int e) { // no lock held
        const long pg = sysconf(_SC_PAGESIZE);
        for (const auto &r : ranges[l][e]) {
            uintptr_t lo = ((uintptr_t) r.addr + pg - 1) & ~(uintptr_t)(pg - 1);
            uintptr_t hi = ((uintptr_t) r.addr + r.len) & ~(uintptr_t)(pg - 1);
            if (hi > lo) madvise((void *) lo, hi - lo, MADV_DONTNEED);
        }
    }

    // returns list of victims to drop (caller drops without lock)
    std::vector<std::pair<int,int>> evict_to_budget_locked() {
        std::vector<std::pair<int,int>> victims;
        while (resident_bytes > budget) {
            float best = 1e30f; int bl = -1, be = -1;
            for (int l = 0; l < n_layer; l++) {
                for (int e = 0; e < n_expert; e++) {
                    const int i = idx(l, e);
                    if (!valid[i] || inflight[i]) continue;
                    // pin only the active window: used this pass AND at/after
                    // the previous layer (prefill must stream, not accumulate)
                    if (last_pass[i] == pass_id && l >= cur_layer - 1) continue;
                    if (score[i] < best) { best = score[i]; bl = l; be = e; }
                }
            }
            if (bl < 0) break; // nothing evictable
            valid[idx(bl, be)] = 0;
            resident_bytes -= bytes_per_expert;
            victims.push_back({bl, be});
            n_evict++;
        }
        return victims;
    }

    void worker() {
        while (!stop) {
            req r;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop.load() || !q.empty(); });
                if (stop) return;
                std::pop_heap(q.begin(), q.end(), [](const req &a, const req &b) { return a.prio > b.prio; });
                r = q.back();
                q.pop_back();
                const int i = idx(r.layer, r.expert);
                if (valid[i] || inflight[i]) continue;
                inflight[i] = 1;
            }
            do_load(r.layer, r.expert);
            std::vector<std::pair<int,int>> victims;
            {
                std::lock_guard<std::mutex> lk(mu);
                const int i = idx(r.layer, r.expert);
                inflight[i] = 0;
                valid[i] = 1;
                score[i] = std::max(score[i], 1.5f);
                resident_bytes += bytes_per_expert;
                victims = evict_to_budget_locked();
            }
            for (auto [l, e] : victims) drop(l, e);
            n_prefetch++;
        }
    }

    void enqueue(int l, int e, int prio) {
        std::lock_guard<std::mutex> lk(mu);
        if (!pageable[l]) return;
        const int i = idx(l, e);
        if (valid[i] || inflight[i]) return;
        q.push_back({l, e, prio});
        std::push_heap(q.begin(), q.end(), [](const req &a, const req &b) { return a.prio > b.prio; });
        cv.notify_one();
    }

    // the guarantee: called before layer l's expert matmul executes
    void on_topk(int l, const int32_t *ids, int k, int n_tokens) {
        if (l < 0 || l >= n_layer || !pageable[l]) return; // kernel-paged layer
        std::vector<int> need;
        std::vector<std::pair<int,int>> victims;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (l <= 1) pass_id++; // first MoE layer of a forward pass
            if (l <= 1) for (auto &s : score) s *= 0.98f;
            cur_layer = l;
            for (int t = 0; t < n_tokens; t++) {
                for (int i2 = 0; i2 < k; i2++) {
                    int e = ids[t * k + i2];
                    if (e < 0 || e >= n_expert) continue;
                    const int i = idx(l, e);
                    score[i] += 1.0f;
                    last_pass[i] = pass_id;
                    if (valid[i]) { n_hit++; continue; }
                    n_miss++;
                    if (!inflight[i] &&
                        std::find(need.begin(), need.end(), e) == need.end()) {
                        need.push_back(e);
                        // demand loads go to the worker pool at top priority so
                        // a layer's misses are read in parallel, not serially
                        q.push_back({l, e, 0});
                        std::push_heap(q.begin(), q.end(),
                                       [](const req &a, const req &b) { return a.prio > b.prio; });
                    }
                }
            }
            if (!need.empty()) {
                n_sync_load += need.size();
                cv.notify_all();
            }
        }
        // wait until every expert this layer needs is loaded
        (void) victims;
        if (!pageable[l]) return;
        for (;;) {
            bool pending = false;
            {
                std::lock_guard<std::mutex> lk(mu);
                for (int t = 0; t < n_tokens && !pending; t++) {
                    for (int i2 = 0; i2 < k; i2++) {
                        int e = ids[t * k + i2];
                        if (e >= 0 && e < n_expert && !valid[idx(l, e)]) { pending = true; break; }
                    }
                }
            }
            if (!pending) break;
            std::this_thread::yield();
        }
    }

    void on_hidden(int l, const float *h, int delta, int fetch) {
        if (fetch <= 0) return;
        const int d_lo = (l <= 1) ? 1 : delta;
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
            for (int i = 0; i < fetch; i++) enqueue(l + d, sc[i].second, d);
        }
    }

    void init_mappings() {
        FILE * f = fopen("/proc/self/maps", "r");
        if (!f) return;
        char line[512];
        while (fgets(line, sizeof(line), f)) {
            if (!strstr(line, model_path.c_str())) continue;
            uintptr_t s, e; size_t off;
            if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR " %*4s %zx", &s, &e, &off) == 3) {
                maps.push_back({s, e, off});
            }
        }
        fclose(f);
    }

    uint8_t * addr_of(off_t foff) const {
        for (const auto &m : maps) {
            if ((size_t) foff >= m.foff && (size_t) foff < m.foff + (m.end - m.start)) {
                return (uint8_t *) (m.start + ((size_t) foff - m.foff));
            }
        }
        return nullptr;
    }
};

static arena g_ar;
static bool g_on = true;
static int g_delta = 3, g_fetch = 10, g_workers = 4;

static bool cb_arena(struct ggml_tensor * t, bool ask, void *) {
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const bool is_norm = strncmp(t->name, "ffn_norm-", 9) == 0 ||
                         strncmp(t->name, "post_attn_norm-", 15) == 0;
    if (ask) return g_on && (is_topk || (is_norm && t->ne[1] == 1));
    if (!g_on) return true;
    if (is_topk) {
        const int l = atoi(t->name + 13);
        const int k = (int) t->ne[0], n_tokens = (int) t->ne[1];
        static std::vector<int32_t> ids;
        ids.resize((size_t) k * n_tokens);
        for (int j = 0; j < n_tokens; j++) {
            ggml_backend_tensor_get(t, ids.data() + (size_t) j * k, j * t->nb[1], k * sizeof(int32_t));
        }
        g_ar.on_topk(l, ids.data(), k, n_tokens);
    } else if (is_norm && t->ne[1] == 1 && t->type == GGML_TYPE_F32) {
        const int l = atoi(strchr(t->name, '-') + 1);
        static std::vector<float> h;
        h.resize(t->ne[0]);
        ggml_backend_tensor_get(t, h.data(), 0, t->ne[0] * sizeof(float));
        g_ar.on_hidden(l, h.data(), g_delta, g_fetch);
    }
    return true;
}

// parse GGUF: expert slice file ranges + router weights
static bool load_meta(const char * path) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { true, &meta };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (!g) return false;
    const std::string arch = gguf_get_val_str(g, gguf_find_key(g, "general.architecture"));
    auto kv = [&](const char * s) {
        const int i = gguf_find_key(g, (arch + "." + s).c_str());
        return i < 0 ? 0 : (int) gguf_get_val_u32(g, i);
    };
    g_ar.n_layer = kv("block_count");
    g_ar.n_expert = kv("expert_count");
    g_ar.n_embd = kv("embedding_length");
    const size_t data_off = gguf_get_data_offset(g);

    g_ar.ranges.assign(g_ar.n_layer, {});
    for (auto &v : g_ar.ranges) v.assign(g_ar.n_expert, {});
    g_ar.rw.assign(g_ar.n_layer, {});
    g_ar.rb.assign(g_ar.n_layer, {});
    g_ar.fd = open(path, O_RDONLY);

    const char * projs[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    size_t per_expert = 0;
    for (int l = 0; l < g_ar.n_layer; l++) {
        size_t layer_bytes = 0;
        for (const char * pj : projs) {
            char name[128];
            snprintf(name, sizeof(name), "blk.%d.%s.weight", l, pj);
            const int64_t ti = gguf_find_tensor(g, name);
            if (ti < 0) continue;
            ggml_tensor * t = ggml_get_tensor(meta, name);
            const size_t nbytes = ggml_nbytes(t);
            const size_t slice = nbytes / g_ar.n_expert;
            const off_t base = (off_t) (data_off + gguf_get_tensor_offset(g, ti));
            for (int e = 0; e < g_ar.n_expert; e++) {
                // store raw range now; trimmed to arena + addr-resolved later
                g_ar.ranges[l][e].push_back({nullptr, base + (off_t) (slice * e), slice});
            }
            layer_bytes += slice;
        }
        if (per_expert == 0 && layer_bytes > 0) per_expert = layer_bytes;

        char rname[128];
        snprintf(rname, sizeof(rname), "blk.%d.ffn_gate_inp.weight", l);
        const int64_t ri = gguf_find_tensor(g, rname);
        if (ri >= 0) {
            ggml_tensor * rt = ggml_get_tensor(meta, rname);
            const size_t n = ggml_nelements(rt);
            std::vector<float> w(n);
            const off_t roff = (off_t) (data_off + gguf_get_tensor_offset(g, ri));
            bool ok = false;
            if (rt->type == GGML_TYPE_F32) {
                ok = pread(g_ar.fd, w.data(), n * 4, roff) == (ssize_t) (n * 4);
            } else if (rt->type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> tmp(n);
                ok = pread(g_ar.fd, tmp.data(), n * 2, roff) == (ssize_t) (n * 2);
                if (ok) for (size_t i = 0; i < n; i++) w[i] = ggml_fp16_to_fp32(tmp[i]);
            }
            if (ok) g_ar.rw[l] = std::move(w);
        }
        snprintf(rname, sizeof(rname), "blk.%d.exp_probs_b.bias", l);
        const int64_t bi = gguf_find_tensor(g, rname);
        if (bi >= 0) {
            g_ar.sigmoid_bias = true;
            ggml_tensor * bt = ggml_get_tensor(meta, rname);
            const size_t n = ggml_nelements(bt);
            std::vector<float> b(n);
            if (bt->type == GGML_TYPE_F32 &&
                pread(g_ar.fd, b.data(), n * 4,
                      (off_t) (data_off + gguf_get_tensor_offset(g, bi))) == (ssize_t) (n * 4)) {
                g_ar.rb[l] = std::move(b);
            }
        }
    }
    g_ar.bytes_per_expert = per_expert;
    const size_t N = (size_t) g_ar.n_layer * g_ar.n_expert;
    g_ar.score.assign(N, 0.f);
    g_ar.valid.assign(N, 0);
    g_ar.inflight.assign(N, 0);
    g_ar.last_pass.assign(N, 0);
    gguf_free(g);
    ggml_free(meta);
    return per_expert > 0;
}

// replace expert tensors' page interiors with anonymous memory; trim ranges
static bool install_arena() {
    g_ar.init_mappings();
    if (g_ar.maps.empty()) { LOG_ERR("arena: model mapping not found\n"); return false; }
    const long pg = sysconf(_SC_PAGESIZE);
    size_t replaced = 0;
    int tensors = 0;
    g_ar.pageable.assign(g_ar.n_layer, 0);

    // group per-tensor: contiguous run of expert slices of one projection
    for (int l = 0; l < g_ar.n_layer; l++) {
        if (g_ar.ranges[l].empty() || g_ar.ranges[l][0].empty()) continue;
        const size_t nproj = g_ar.ranges[l][0].size();
        bool layer_ok = true;
        for (size_t p = 0; p < nproj; p++) {
            const off_t t_lo = g_ar.ranges[l][0][p].foff;
            const off_t t_hi = g_ar.ranges[l][g_ar.n_expert - 1][p].foff
                             + (off_t) g_ar.ranges[l][g_ar.n_expert - 1][p].len;
            uint8_t * base = g_ar.addr_of(t_lo);
            if (!base || g_ar.addr_of(t_hi - 1) != base + (t_hi - 1 - t_lo)) {
                LOG_ERR("arena: tensor spans mappings, layer %d stays kernel-paged\n", l);
                layer_ok = false;
                continue;
            }
            const uintptr_t a_lo = (((uintptr_t) base) + pg - 1) & ~(uintptr_t)(pg - 1);
            const uintptr_t a_hi = (((uintptr_t) base) + (t_hi - t_lo)) & ~(uintptr_t)(pg - 1);
            if (a_hi <= a_lo) continue;
            void * r = mmap((void *) a_lo, a_hi - a_lo, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
            if (r == MAP_FAILED) { LOG_ERR("arena: MAP_FIXED failed\n"); return false; }
            replaced += a_hi - a_lo;
            tensors++;
            // trim each expert's range for this projection to the anon interior
            const off_t anon_lo_foff = t_lo + ((off_t) a_lo - (off_t)(uintptr_t) base);
            const off_t anon_hi_foff = anon_lo_foff + (off_t) (a_hi - a_lo);
            for (int e = 0; e < g_ar.n_expert; e++) {
                auto &rg = g_ar.ranges[l][e][p];
                const off_t lo = std::max(rg.foff, anon_lo_foff);
                const off_t hi = std::min((off_t) (rg.foff + rg.len), anon_hi_foff);
                if (hi <= lo) { rg.len = 0; continue; }
                rg.addr = (uint8_t *) (a_lo + (lo - anon_lo_foff));
                rg.foff = lo;
                rg.len = (size_t) (hi - lo);
            }
        }
        g_ar.pageable[l] = layer_ok;
    }
    LOG_INF("arena: replaced %.1f GiB across %d expert tensors (%d x %d experts, "
            "%.1f MiB each), budget %.1f GiB\n",
            replaced / (double) (1 << 30), tensors, g_ar.n_layer, g_ar.n_expert,
            g_ar.bytes_per_expert / 1048576.0, g_ar.budget / (double) (1 << 30));
    return true;
}

int main(int argc, char ** argv) {
    double budget_gib = 15.0;
    bool stats = false;
    std::vector<char *> args;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--budget-gib") budget_gib = atof(argv[++i]);
        else if (a == "--delta") g_delta = atoi(argv[++i]);
        else if (a == "--fetch") g_fetch = atoi(argv[++i]);
        else if (a == "--workers") g_workers = atoi(argv[++i]);
        else if (a == "--no-pager") g_on = false;
        else if (a == "--stats") stats = true;
        else args.push_back(argv[i]);
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    llama_backend_init();
    llama_numa_init(params.numa);

    if (g_on) {
        if (!load_meta(params.model.path.c_str())) return 1;
        g_ar.budget = (size_t) (budget_gib * (1 << 30));
        params.cb_eval = cb_arena;
        params.warmup = false;
    }

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    if (!model || !lctx) return 1;

    if (g_on) {
        g_ar.model_path = params.model.path;
        if (!install_arena()) return 1;
        for (int i = 0; i < g_workers; i++) g_ar.workers.emplace_back([] { g_ar.worker(); });
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks =
        common_tokenize(lctx, params.prompt, llama_vocab_get_add_bos(vocab), true);

    const double t_p0 = ggml_time_us() / 1e6;
    if (llama_decode(lctx, llama_batch_get_one(toks.data(), (int32_t) toks.size()))) return 1;
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

    g_ar.stop = true;
    g_ar.cv.notify_all();
    for (auto &w : g_ar.workers) w.join();

    fprintf(stderr, "\nACHILLES prefill: %zu tok in %.1fs = %.2f tok/s\n",
            toks.size(), t_p1 - t_p0, toks.size() / (t_p1 - t_p0));
    fprintf(stderr, "ACHILLES decode:  %d tok in %.1fs = %.3f tok/s\n",
            gen, t_g1 - t_g0, gen / (t_g1 - t_g0));
    if (g_on && stats) {
        const uint64_t h = g_ar.n_hit, m = g_ar.n_miss;
        fprintf(stderr, "ACHILLES arena: prefetch=%" PRIu64 " sync_load=%" PRIu64
                " evict=%" PRIu64 " hit=%" PRIu64 " miss=%" PRIu64 " (hit rate %.3f)\n",
                g_ar.n_prefetch.load(), g_ar.n_sync_load.load(), g_ar.n_evict.load(),
                h, m, h + m ? (double) h / (h + m) : 0.0);
    }
    fprintf(stderr, "OUTPUT: %.240s\n", text.c_str());
    llama_sampler_free(smpl);
    return 0;
}
