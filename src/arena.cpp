// achilles-arena: Phase 2 v2 — owned expert memory instead of page-cache
// steering. Multi-shard GGUF capable (GLM-5.2's 7 shards).
//
// After llama.cpp mmaps the GGUF shard(s), we replace each expert tensor's
// page-interior address range with anonymous memory (mmap MAP_FIXED,
// PROT_READ|WRITE). Tensor data pointers are unchanged and llama.cpp is none
// the wiser, but expert bytes are now ours:
//   - LOAD:  pread() the expert's slice from its shard into its own addresses.
//            Demand misses scatter across the worker pool (the topk callback
//            fires before the expert matmul, so we block until valid);
//            gate-ahead prefetch uses the same pool at lower priority.
//   - EVICT: madvise(MADV_DONTNEED) on anonymous memory (always works).
//   - PREDICT: gate-ahead scoring runs on a dedicated thread (256x6144 at
//            GLM-5.2 scale is too costly for the compute thread).
//
// Correctness invariant: an expert's bytes are only read by the matmul after
// on_topk() found it valid or its demand load completed. Experts used in the
// active layer window are never evicted (prefill streams within budget).
//
// Usage: achilles-arena -m model-00001-of-00007.gguf -p "..." -n 64 -t 10 \
//        -ngl 99 -ot exps=CPU --budget-gib 38 --delta 3 --fetch 10 \
//        --workers 6 [--no-pager] [--stats]
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <fcntl.h>
#include <liburing.h>
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

struct load_range {           // contiguous piece of an expert, trimmed to arena
    uint8_t * addr = nullptr;
    int       file = 0;
    off_t     foff = 0;
    size_t    len  = 0;
};

struct file_map { int file; uintptr_t start, end; size_t foff; };

struct arena {
    std::vector<std::string> paths;
    std::vector<int> fds;         // buffered
    std::vector<int> fds_direct;  // O_DIRECT (io_uring path); -1 if unavailable
    std::vector<file_map> maps;
    int n_layer = 0, n_expert = 0, n_embd = 0;
    bool sigmoid_bias = false;

    std::vector<std::vector<std::vector<load_range>>> ranges; // [layer][expert][proj]
    std::vector<size_t> bytes_of;              // per-layer expert size (UD quants vary)
    std::vector<std::vector<float>> rw, rb;

    // state (mu-protected)
    std::vector<float> score;
    std::vector<uint8_t> valid, inflight, pageable;
    std::vector<uint32_t> last_pass;
    uint32_t pass_id = 0;
    int cur_layer = 0;
    size_t budget = 0, resident_bytes = 0;
    std::mutex mu;
    std::condition_variable cv;

    struct req { int layer, expert, prio; };
    std::deque<req> q;
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;

    // gate-ahead scoring runs off the compute thread
    std::mutex smu;
    std::condition_variable scv;
    std::deque<std::pair<int, std::vector<float>>> sq;
    std::thread scorer;
    int delta = 3, fetch = 10;
    float decay = 0.98f;
    bool probe_mode = false;  // rw holds trained probes for (l -> l+delta); score only d==delta
    int probe_layers = 0;
    std::vector<float> probe_data;

    std::atomic<uint64_t> n_prefetch{0}, n_evict{0}, n_hit{0}, n_miss{0}, n_demand{0}, n_uring{0}, n_fallback{0};

    int idx(int l, int e) const { return l * n_expert + e; }

    void load_buffered(const load_range &r) {
        size_t done = 0;
        while (done < r.len) {
            ssize_t n = pread(fds[r.file], r.addr + done, r.len - done, r.foff + (off_t) done);
            if (n <= 0) { LOG_ERR("pread failed\n"); abort(); }
            done += (size_t) n;
        }
        // buffered read leaves a second copy in the page cache; drop it
        posix_fadvise(fds[r.file], r.foff, r.len, POSIX_FADV_DONTNEED);
    }

    // O_DIRECT via the worker's private ring. Arena addresses are congruent
    // with file offsets mod 4096, so aligning the file range down/up keeps the
    // buffer aligned too; overshoot writes neighbouring experts' bytes with
    // identical data (benign) and never leaves the anonymous region.
    void do_load(int l, int e, io_uring * ring) {
        const auto &rs = ranges[l][e];
        int submitted = 0;
        for (const auto &r : rs) {
            if (!r.addr || r.len == 0) continue;
            if (!ring || fds_direct[r.file] < 0) { load_buffered(r); continue; }
            const off_t  a_off = r.foff & ~(off_t) 4095;
            const size_t head  = (size_t) (r.foff - a_off);
            const size_t a_len = (r.len + head + 4095) & ~(size_t) 4095;
            io_uring_sqe * sqe = io_uring_get_sqe(ring);
            if (!sqe) { load_buffered(r); continue; }
            io_uring_prep_read(sqe, fds_direct[r.file], r.addr - head, (unsigned) a_len, a_off);
            io_uring_sqe_set_data(sqe, (void *) &r);
            submitted++;
        }
        if (!submitted) return;
        io_uring_submit_and_wait(ring, submitted);
        for (int i = 0; i < submitted; i++) {
            io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe(ring, &cqe) < 0) { LOG_ERR("cqe wait failed\n"); abort(); }
            if (cqe->res < 0) { // e.g. compressed extent rejecting O_DIRECT
                load_buffered(*(const load_range *) io_uring_cqe_get_data(cqe));
                n_fallback++;
            } else {
                n_uring++;
            }
            io_uring_cqe_seen(ring, cqe);
        }
    }

    void drop(int l, int e) {
        const long pg = sysconf(_SC_PAGESIZE);
        for (const auto &r : ranges[l][e]) {
            if (!r.addr || r.len == 0) continue;
            uintptr_t lo = ((uintptr_t) r.addr + pg - 1) & ~(uintptr_t)(pg - 1);
            uintptr_t hi = ((uintptr_t) r.addr + r.len) & ~(uintptr_t)(pg - 1);
            if (hi > lo) madvise((void *) lo, hi - lo, MADV_DONTNEED);
        }
    }

    std::vector<std::pair<int,int>> evict_to_budget_locked() {
        std::vector<std::pair<int,int>> victims;
        while (resident_bytes > budget) {
            float best = 1e30f; int bl = -1, be = -1;
            for (int l = 0; l < n_layer; l++) {
                if (!pageable[l]) continue;
                for (int e = 0; e < n_expert; e++) {
                    const int i = idx(l, e);
                    if (!valid[i] || inflight[i]) continue;
                    if (last_pass[i] == pass_id && l >= cur_layer - 1) continue;
                    if (score[i] < best) { best = score[i]; bl = l; be = e; }
                }
            }
            if (bl < 0) break;
            valid[idx(bl, be)] = 0;
            resident_bytes -= bytes_of[bl];
            victims.push_back({bl, be});
            n_evict++;
        }
        return victims;
    }

    void worker() {
        io_uring ring;
        io_uring * pring = io_uring_queue_init(16, &ring, 0) == 0 ? &ring : nullptr;
        while (!stop) {
            req r;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop.load() || !q.empty(); });
                if (stop) break;
                std::pop_heap(q.begin(), q.end(), [](const req &a, const req &b) { return a.prio > b.prio; });
                r = q.back();
                q.pop_back();
                const int i = idx(r.layer, r.expert);
                if (valid[i] || inflight[i]) continue;
                inflight[i] = 1;
            }
            do_load(r.layer, r.expert, pring);
            std::vector<std::pair<int,int>> victims;
            {
                std::lock_guard<std::mutex> lk(mu);
                const int i = idx(r.layer, r.expert);
                inflight[i] = 0;
                valid[i] = 1;
                score[i] = std::max(score[i], 1.5f);
                resident_bytes += bytes_of[r.layer];
                victims = evict_to_budget_locked();
            }
            for (auto [l, e] : victims) drop(l, e);
            if (r.prio == 0) n_demand++; else n_prefetch++;
        }
        if (pring) io_uring_queue_exit(pring);
    }

    void enqueue(int l, int e, int prio) {
        std::lock_guard<std::mutex> lk(mu);
        if (!pageable[l]) return;
        if (prio > 0 && q.size() >= 96) return; // never starve the demand path
        const int i = idx(l, e);
        if (valid[i] || inflight[i]) return;
        for (const auto &r : q) if (r.layer == l && r.expert == e) return;
        q.push_back({l, e, prio});
        std::push_heap(q.begin(), q.end(), [](const req &a, const req &b) { return a.prio > b.prio; });
        cv.notify_one();
    }

    void on_topk(int l, const int32_t *ids, int k, int n_tokens) {
        if (l < 0 || l >= n_layer || !pageable[l]) return;
        bool queued = false;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (l <= cur_layer) { pass_id++; if (decay < 1.0f) for (auto &s : score) s *= decay; }
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
                    if (!inflight[i]) {
                        bool dup = false;
                        for (const auto &r : q) if (r.layer == l && r.expert == e) { dup = true; break; }
                        if (!dup) {
                            q.push_back({l, e, 0});
                            std::push_heap(q.begin(), q.end(),
                                           [](const req &a, const req &b) { return a.prio > b.prio; });
                            queued = true;
                        }
                    }
                }
            }
        }
        if (queued) cv.notify_all();
        // block until every expert this layer needs is loaded
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

    // ---- gate-ahead prediction ----
    // Inline on the compute thread by default: with -O3 -march=native the dot
    // products vectorize (~0.1ms at Air scale, ~0.5ms/layer at GLM-5.2 scale).
    // Async scoring (submit_hidden -> score_loop) measured 28% SLOWER end to
    // end: prediction latency turns timely prefetches into demand stalls.
    void submit_hidden(int l, const float *h, int n) {
        (void) n;
        score_hidden(l, h);
    }

    void score_loop() {
        while (!stop) {
            std::pair<int, std::vector<float>> job;
            {
                std::unique_lock<std::mutex> lk(smu);
                scv.wait(lk, [&] { return stop.load() || !sq.empty(); });
                if (stop) return;
                job = std::move(sq.front());
                sq.pop_front();
            }
            score_hidden(job.first, job.second.data());
        }
    }

    void score_hidden(int l, const float *h) {
        {
            const int d_lo = probe_mode ? delta : ((l <= 1) ? 1 : delta);
            for (int d = d_lo; d <= delta && l + d < n_layer; d++) {
                if (!pageable[l + d]) continue;
                const auto &W = probe_mode ? rw[l] : rw[l + d];
                if (W.empty()) continue;
                std::vector<std::pair<float, int>> sc(n_expert);
                for (int e = 0; e < n_expert; e++) {
                    float dot = 0;
                    const float *w = &W[(size_t) e * n_embd];
                    for (int j = 0; j < n_embd; j++) dot += w[j] * h[j];
                    if (!probe_mode && sigmoid_bias) {
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
    }

    // ---- shard-aware file plumbing ----
    void init_mappings() {
        FILE * f = fopen("/proc/self/maps", "r");
        if (!f) return;
        char line[1024];
        while (fgets(line, sizeof(line), f)) {
            for (size_t fi = 0; fi < paths.size(); fi++) {
                if (!strstr(line, paths[fi].c_str())) continue;
                uintptr_t s, e; size_t off;
                if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR " %*4s %zx", &s, &e, &off) == 3) {
                    maps.push_back({(int) fi, s, e, off});
                }
            }
        }
        fclose(f);
    }

    uint8_t * addr_of(int file, off_t foff) const {
        for (const auto &m : maps) {
            if (m.file == file && (size_t) foff >= m.foff &&
                (size_t) foff < m.foff + (m.end - m.start)) {
                return (uint8_t *) (m.start + ((size_t) foff - m.foff));
            }
        }
        return nullptr;
    }
};

static arena g_ar;
static bool g_on = true;
static FILE * g_dump = nullptr;  // trace dump: 'H' l n h[fp32*n] | 'T' l k ids[i32*k]
static std::mutex g_dump_mu;
static int g_workers = 6;

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
        if (g_dump && n_tokens == 1) {
            std::lock_guard<std::mutex> lk(g_dump_mu);
            fputc('T', g_dump);
            int32_t hdr[2] = {l, k};
            fwrite(hdr, 4, 2, g_dump);
            fwrite(ids.data(), 4, k, g_dump);
        }
        g_ar.on_topk(l, ids.data(), k, n_tokens);
    } else if (is_norm && t->ne[1] == 1 && t->type == GGML_TYPE_F32) {
        const int l = atoi(strchr(t->name, '-') + 1);
        static std::vector<float> h;
        h.resize(t->ne[0]);
        ggml_backend_tensor_get(t, h.data(), 0, t->ne[0] * sizeof(float));
        if (g_dump) {
            std::lock_guard<std::mutex> lk(g_dump_mu);
            fputc('H', g_dump);
            int32_t hdr[2] = {l, (int32_t) t->ne[0]};
            fwrite(hdr, 4, 2, g_dump);
            fwrite(h.data(), 4, t->ne[0], g_dump);
        }
        g_ar.submit_hidden(l, h.data(), (int) t->ne[0]);
    }
    return true;
}

// expand "-00001-of-00007.gguf" into the full shard list
static std::vector<std::string> shard_paths(const std::string &p) {
    const auto pos = p.find("-of-");
    if (pos == std::string::npos || pos < 6) return {p};
    const int count = atoi(p.substr(pos + 4, 5).c_str());
    if (count <= 1) return {p};
    std::vector<std::string> out;
    for (int i = 1; i <= count; i++) {
        char idx[8];
        snprintf(idx, sizeof(idx), "%05d", i);
        std::string s = p;
        s.replace(pos - 5, 5, idx);
        out.push_back(s);
    }
    return out;
}

static bool load_meta(const std::string &model_path) {
    g_ar.paths = shard_paths(model_path);
    LOG_INF("arena: %zu shard(s)\n", g_ar.paths.size());

    bool have_kv = false;
    for (size_t fi = 0; fi < g_ar.paths.size(); fi++) {
        ggml_context * meta = nullptr;
        gguf_init_params gp = { true, &meta };
        gguf_context * g = gguf_init_from_file(g_ar.paths[fi].c_str(), gp);
        if (!g) { LOG_ERR("arena: gguf parse failed: %s\n", g_ar.paths[fi].c_str()); return false; }
        g_ar.fds.push_back(open(g_ar.paths[fi].c_str(), O_RDONLY));
        g_ar.fds_direct.push_back(open(g_ar.paths[fi].c_str(), O_RDONLY | O_DIRECT));

        if (!have_kv) {
            const int ia = gguf_find_key(g, "general.architecture");
            if (ia >= 0) {
                const std::string arch = gguf_get_val_str(g, ia);
                auto kv = [&](const char * s) {
                    const int i = gguf_find_key(g, (arch + "." + s).c_str());
                    return i < 0 ? 0 : (int) gguf_get_val_u32(g, i);
                };
                g_ar.n_layer = kv("block_count");
                g_ar.n_expert = kv("expert_count");
                g_ar.n_embd = kv("embedding_length");
                if (g_ar.n_layer && g_ar.n_expert && g_ar.n_embd) {
                    have_kv = true;
                    g_ar.ranges.assign(g_ar.n_layer, {});
                    for (auto &v : g_ar.ranges) v.assign(g_ar.n_expert, {});
                    g_ar.bytes_of.assign(g_ar.n_layer, 0);
                    g_ar.rw.assign(g_ar.n_layer, {});
                    g_ar.rb.assign(g_ar.n_layer, {});
                }
            }
        }
        if (!have_kv) { LOG_ERR("arena: first shard lacks arch KVs\n"); return false; }

        const size_t data_off = gguf_get_data_offset(g);
        const int64_t nt = gguf_get_n_tensors(g);
        for (int64_t ti = 0; ti < nt; ti++) {
            const char * name = gguf_get_tensor_name(g, ti);
            int l; char what[64];
            if (sscanf(name, "blk.%d.%63[^.].weight", &l, what) < 2 &&
                sscanf(name, "blk.%d.%63[^.].bias", &l, what) < 2) continue;
            if (l < 0 || l >= g_ar.n_layer) continue;
            ggml_tensor * t = ggml_get_tensor(meta, name);
            const off_t base = (off_t) (data_off + gguf_get_tensor_offset(g, ti));

            if (!strcmp(what, "ffn_gate_exps") || !strcmp(what, "ffn_up_exps") ||
                !strcmp(what, "ffn_down_exps")) {
                const size_t slice = ggml_nbytes(t) / g_ar.n_expert;
                for (int e = 0; e < g_ar.n_expert; e++) {
                    g_ar.ranges[l][e].push_back({nullptr, (int) fi, base + (off_t) (slice * e), slice});
                }
                g_ar.bytes_of[l] += slice;
            } else if (!strcmp(what, "ffn_gate_inp")) {
                const size_t n = ggml_nelements(t);
                std::vector<float> w(n);
                bool ok = false;
                if (t->type == GGML_TYPE_F32) {
                    ok = pread(g_ar.fds[fi], w.data(), n * 4, base) == (ssize_t) (n * 4);
                } else if (t->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> tmp(n);
                    ok = pread(g_ar.fds[fi], tmp.data(), n * 2, base) == (ssize_t) (n * 2);
                    if (ok) for (size_t i = 0; i < n; i++) w[i] = ggml_fp16_to_fp32(tmp[i]);
                }
                if (ok) g_ar.rw[l] = std::move(w);
            } else if (!strcmp(what, "exp_probs_b")) {
                g_ar.sigmoid_bias = true;
                const size_t n = ggml_nelements(t);
                std::vector<float> b(n);
                if (t->type == GGML_TYPE_F32 &&
                    pread(g_ar.fds[fi], b.data(), n * 4, base) == (ssize_t) (n * 4)) {
                    g_ar.rb[l] = std::move(b);
                }
            }
        }
        gguf_free(g);
        ggml_free(meta);
    }
    const size_t N = (size_t) g_ar.n_layer * g_ar.n_expert;
    g_ar.score.assign(N, 0.f);
    g_ar.valid.assign(N, 0);
    g_ar.inflight.assign(N, 0);
    g_ar.last_pass.assign(N, 0);
    size_t total = 0;
    int moe_layers = 0;
    for (int l = 0; l < g_ar.n_layer; l++) {
        if (g_ar.bytes_of[l]) { total += g_ar.bytes_of[l] * g_ar.n_expert; moe_layers++; }
    }
    LOG_INF("arena: %d MoE layers x %d experts, %.1f GiB of experts total\n",
            moe_layers, g_ar.n_expert, total / (double) (1 << 30));
    return moe_layers > 0;
}

static bool install_arena() {
    g_ar.init_mappings();
    if (g_ar.maps.empty()) { LOG_ERR("arena: no model mappings found\n"); return false; }
    const long pg = sysconf(_SC_PAGESIZE);
    size_t replaced = 0;
    g_ar.pageable.assign(g_ar.n_layer, 0);

    for (int l = 0; l < g_ar.n_layer; l++) {
        if (g_ar.ranges[l].empty() || g_ar.ranges[l][0].empty()) continue;
        const size_t nproj = g_ar.ranges[l][0].size();
        bool layer_ok = true;
        for (size_t p = 0; p < nproj; p++) {
            const auto &first = g_ar.ranges[l][0][p];
            const auto &last  = g_ar.ranges[l][g_ar.n_expert - 1][p];
            const off_t t_lo = first.foff;
            const off_t t_hi = last.foff + (off_t) last.len;
            uint8_t * base = g_ar.addr_of(first.file, t_lo);
            if (!base || g_ar.addr_of(first.file, t_hi - 1) != base + (t_hi - 1 - t_lo)) {
                LOG_ERR("arena: layer %d proj %zu not mappable, stays kernel-paged\n", l, p);
                layer_ok = false;
                continue;
            }
            const uintptr_t a_lo = (((uintptr_t) base) + pg - 1) & ~(uintptr_t)(pg - 1);
            const uintptr_t a_hi = (((uintptr_t) base) + (t_hi - t_lo)) & ~(uintptr_t)(pg - 1);
            if (a_hi <= a_lo) continue;
            if (mmap((void *) a_lo, a_hi - a_lo, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0) == MAP_FAILED) {
                LOG_ERR("arena: MAP_FIXED failed l=%d\n", l);
                return false;
            }
            replaced += a_hi - a_lo;
            const off_t anon_lo = t_lo + ((off_t) a_lo - (off_t)(uintptr_t) base);
            const off_t anon_hi = anon_lo + (off_t) (a_hi - a_lo);
            for (int e = 0; e < g_ar.n_expert; e++) {
                auto &rg = g_ar.ranges[l][e][p];
                const off_t lo = std::max(rg.foff, anon_lo);
                const off_t hi = std::min((off_t) (rg.foff + rg.len), anon_hi);
                if (hi <= lo) { rg.len = 0; rg.addr = nullptr; continue; }
                rg.addr = (uint8_t *) (a_lo + (lo - anon_lo));
                rg.foff = lo;
                rg.len = (size_t) (hi - lo);
            }
        }
        g_ar.pageable[l] = layer_ok;
    }
    LOG_INF("arena: replaced %.1f GiB with anonymous expert memory, budget %.1f GiB\n",
            replaced / (double) (1 << 30), g_ar.budget / (double) (1 << 30));
    return replaced > 0;
}

int main(int argc, char ** argv) {
    double budget_gib = 15.0;
    bool stats = false;
    std::vector<char *> args;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--budget-gib") budget_gib = atof(argv[++i]);
        else if (a == "--delta") g_ar.delta = atoi(argv[++i]);
        else if (a == "--fetch") g_ar.fetch = atoi(argv[++i]);
        else if (a == "--decay") g_ar.decay = (float) atof(argv[++i]);
        else if (a == "--dump") g_dump = fopen(argv[++i], "wb");
        else if (a == "--probe") {
            FILE * pf = fopen(argv[++i], "rb");
            if (pf) {
                int32_t nl, ne, nh;
                if (fread(&nl, 4, 1, pf) == 1 && fread(&ne, 4, 1, pf) == 1 &&
                    fread(&nh, 4, 1, pf) == 1) {
                    g_ar.probe_layers = nl;
                    g_ar.probe_data.assign((size_t) nl * ne * nh, 0.f);
                    if (fread(g_ar.probe_data.data(), 4, g_ar.probe_data.size(), pf)
                        == g_ar.probe_data.size()) {
                        g_ar.probe_mode = true;
                    }
                }
                fclose(pf);
            }
        }
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
        if (!load_meta(params.model.path)) return 1;
        if (g_ar.probe_mode) {  // trained probes replace router weights for scoring
            const size_t per = (size_t) g_ar.n_expert * g_ar.n_embd;
            int applied = 0;
            for (int l = 0; l < std::min(g_ar.probe_layers, g_ar.n_layer); l++) {
                std::vector<float> w(g_ar.probe_data.begin() + l * per,
                                     g_ar.probe_data.begin() + (l + 1) * per);
                bool nonzero = false;
                for (size_t j = 0; j < per && !nonzero; j += 997) nonzero = w[j] != 0.f;
                if (nonzero) { g_ar.rw[l] = std::move(w); applied++; }
                else g_ar.rw[l].clear(); // no probe -> no gate-ahead from this layer
            }
            for (int l = g_ar.probe_layers; l < g_ar.n_layer; l++) g_ar.rw[l].clear();
            g_ar.probe_data.clear();
            LOG_INF("arena: trained probes applied to %d layers (delta=%d)\n", applied, g_ar.delta);
        }
        g_ar.budget = (size_t) (budget_gib * (1 << 30));
        params.cb_eval = cb_arena;
        params.warmup = false;
    }

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * lctx  = llama_init->context();
    if (!model || !lctx) return 1;

    if (g_on) {
        if (!install_arena()) return 1;
        for (int i = 0; i < g_workers; i++) g_ar.workers.emplace_back([] { g_ar.worker(); });
        g_ar.scorer = std::thread([] { g_ar.score_loop(); });
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
    g_ar.scv.notify_all();
    for (auto &w : g_ar.workers) w.join();
    if (g_ar.scorer.joinable()) g_ar.scorer.join();

    fprintf(stderr, "\nACHILLES prefill: %zu tok in %.1fs = %.2f tok/s\n",
            toks.size(), t_p1 - t_p0, toks.size() / (t_p1 - t_p0));
    fprintf(stderr, "ACHILLES decode:  %d tok in %.1fs = %.3f tok/s\n",
            gen, t_g1 - t_g0, gen / (t_g1 - t_g0));
    if (g_on && stats) {
        const uint64_t h = g_ar.n_hit, m = g_ar.n_miss;
        fprintf(stderr, "ACHILLES arena: prefetch=%" PRIu64 " demand=%" PRIu64
                " evict=%" PRIu64 " hit=%" PRIu64 " miss=%" PRIu64 " (hit rate %.3f)\n",
                g_ar.n_prefetch.load(), g_ar.n_demand.load(), g_ar.n_evict.load(),
                h, m, h + m ? (double) h / (h + m) : 0.0);
        fprintf(stderr, "ACHILLES io: uring_ok=%" PRIu64 " fallback=%" PRIu64 "\n",
                g_ar.n_uring.load(), g_ar.n_fallback.load());
    }
    fprintf(stderr, "OUTPUT: %.240s\n", text.c_str());
    llama_sampler_free(smpl);
    return 0;
}
