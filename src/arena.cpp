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
#include "speculative.h"
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
#include <chrono>
#include <vector>

extern bool g_no_uring;  // fwd (defined after arena)

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
    std::vector<float> pop;          // reuse policy: EWMA activation rate
    std::vector<uint32_t> pop_up;    // pass of last pop update (lazy decay)
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
    std::thread janitor;
    std::condition_variable jcv;
    std::vector<std::pair<int,int>> pending_drops;
    size_t pending_drop_bytes = 0;  // evicted-but-not-yet-madvised (real RSS!)
    int delta = 3, fetch = 2, pstream = 1;   // fetch>2 over-fetches: saturated-SSD sweep 2026-07-13
    float decay = 0.98f;
    bool lru = true;                    // Belady analysis: LRU 61% vs LFU 50% on GLM-5.2
    bool reuse = false;                 // gap/pop reuse-distance policy (sim: +2pp over LRU)
    static constexpr float POP_ALPHA = 0.02f, POP_MIN = 1e-4f;
    uint64_t touch_clock = 1000;        // starts above the 1.5 prefetch floor
    float bias_strength = 0.f;          // cache-aware routing bias (0 = off)
    bool probe_mode = false;  // rw holds trained probes for (l -> l+delta); score only d==delta
    int probe_layers = 0;
    std::vector<float> probe_data;
    int probe8_layers = 0;
    std::vector<float> probe8_data;
    std::vector<std::vector<float>> rw8;    // far-stage probes (l -> l+8)
    std::vector<uint32_t> plan_hint;        // pass id when expert was last planned

    std::atomic<uint64_t> n_prefetch{0}, n_evict{0}, n_hit{0}, n_miss{0}, n_demand{0}, n_uring{0}, n_fallback{0};
    std::atomic<uint64_t> stall_ns{0}, io_bytes{0}, io_ns{0};  // time attribution
    uint64_t snap[10] = {0};
    void snapshot() {
        snap[0] = stall_ns; snap[1] = io_bytes; snap[2] = io_ns;
        snap[3] = n_hit; snap[4] = n_miss; snap[5] = n_demand; snap[6] = n_prefetch;
    }

    int idx(int l, int e) const { return l * n_expert + e; }

    void load_buffered(const load_range &r) {
        const auto t0 = std::chrono::steady_clock::now();
        size_t done = 0;
        while (done < r.len) {
            ssize_t n = pread(fds[r.file], r.addr + done, r.len - done, r.foff + (off_t) done);
            if (n <= 0) { LOG_ERR("pread failed\n"); abort(); }
            done += (size_t) n;
        }
        // buffered read leaves a second copy in the page cache; drop it
        posix_fadvise(fds[r.file], r.foff, r.len, POSIX_FADV_DONTNEED);
        io_bytes += r.len;
        io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0).count();
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
        const auto t0 = std::chrono::steady_clock::now();
        io_uring_submit_and_wait(ring, submitted);
        for (int i = 0; i < submitted; i++) {
            io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe(ring, &cqe) < 0) { LOG_ERR("cqe wait failed\n"); abort(); }
            if (cqe->res < 0) { // e.g. compressed extent rejecting O_DIRECT
                load_buffered(*(const load_range *) io_uring_cqe_get_data(cqe));
                n_fallback++;
            } else {
                io_bytes += (size_t) cqe->res;
                n_uring++;
            }
            io_uring_cqe_seen(ring, cqe);
        }
        io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0).count();
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

    void evict_to_budget_locked() {
        auto &victims = pending_drops;
        while (resident_bytes > budget) {
            float best = 1e30f; int bl = -1, be = -1;
            for (int l = 0; l < n_layer; l++) {
                if (!pageable[l]) continue;
                for (int e = 0; e < n_expert; e++) {
                    const int i = idx(l, e);
                    if (!valid[i] || inflight[i]) continue;
                    if (last_pass[i] == pass_id && l >= cur_layer - 1) continue;
                    float sc2 = score[i];
                    if (reuse) {  // keep-worthiness = decayed popularity / staleness
                        static float dk[256] = {0};
                        if (dk[0] == 0.f) for (int g = 0; g < 256; g++) dk[g] = powf(1.f - POP_ALPHA, (float) g);
                        const uint32_t g = std::min(pass_id - pop_up[i], 255u);
                        const float pd = std::max(dk[g] * pop[i], POP_MIN);
                        sc2 = pd / (float) (pass_id - last_pass[i] + 1);
                    }
                    if (!plan_hint.empty() && plan_hint[i] == pass_id && l > cur_layer) {
                        sc2 += 1e12f; // planned for the rest of this token: evict last
                    }
                    if (sc2 < best) { best = sc2; bl = l; be = e; }
                }
            }
            if (bl < 0) break;
            valid[idx(bl, be)] = 0;
            resident_bytes -= bytes_of[bl];
            pending_drop_bytes += bytes_of[bl];
            victims.push_back({bl, be});
            n_evict++;
        }
    }

    void janitor_loop() {
        while (!stop) {
            std::vector<std::pair<int,int>> victims;
            {
                std::unique_lock<std::mutex> lk(mu);
                jcv.wait(lk, [&] { return stop.load() || !pending_drops.empty(); });
                if (stop) return;
                victims.swap(pending_drops);
            }
            size_t freed = 0;
            for (auto [l, e] : victims) { drop(l, e); freed += bytes_of[l]; }
            {
                std::lock_guard<std::mutex> lk(mu);
                pending_drop_bytes -= std::min(freed, pending_drop_bytes);
            }
            cv.notify_all(); // wake loaders blocked on backpressure
        }
    }

    // Stall packing (2026-07-13): (A) the first g_demand_reserve workers only
    // take demand loads, so a blocked layer never waits behind an in-flight
    // speculative expert; (B) workers pop up to WORKER_BATCH experts per ring
    // submission - deeper NVMe queue, fewer submit/wait round trips. The
    // validity invariant is unchanged: an expert flips valid only after every
    // one of its ranges completed.
    static constexpr int WORKER_BATCH = 4;

    void worker(bool demand_only) {
        io_uring ring;
        io_uring * pring = (!g_no_uring && io_uring_queue_init(64, &ring, 0) == 0) ? &ring : nullptr;
        req batch[WORKER_BATCH];
        while (!stop) {
            int nb = 0;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] {
                    return stop.load() ||
                           (!q.empty() && (!demand_only || q.front().prio == 0));
                });
                if (stop) break;
                // backpressure: real footprint = resident + not-yet-dropped;
                // never let it run more than 2 GiB past budget (was the source
                // of 17-27 GB zram storms during prefill eviction bursts)
                cv.wait(lk, [&] {
                    return stop.load() ||
                           resident_bytes + pending_drop_bytes < budget + (2ULL << 30);
                });
                if (stop) break;
                jcv.notify_one();
                while (nb < WORKER_BATCH && !q.empty() &&
                       (!demand_only || q.front().prio == 0) &&
                       (nb == 0 || q.front().prio == batch[0].prio)) {
                    std::pop_heap(q.begin(), q.end(), [](const req &a, const req &b) { return a.prio > b.prio; });
                    req r = q.back();
                    q.pop_back();
                    const int i = idx(r.layer, r.expert);
                    if (valid[i] || inflight[i]) continue;
                    inflight[i] = 1;
                    batch[nb++] = r;
                }
            }
            if (nb == 0) continue;
            do_load_batch(batch, nb, pring);
            jcv.notify_one();
        }
        if (pring) io_uring_queue_exit(pring);
    }

    // completion bookkeeping shared by both IO paths
    void complete_expert(const req & r) {
        {
            std::lock_guard<std::mutex> lk(mu);
            const int i = idx(r.layer, r.expert);
            inflight[i] = 0;
            valid[i] = 1;
            score[i] = std::max(score[i], 1.5f);
            if (reuse) {  // fresh prefetch: grace period against instant eviction
                pop[i] = std::max(pop[i], POP_ALPHA);
                pop_up[i] = pass_id;
                last_pass[i] = pass_id;
            }
            resident_bytes += bytes_of[r.layer];
            evict_to_budget_locked();
        }
        cv.notify_all();
        if (r.prio == 0) n_demand++; else n_prefetch++;
    }

    struct sqe_tag { const load_range * r; int req_idx; };

    void do_load_batch(const req * batch, int nb, io_uring * ring) {
        if (!ring) {
            for (int b = 0; b < nb; b++) {
                for (const auto & rg : ranges[batch[b].layer][batch[b].expert]) {
                    if (rg.addr && rg.len) load_buffered(rg);
                }
                complete_expert(batch[b]);
            }
            return;
        }
        sqe_tag tags[WORKER_BATCH * 8];
        int pending[WORKER_BATCH] = {0};
        int n_tags = 0, submitted = 0;
        for (int b = 0; b < nb; b++) {
            for (const auto & rg : ranges[batch[b].layer][batch[b].expert]) {
                if (!rg.addr || rg.len == 0) continue;
                if (fds_direct[rg.file] < 0 || n_tags >= WORKER_BATCH * 8) {
                    load_buffered(rg);
                    continue;
                }
                const off_t  a_off = rg.foff & ~(off_t) 4095;
                const size_t head  = (size_t) (rg.foff - a_off);
                const size_t a_len = (rg.len + head + 4095) & ~(size_t) 4095;
                io_uring_sqe * sqe = io_uring_get_sqe(ring);
                if (!sqe) { load_buffered(rg); continue; }
                tags[n_tags] = {&rg, b};
                io_uring_prep_read(sqe, fds_direct[rg.file], rg.addr - head, (unsigned) a_len, a_off);
                io_uring_sqe_set_data(sqe, &tags[n_tags]);
                n_tags++;
                pending[b]++;
                submitted++;
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        if (submitted) io_uring_submit_and_wait(ring, submitted);
        for (int i = 0; i < submitted; i++) {
            io_uring_cqe * cqe = nullptr;
            if (io_uring_wait_cqe(ring, &cqe) < 0) { LOG_ERR("cqe wait failed\n"); abort(); }
            auto * tag = (sqe_tag *) io_uring_cqe_get_data(cqe);
            if (cqe->res < 0) { // e.g. compressed extent rejecting O_DIRECT
                load_buffered(*tag->r);
                n_fallback++;
            } else {
                io_bytes += (size_t) cqe->res;
                n_uring++;
            }
            if (--pending[tag->req_idx] == 0) {
                complete_expert(batch[tag->req_idx]);
            }
            io_uring_cqe_seen(ring, cqe);
        }
        io_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - t0).count();
        // experts whose every range fell back to buffered still need completion
        for (int b = 0; b < nb; b++) {
            bool had_sqe = false;
            for (int t = 0; t < n_tags; t++) if (tags[t].req_idx == b) { had_sqe = true; break; }
            if (!had_sqe) complete_expert(batch[b]);
        }
    }

    void enqueue(int l, int e, int prio) {
        std::lock_guard<std::mutex> lk(mu);
        if (!pageable[l]) return;
        if (prio > 2 && q.size() >= 96) return; // throttle only far speculation
        const int i = idx(l, e);
        if (valid[i] || inflight[i]) return;
        for (const auto &r : q) if (r.layer == l && r.expert == e) return;
        q.push_back({l, e, prio});
        std::push_heap(q.begin(), q.end(), [](const req &a, const req &b) { return a.prio > b.prio; });
        if (prio == 0) cv.notify_all(); else cv.notify_one();
    }

    void on_topk(int l, const int32_t *ids, int k, int n_tokens) {
        if (l < 0 || l >= n_layer || !pageable[l]) return;
        bool queued = false;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (l <= cur_layer) { pass_id++; if (!lru && decay < 1.0f) for (auto &s : score) s *= decay; }
            cur_layer = l;
            std::vector<uint8_t> counted(n_tokens > 1 ? n_expert : 0, 0);  // stats: unique per callback
            for (int t = 0; t < n_tokens; t++) {
                for (int i2 = 0; i2 < k; i2++) {
                    int e = ids[t * k + i2];
                    if (e < 0 || e >= n_expert) continue;
                    const int i = idx(l, e);
                    if (lru) score[i] = (float) ++touch_clock;
                    else score[i] += 1.0f;
                    if (reuse && pop_up[i] != pass_id) {  // one Bernoulli obs per pass
                        pop[i] = powf(1.f - POP_ALPHA, (float) (pass_id - pop_up[i])) * pop[i] + POP_ALPHA;
                        pop_up[i] = pass_id;
                    }
                    last_pass[i] = pass_id;
                    const bool count = counted.empty() || !counted[e];
                    if (!counted.empty()) counted[e] = 1;
                    if (valid[i]) { if (count) n_hit++; continue; }
                    if (count) n_miss++;
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
        // prefill layer-streaming: with a batch of tokens in flight the next
        // layers need nearly every expert - stream them (sequential, perfectly
        // predictable) while this layer computes; plan-hint them for the evictor
        if (pstream && n_tokens >= 64) {
            for (int d = 1; d <= 1 && l + d < n_layer; d++) {
                if (!pageable[l + d]) continue;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    for (int e = 0; e < n_expert; e++) plan_hint[idx(l + d, e)] = pass_id;
                }
                for (int e = 0; e < n_expert; e++) enqueue(l + d, e, d);
                queued = true;
            }
        }
        if (queued) cv.notify_all();
        // block until every expert this layer needs is loaded
        const auto ts0 = std::chrono::steady_clock::now();
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
        stall_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - ts0).count();
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
        // stages: near (delta, d3 probes or gate-ahead) + far (8, d8 probes)
        struct stage { int d; const std::vector<float> * W; };
        stage stages[2];
        int n_stages = 0;
        if (probe_mode || !rw.empty()) stages[n_stages++] = {delta, nullptr};
        if (!rw8.empty() && l < (int) rw8.size() && !rw8[l].empty()) {
            stages[n_stages++] = {8, &rw8[l]};
        }
        for (int si = 0; si < n_stages; si++) {
            const int d_lo = (si > 0 || probe_mode) ? stages[si].d
                                                    : ((l <= 1) ? 1 : stages[si].d);
            for (int d = d_lo; d <= stages[si].d && l + d < n_layer; d++) {
                if (!pageable[l + d]) continue;
                const auto &W = stages[si].W ? *stages[si].W
                              : (probe_mode ? rw[l] : rw[l + d]);
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
                const int m = std::min(fetch * 2, n_expert); // plan top-2m, fetch top-m
                std::partial_sort(sc.begin(), sc.begin() + m, sc.end(),
                                  [](auto &a, auto &b) { return a.first > b.first; });
                {
                    std::lock_guard<std::mutex> lk(mu);
                    for (int i = 0; i < m; i++) {
                        plan_hint[idx(l + d, sc[i].second)] = pass_id; // evictor: hands off
                    }
                }
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
static std::string g_ppl_file;
static int g_ppl_n = 600;
static std::string g_gate;  // wait for this file post-install (cgroup staging)
static int g_spec = 0;
static int g_spec_mtp = 0;   // MTP self-drafting depth (blk.78 nextn head)
static float g_spec_pmin = 0.0f;  // only draft tokens above this probability
static common_speculative * g_spec_ctx = nullptr;
static llama_context * g_ctx_dft = nullptr;
static int g_pstream = 1;  // prefill layer-streaming (0 = off)  // n-gram lookup speculation depth (0 = off)
static FILE * g_dump = nullptr;  // trace dump: 'H' l n h[fp32*n] | 'T' l k ids[i32*k]
static std::mutex g_dump_mu;
static int g_workers = 6;
bool g_no_uring = false;

static bool cb_arena(struct ggml_tensor * t, bool ask, void *) {
    const bool is_topk = strncmp(t->name, "ffn_moe_topk-", 13) == 0;
    const bool is_norm = strncmp(t->name, "ffn_norm-", 9) == 0 ||
                         strncmp(t->name, "post_attn_norm-", 15) == 0;
    // selection-probability tensor: safe to bias (mixture weights read the
    // UNbiased probs tensor; only expert CHOICE is nudged toward residents)
    const bool is_sel = strncmp(t->name, "ffn_moe_probs_biased-", 21) == 0;
    // the graph may hand norms over reshaped 3D ([n_embd, 1, n_tokens]) — a
    // single token means ONE ROW total, not ne[1]==1; and the last layer's
    // output-row slice can be ZERO rows on non-final prefill ubatches
    if (ask) {
        // norms: single token (decode) or small spec-verify batches; big
        // prefill ubatches stay excluded (wrong hidden + zero-row last layer)
        const int64_t nr = is_norm ? ggml_nrows(t) : 0;
        return g_on && (is_topk || (is_norm && nr >= 1 && nr <= 8) ||
                        (is_sel && g_ar.bias_strength > 0.f));
    }
    if (!g_on) return true;
    if (ggml_nelements(t) == 0) return true;
    if (is_sel && g_ar.bias_strength > 0.f && t->type == GGML_TYPE_F32) {
        const int l = atoi(t->name + 21);
        if (l >= 0 && l < g_ar.n_layer && g_ar.pageable[l]) {
            const int E = (int) t->ne[0], T = (int) t->ne[1];
            static std::vector<float> sel;
            sel.resize((size_t) E * T);
            ggml_backend_tensor_get(t, sel.data(), 0, sel.size() * 4);
            {
                std::lock_guard<std::mutex> lk(g_ar.mu);
                for (int e = 0; e < E; e++) {
                    if (!g_ar.valid[g_ar.idx(l, e)]) continue;
                    for (int j = 0; j < T; j++) sel[(size_t) j * E + e] += g_ar.bias_strength;
                }
            }
            ggml_backend_tensor_set(t, sel.data(), 0, sel.size() * 4);
        }
        return true;
    }
    if (is_topk) {
        const int l = atoi(t->name + 13);
        const int k = (int) t->ne[0], n_tokens = (int) t->ne[1];
        if (getenv("ACHILLES_DBG_CB")) {
            fprintf(stderr, "CB %s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] nbytes=%zu type=%d\n",
                    t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                    t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_nbytes(t), (int)t->type);
        }
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
        } else if (g_dump && n_tokens > 1) {  // prefill routing: 'P' l n ids
            std::lock_guard<std::mutex> lk(g_dump_mu);
            fputc('P', g_dump);
            int32_t hdr[2] = {l, k * n_tokens};
            fwrite(hdr, 4, 2, g_dump);
            fwrite(ids.data(), 4, (size_t) k * n_tokens, g_dump);
        }
        g_ar.on_topk(l, ids.data(), k, n_tokens);
    } else if (is_norm && ggml_nrows(t) >= 1 && ggml_nrows(t) <= 8 && t->type == GGML_TYPE_F32) {
        const int l = atoi(strchr(t->name, '-') + 1);
        const int nr = (int) ggml_nrows(t);
        if (getenv("ACHILLES_DBG_CB")) {
            fprintf(stderr, "CB %s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] nbytes=%zu type=%d\n",
                    t->name, (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                    t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_nbytes(t), (int)t->type);
        }
        static std::vector<float> h;
        h.resize((size_t) t->ne[0] * nr);
        ggml_backend_tensor_get(t, h.data(), 0, h.size() * sizeof(float));
        if (g_dump && nr == 1) {
            std::lock_guard<std::mutex> lk(g_dump_mu);
            fputc('H', g_dump);
            int32_t hdr[2] = {l, (int32_t) t->ne[0]};
            fwrite(hdr, 4, 2, g_dump);
            fwrite(h.data(), 4, t->ne[0], g_dump);
        }
        // spec verify batches: score every drafted token's hidden -> the plan
        // covers all in-flight trajectories, not just the committed token
        for (int r = 0; r < nr; r++) {
            g_ar.submit_hidden(l, h.data() + (size_t) r * t->ne[0], (int) t->ne[0]);
        }
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

// expert-major shadow file (scripts/repack_shadow.py): per (l,e,proj) offsets
static std::string g_shadow_prefix;
static std::vector<uint64_t> g_shadow_off, g_shadow_len;  // [l*nE*3 + e*3 + pi]
static int g_shadow_fi = -1;
struct shadow_src { off_t base; size_t slice; int pi; };   // per (layer, push-order proj)
static std::vector<std::vector<shadow_src>> g_shadow_map;

static bool load_shadow_idx(int n_layer, int n_expert) {
    FILE * f = fopen((g_shadow_prefix + ".idx").c_str(), "rb");
    if (!f) { LOG_ERR("arena: shadow idx missing\n"); return false; }
    char magic[4]; uint32_t ver, nl, ne;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "ASHD", 4) ||
        fread(&ver, 4, 1, f) != 1 || fread(&nl, 4, 1, f) != 1 || fread(&ne, 4, 1, f) != 1 ||
        (int) nl != n_layer || (int) ne != n_expert) {
        LOG_ERR("arena: shadow idx header mismatch\n"); fclose(f); return false;
    }
    const size_t n = (size_t) n_layer * n_expert * 3;
    g_shadow_off.resize(n); g_shadow_len.resize(n);
    for (size_t i = 0; i < n; i++) {
        if (fread(&g_shadow_off[i], 8, 1, f) != 1 || fread(&g_shadow_len[i], 8, 1, f) != 1) {
            LOG_ERR("arena: shadow idx truncated\n"); fclose(f); return false;
        }
    }
    fclose(f);
    return true;
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
                if ((int) g_shadow_map.size() < g_ar.n_layer) g_shadow_map.resize(g_ar.n_layer);
                g_shadow_map[l].push_back({base, slice,
                    !strcmp(what, "ffn_gate_exps") ? 0 : !strcmp(what, "ffn_up_exps") ? 1 : 2});
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
    g_ar.pop.assign(N, 0.f);
    g_ar.pop_up.assign(N, 0);
    g_ar.valid.assign(N, 0);
    g_ar.inflight.assign(N, 0);
    g_ar.last_pass.assign(N, 0);
    g_ar.plan_hint.assign(N, 0);
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
    // the loader faulted ~all dense tensors through these mappings while
    // uploading to VRAM; those pages stay resident (mapped, so fadvise can't
    // touch them) and cost ~12GB of cgroup budget on GLM-5.2. Drop the whole
    // mappings now: dense is in VRAM, experts aren't loaded yet, and edges
    // refault from file on demand.
    for (const auto &m : g_ar.maps) {
        madvise((void *) m.start, m.end - m.start, MADV_DONTNEED);
    }
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
    for (int fd : g_ar.fds) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); // dense upload leftovers
    }
    if (!g_shadow_prefix.empty()) {  // retarget loads at the expert-major shadow
        if (!load_shadow_idx(g_ar.n_layer, g_ar.n_expert)) return false;
        g_shadow_fi = (int) g_ar.fds.size();
        g_ar.paths.push_back(g_shadow_prefix + ".bin");
        g_ar.fds.push_back(open(g_ar.paths.back().c_str(), O_RDONLY));
        g_ar.fds_direct.push_back(open(g_ar.paths.back().c_str(), O_RDONLY | O_DIRECT));
        if (g_ar.fds.back() < 0) { LOG_ERR("arena: shadow bin missing\n"); return false; }
        size_t remapped = 0;
        for (int l = 0; l < g_ar.n_layer; l++) {
            if (!g_ar.pageable[l] || l >= (int) g_shadow_map.size()) continue;
            for (size_t p = 0; p < g_shadow_map[l].size(); p++) {
                const auto &src = g_shadow_map[l][p];
                for (int e = 0; e < g_ar.n_expert; e++) {
                    auto &rg = g_ar.ranges[l][e][p];
                    if (!rg.len || !rg.addr) continue;
                    const size_t ii = ((size_t) l * g_ar.n_expert + e) * 3 + src.pi;
                    if (g_shadow_len[ii] != src.slice) {
                        LOG_ERR("arena: shadow len mismatch l=%d e=%d p=%d\n", l, e, src.pi);
                        return false;
                    }
                    const off_t delta = rg.foff - (src.base + (off_t) (src.slice * e));
                    rg.file = g_shadow_fi;
                    rg.foff = (off_t) g_shadow_off[ii] + delta;
                    remapped++;
                }
            }
        }
        LOG_INF("arena: shadow remap: %zu ranges -> %s.bin\n", remapped, g_shadow_prefix.c_str());
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
        else if (a == "--policy") {
            const std::string pol = argv[++i];
            g_ar.lru = pol != "lfu";
            g_ar.reuse = pol == "reuse";
        }
        else if (a == "--bias") g_ar.bias_strength = (float) atof(argv[++i]);
        else if (a == "--ppl") g_ppl_file = argv[++i];
        else if (a == "--ppl-n") g_ppl_n = atoi(argv[++i]);
        else if (a == "--gate") g_gate = argv[++i];
        else if (a == "--probe-d8") {
            FILE * pf = fopen(argv[++i], "rb");
            if (pf) {
                int32_t nl, ne, nh;
                if (fread(&nl, 4, 1, pf) == 1 && fread(&ne, 4, 1, pf) == 1 &&
                    fread(&nh, 4, 1, pf) == 1) {
                    g_ar.probe8_layers = nl;
                    g_ar.probe8_data.assign((size_t) nl * ne * nh, 0.f);
                    if (fread(g_ar.probe8_data.data(), 4, g_ar.probe8_data.size(), pf)
                        != g_ar.probe8_data.size()) {
                        g_ar.probe8_data.clear();
                    }
                }
                fclose(pf);
            }
        }
        else if (a == "--spec") g_spec = atoi(argv[++i]);
        else if (a == "--spec-mtp") g_spec_mtp = atoi(argv[++i]);
        else if (a == "--spec-pmin") g_spec_pmin = (float) atof(argv[++i]);
        else if (a == "--pstream") g_ar.pstream = atoi(argv[++i]);
        else if (a == "--dump") g_dump = fopen(argv[++i], "wb");
        else if (a == "--shadow") g_shadow_prefix = argv[++i];
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
        else if (a == "--no-uring") g_no_uring = true;
        else if (a == "--stats") stats = true;
        else args.push_back(argv[i]);
    }
    common_params params;
    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    // The arena's MAP_FIXED replacement requires tensor bytes to stay at their
    // mmap addresses. CPU weight repacking copies tensors into anonymous
    // buffers - breaks that invariant AND tries to materialize 218 GiB of
    // experts in RAM under -ngl 0 (the three 2026-07-13 system crashes).
    params.no_extra_bufts = true;
    llama_backend_init();
    llama_numa_init(params.numa);

    // load-phase janitor: llama's mmap loader floods the page cache with the
    // whole model (47GB observed) while uploading dense tensors to VRAM, and
    // actively-prefetched mapped pages resist cgroup reclaim -> OOM before
    // prefill. Sweep the mappings every 3s during load; the loader refaults
    // what it still needs, capping the flood at a few seconds of reads.
    std::atomic<bool> loading{true};
    std::thread load_janitor;
    if (g_on) {
        if (!load_meta(params.model.path)) return 1;
        load_janitor = std::thread([&loading] {
            while (loading) {
                for (int i = 0; i < 30 && loading; i++) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                FILE * f = fopen("/proc/self/maps", "r");
                if (!f) continue;
                char line[1024];
                while (fgets(line, sizeof(line), f)) {
                    for (const auto &path : g_ar.paths) {
                        if (!strstr(line, path.c_str())) continue;
                        uintptr_t a, b;
                        if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR, &a, &b) == 2) {
                            madvise((void *) a, b - a, MADV_DONTNEED);
                        }
                    }
                }
                fclose(f);
                // madvise only unmaps; drop the now-unmapped cache pages too
                for (int fd : g_ar.fds) {
                    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                }
            }
        });
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
        if (!g_ar.probe8_data.empty()) {
            const size_t per = (size_t) g_ar.n_expert * g_ar.n_embd;
            g_ar.rw8.assign(g_ar.n_layer, {});
            int applied8 = 0;
            for (int l = 0; l < std::min(g_ar.probe8_layers, g_ar.n_layer); l++) {
                std::vector<float> w(g_ar.probe8_data.begin() + l * per,
                                     g_ar.probe8_data.begin() + (l + 1) * per);
                bool nonzero = false;
                for (size_t j = 0; j < per && !nonzero; j += 997) nonzero = w[j] != 0.f;
                if (nonzero) { g_ar.rw8[l] = std::move(w); applied8++; }
            }
            g_ar.probe8_data.clear();
            LOG_INF("arena: far-stage d8 probes applied to %d layers\n", applied8);
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
        loading = false;
        if (load_janitor.joinable()) load_janitor.join();
        if (!install_arena()) return 1;
        if (!g_gate.empty()) { // wait for the supervisor to raise the ceiling
            LOG_INF("arena: waiting at gate %s\n", g_gate.c_str());
            for (int i = 0; i < 1200 && access(g_gate.c_str(), F_OK) != 0; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        for (int i = 0; i < g_workers; i++) {
            const bool demand_only = i < 2 && g_workers > 3;  // reserve 2 for demand latency
            g_ar.workers.emplace_back([demand_only] { g_ar.worker(demand_only); });
        }
        g_ar.scorer = std::thread([] { g_ar.score_loop(); });
        g_ar.janitor = std::thread([] { g_ar.janitor_loop(); });
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (!g_ppl_file.empty()) {  // teacher-forced quality eval, then exit
        FILE * tf = fopen(g_ppl_file.c_str(), "rb");
        if (!tf) { LOG_ERR("ppl file missing\n"); return 1; }
        std::string text;
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), tf)) > 0) text.append(buf, n);
        fclose(tf);
        std::vector<llama_token> tk = common_tokenize(lctx, text, true, false);
        if ((int) tk.size() > g_ppl_n) tk.resize(g_ppl_n);
        llama_batch batch = llama_batch_init((int) tk.size(), 0, 1);
        for (size_t i = 0; i < tk.size(); i++) {
            batch.token[i] = tk[i];
            batch.pos[i] = (llama_pos) i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        batch.n_tokens = (int) tk.size();
        const double t0 = ggml_time_us() / 1e6;
        if (llama_decode(lctx, batch)) { LOG_ERR("ppl decode failed\n"); return 1; }
        const int n_vocab = llama_vocab_n_tokens(vocab);
        double nll = 0;
        for (size_t i = 0; i + 1 < tk.size(); i++) {
            const float * lg = llama_get_logits_ith(lctx, (int) i);
            float mx = -1e30f;
            for (int v = 0; v < n_vocab; v++) mx = std::max(mx, lg[v]);
            double Z = 0;
            for (int v = 0; v < n_vocab; v++) Z += exp((double) lg[v] - mx);
            nll -= ((double) lg[tk[i + 1]] - mx) - log(Z);
        }
        const double t1 = ggml_time_us() / 1e6;
        const double mean_nll = nll / (double) (tk.size() - 1);
        fprintf(stderr, "\nACHILLES ppl: tokens=%zu mean_nll=%.5f ppl=%.3f bias=%.4f (%.0fs)\n",
                tk.size(), mean_nll, exp(mean_nll), g_ar.bias_strength, t1 - t0);
        llama_batch_free(batch);
        g_ar.stop = true;
        g_ar.cv.notify_all();
        g_ar.scv.notify_all();
        g_ar.jcv.notify_all();
        for (auto &w : g_ar.workers) w.join();
        if (g_ar.scorer.joinable()) g_ar.scorer.join();
        if (g_ar.janitor.joinable()) g_ar.janitor.join();
        return 0;
    }

    std::vector<llama_token> toks =
        common_tokenize(lctx, params.prompt, llama_vocab_get_add_bos(vocab), true);

    static common_speculative_init_result_ptr spec_res;  // owns the MTP draft context
    if (g_spec_mtp > 0) {
        params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.speculative.draft.n_max = g_spec_mtp;
        params.speculative.draft.n_min = 0;
        params.speculative.draft.p_min = g_spec_pmin;
        // the MTP ctx only ever decodes draft-sized batches; a full-size batch
        // makes the scheduler reserve worst-case (multi-GB, host-visible
        // fallback when VRAM is full - the 2026-07-13 anon explosions)
        common_params mtp_params = params;
        mtp_params.n_batch  = std::max(64, g_spec_mtp + 2);
        mtp_params.n_ubatch = mtp_params.n_batch;
        spec_res = common_speculative_init_from_params(mtp_params, model, lctx);
        g_ctx_dft = spec_res ? spec_res->context() : nullptr;
        if (!g_ctx_dft) { LOG_ERR("arena: MTP draft context failed (no nextn layers?)\n"); return 1; }
        params.speculative.draft.ctx_tgt = lctx;
        params.speculative.draft.ctx_dft = g_ctx_dft;
        g_spec_ctx = common_speculative_init(params.speculative, 1);
        LOG_INF("arena: MTP self-drafting enabled, depth %d (nextn layers: %d)\n",
                g_spec_mtp, (int) llama_model_n_layer_nextn(model));
    }
    const double t_p0 = ggml_time_us() / 1e6;
    {   // chunked prefill: llama_decode caps batches at n_batch tokens.
        // With MTP, every position needs logits=1 so process() can extract
        // h_nextn per row and populate the draft context's KV.
        const int nb = (int) llama_n_batch(lctx);
        llama_batch pb2 = g_spec_ctx ? llama_batch_init(nb, 0, 1) : llama_batch{};
        for (size_t off = 0; off < toks.size(); off += nb) {
            const int n = (int) std::min(toks.size() - off, (size_t) nb);
            llama_batch pb = llama_batch_get_one(toks.data() + off, n);
            if (g_spec_ctx) {
                for (int i = 0; i < n; i++) {
                    pb2.token[i] = toks[off + i];
                    pb2.pos[i] = (llama_pos) (off + i);
                    pb2.n_seq_id[i] = 1;
                    pb2.seq_id[i][0] = 0;
                    pb2.logits[i] = true;
                }
                pb2.n_tokens = n;
                pb = pb2;
            }
            if (llama_decode(lctx, pb)) { LOG_ERR("prefill chunk failed\n"); return 1; }
            if (g_spec_ctx && !common_speculative_process(g_spec_ctx, pb)) {
                LOG_ERR("arena: MTP process failed on prefill chunk\n"); return 1;
            }
        }
        if (g_spec_ctx) llama_batch_free(pb2);
    }
    if (g_spec_ctx) common_speculative_begin(g_spec_ctx, 0, toks);
    const double t_p1 = ggml_time_us() / 1e6;
    g_ar.snapshot();  // decode-only deltas for the final stats print

    auto sp = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));

    std::string text;
    const int n_gen = params.n_predict > 0 ? params.n_predict : 64;
    int gen = 0;
    // n-gram lookup speculation: draft the continuation of a previously seen
    // n-gram, decode t0+draft as ONE batch (expert loads amortize across the
    // batch), accept the longest prefix the sampler agrees with, rewind KV.
    std::vector<llama_token> hist = toks;
    auto ngram_draft = [&](int depth) {
        std::vector<llama_token> d;
        for (int n = 3; n >= 2 && d.empty(); n--) {
            if ((int) hist.size() < n + 1) continue;
            for (int i = (int) hist.size() - n - 1; i >= 0; i--) {
                bool match = true;
                for (int j = 0; j < n; j++) {
                    if (hist[i + j] != hist[hist.size() - n + j]) { match = false; break; }
                }
                if (match) {
                    for (int j = 0; j < depth && i + n + j < (int) hist.size(); j++) {
                        d.push_back(hist[i + n + j]);
                    }
                    break;
                }
            }
        }
        return d;
    };
    llama_memory_t mem = llama_get_memory(lctx);
    llama_pos n_past = (llama_pos) toks.size();
    llama_batch sb = llama_batch_init(std::max(g_spec, g_spec_mtp) + 1, 0, 1);
    bool have_pending = false;
    llama_token pending = 0;
    uint64_t n_spec_acc = 0, n_spec_try = 0;
    const double t_g0 = ggml_time_us() / 1e6;
    while (gen < n_gen) {
        llama_token t0 = have_pending ? pending : llama_sampler_sample(smpl, lctx, -1);
        have_pending = false;
        if (llama_vocab_is_eog(vocab, t0)) break;
        text += common_token_to_piece(lctx, t0);
        hist.push_back(t0);
        gen++;
        std::vector<llama_token> draft;
        if (g_spec_ctx) {
            hist.pop_back();  // draft API wants the prompt WITHOUT id_last
            common_speculative_get_draft_params(g_spec_ctx, 0) = {
                /*drafting*/ true, /*n_max*/ -1, /*n_past*/ n_past, /*id_last*/ t0,
                /*prompt*/ &hist, /*result*/ &draft };
            common_speculative_draft(g_spec_ctx);
            hist.push_back(t0);
            // draft() decoded speculative rows into ctx_dft's KV at >= n_past;
            // rewind before the verify decode or process() sees stale positions
            llama_memory_seq_rm(llama_get_memory(g_ctx_dft), 0, n_past, -1);
        } else if (g_spec > 0) {
            draft = ngram_draft(g_spec);
        }
        sb.n_tokens = 1 + (int) draft.size();
        sb.token[0] = t0; sb.pos[0] = n_past; sb.n_seq_id[0] = 1; sb.seq_id[0][0] = 0;
        sb.logits[0] = true;
        for (size_t i = 0; i < draft.size(); i++) {
            sb.token[i + 1] = draft[i]; sb.pos[i + 1] = n_past + 1 + (llama_pos) i;
            sb.n_seq_id[i + 1] = 1; sb.seq_id[i + 1][0] = 0; sb.logits[i + 1] = true;
        }
        if (llama_decode(lctx, sb)) break;
        if (g_spec_ctx && !common_speculative_process(g_spec_ctx, sb)) break;
        int n_acc = 0;
        for (size_t i = 0; i <= draft.size() && gen < n_gen; i++) {
            llama_token t = llama_sampler_sample(smpl, lctx, (int) i);
            if (i < draft.size()) n_spec_try++;
            if (i < draft.size() && t == draft[i] && !llama_vocab_is_eog(vocab, t)) {
                text += common_token_to_piece(lctx, t);
                hist.push_back(t);
                gen++; n_acc++; n_spec_acc++;
            } else {
                pending = t;
                have_pending = true;
                break;
            }
        }
        if ((int) draft.size() > n_acc) {
            llama_memory_seq_rm(mem, 0, n_past + 1 + n_acc, -1);
        }
        n_past += 1 + n_acc;
        if (g_spec_ctx) {
            common_speculative_accept(g_spec_ctx, 0, (uint16_t) n_acc);
            llama_memory_seq_rm(llama_get_memory(g_ctx_dft), 0, n_past, -1);
        }
    }
    const double t_g1 = ggml_time_us() / 1e6;
    llama_batch_free(sb);
    if (g_spec > 0 || g_spec_mtp > 0) {
        fprintf(stderr, "ACHILLES spec: tried=%" PRIu64 " accepted=%" PRIu64 " (%.0f%%)\n",
                n_spec_try, n_spec_acc, n_spec_try ? 100.0 * n_spec_acc / n_spec_try : 0.0);
    }

    g_ar.stop = true;
    g_ar.cv.notify_all();
    g_ar.scv.notify_all();
    g_ar.jcv.notify_all();
    for (auto &w : g_ar.workers) w.join();
    if (g_ar.scorer.joinable()) g_ar.scorer.join();
    if (g_ar.janitor.joinable()) g_ar.janitor.join();

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
        const double d_stall = (g_ar.stall_ns - g_ar.snap[0]) / 1e9;
        const double d_gb    = (g_ar.io_bytes - g_ar.snap[1]) / 1e9;
        const double d_ios   = (g_ar.io_ns    - g_ar.snap[2]) / 1e9;
        const uint64_t d_hit = g_ar.n_hit - g_ar.snap[3], d_miss = g_ar.n_miss - g_ar.snap[4];
        fprintf(stderr, "ACHILLES decode-detail: stall=%.1fs io=%.1fGB io_worker_time=%.1fs "
                "per_stream_bw=%.2fGB/s hit=%.3f (decode-only)\n",
                d_stall, d_gb, d_ios, d_ios > 0 ? d_gb / d_ios : 0.0,
                d_hit + d_miss ? (double) d_hit / (d_hit + d_miss) : 0.0);
    }
    fprintf(stderr, "OUTPUT: %.240s\n", text.c_str());
    llama_sampler_free(smpl);
    return 0;
}
