// ssdread: O_DIRECT NVMe benchmark at expert-sized granularity.
//
//   ssdread write <file> <gib>                      create scratch (xorshift data)
//   ssdread read  <file> -b <block_mib> -t <threads> -T <secs> -m rand|seq [-j]
//
// Sync pread per thread == queue depth ~= threads (fine for multi-MB blocks).
// O_DIRECT bypasses the page cache; create the file with chattr +C on btrfs
// (nodatacow) so reads hit the raw device path (no checksum/CoW detour).
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ALIGN 4096

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint64_t xs(uint64_t *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return *s;
}

static volatile int g_stop;
typedef struct {
    const char *path;
    size_t bs, fsize;
    int mode; // 0 rand, 1 seq
    int id, nthreads;
    uint64_t bytes, ios;
    double lat_sum, lat_max;
} rctx_t;

static void *reader(void *arg) {
    rctx_t *c = (rctx_t *)arg;
    int fd = open(c->path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); exit(1); }
    void *buf;
    if (posix_memalign(&buf, ALIGN, c->bs)) { perror("memalign"); exit(1); }
    uint64_t seed = 0x9e3779b97f4a7c15ULL * (c->id + 1) ^ (uint64_t)now();
    size_t nblocks = c->fsize / c->bs;
    size_t part = nblocks / c->nthreads;
    size_t pos = (size_t)c->id * part; // seq: private partition, wraps
    while (!g_stop) {
        size_t blk = c->mode == 0 ? xs(&seed) % nblocks
                                  : c->id * part + (pos++ % (part ? part : 1));
        double t0 = now();
        ssize_t r = pread(fd, buf, c->bs, (off_t)(blk * c->bs));
        double dt = now() - t0;
        if (r != (ssize_t)c->bs) { perror("pread"); exit(1); }
        c->bytes += r;
        c->ios++;
        c->lat_sum += dt;
        if (dt > c->lat_max) c->lat_max = dt;
    }
    close(fd);
    free(buf);
    return NULL;
}

static int do_write(const char *path, double gib) {
    int fd = open(path, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (fd < 0) { perror("open"); return 1; }
    size_t bs = 16 << 20, total = (size_t)(gib * (1ULL << 30));
    uint64_t *buf;
    if (posix_memalign((void **)&buf, ALIGN, bs)) { perror("memalign"); return 1; }
    uint64_t seed = 0xdeadbeefcafeULL;
    for (size_t i = 0; i < bs / 8; i++) buf[i] = xs(&seed);
    double t0 = now();
    for (size_t off = 0; off < total; off += bs) {
        buf[0] = off; // vary content per block
        if (pwrite(fd, buf, bs, (off_t)off) != (ssize_t)bs) { perror("pwrite"); return 1; }
        if ((off >> 30) != ((off + bs) >> 30)) { fprintf(stderr, "\r%zu GiB", (off + bs) >> 30); }
    }
    fsync(fd);
    double dt = now() - t0;
    fprintf(stderr, "\n");
    printf("{\"write_GBps\":%.2f,\"gib\":%.0f}\n", total / dt / 1e9, gib);
    close(fd);
    free(buf);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[1], "write")) return do_write(argv[2], atof(argv[3]));
    if (argc < 3 || strcmp(argv[1], "read")) {
        fprintf(stderr, "usage: %s write <file> <gib> | read <file> -b mib -t n -T s -m rand|seq\n",
                argv[0]);
        return 1;
    }
    const char *path = argv[2];
    size_t bs = 20 << 20;
    int threads = 8, secs = 10, mode = 0, opt;
    optind = 3;
    while ((opt = getopt(argc, argv, "b:t:T:m:")) != -1) {
        if (opt == 'b') bs = (size_t)(atof(optarg) * (1 << 20));
        else if (opt == 't') threads = atoi(optarg);
        else if (opt == 'T') secs = atoi(optarg);
        else if (opt == 'm') mode = strcmp(optarg, "seq") == 0;
    }
    bs &= ~(size_t)(ALIGN - 1);
    struct stat st;
    if (stat(path, &st)) { perror("stat"); return 1; }

    rctx_t ctx[256];
    pthread_t th[256];
    memset(ctx, 0, sizeof(ctx));
    g_stop = 0;
    for (int i = 0; i < threads; i++) {
        ctx[i] = (rctx_t){.path = path, .bs = bs, .fsize = (size_t)st.st_size,
                          .mode = mode, .id = i, .nthreads = threads};
        pthread_create(&th[i], NULL, reader, &ctx[i]);
    }
    double t0 = now();
    struct timespec req = {secs, 0};
    nanosleep(&req, NULL);
    g_stop = 1;
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    double dt = now() - t0;

    uint64_t bytes = 0, ios = 0;
    double lat_sum = 0, lat_max = 0;
    for (int i = 0; i < threads; i++) {
        bytes += ctx[i].bytes;
        ios += ctx[i].ios;
        lat_sum += ctx[i].lat_sum;
        if (ctx[i].lat_max > lat_max) lat_max = ctx[i].lat_max;
    }
    printf("{\"mode\":\"%s\",\"block_mib\":%.2f,\"threads\":%d,\"secs\":%.1f,"
           "\"GBps\":%.2f,\"iops\":%.0f,\"lat_avg_ms\":%.2f,\"lat_max_ms\":%.1f}\n",
           mode ? "seq" : "rand", bs / 1048576.0, threads, dt, bytes / dt / 1e9,
           ios / dt, ios ? lat_sum / ios * 1e3 : 0, lat_max * 1e3);
    return 0;
}
