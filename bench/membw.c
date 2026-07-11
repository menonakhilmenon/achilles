// membw: STREAM-style sustained RAM bandwidth benchmark.
// Kernels: read (sum), copy (a=b), triad (a=b+s*c).
// The "read" kernel is the one that matters for weight-streaming GEMV decode.
//
// Usage: membw [-t threads] [-g total_gib] [-r reps]
#define _GNU_SOURCE
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    double *a, *b, *c;
    size_t n;
    int id;
} tctx_t;

static int g_threads = 0;
static pthread_barrier_t g_bar;
static volatile int g_kernel;      // 0=read 1=copy 2=triad -1=exit
static double g_sink;

static void *worker(void *arg) {
    tctx_t *t = (tctx_t *)arg;
    size_t n = t->n;
    double *a = t->a, *b = t->b, *c = t->c;
    double local = 0;
    for (;;) {
        pthread_barrier_wait(&g_bar); // wait for kernel selection
        int k = g_kernel;
        if (k < 0) break;
        pthread_barrier_wait(&g_bar); // synchronized start
        switch (k) {
        case 0: {
            double s = 0;
            for (size_t i = 0; i < n; i++) s += a[i];
            local += s;
            break;
        }
        case 1:
            for (size_t i = 0; i < n; i++) c[i] = a[i];
            break;
        case 2:
            for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
            break;
        }
        pthread_barrier_wait(&g_bar); // synchronized end
    }
    g_sink += local;
    return NULL;
}

int main(int argc, char **argv) {
    int threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    double total_gib = 8.0;
    int reps = 7, opt;
    while ((opt = getopt(argc, argv, "t:g:r:")) != -1) {
        if (opt == 't') threads = atoi(optarg);
        else if (opt == 'g') total_gib = atof(optarg);
        else if (opt == 'r') reps = atoi(optarg);
    }
    g_threads = threads;
    size_t per = (size_t)(total_gib * (1ULL << 30) / threads / 3 / sizeof(double));
    per &= ~(size_t)7;

    pthread_barrier_init(&g_bar, NULL, threads + 1);
    pthread_t th[256];
    tctx_t ctx[256];
    for (int i = 0; i < threads; i++) {
        ctx[i].n = per;
        ctx[i].id = i;
        ctx[i].a = malloc(per * sizeof(double));
        ctx[i].b = malloc(per * sizeof(double));
        ctx[i].c = malloc(per * sizeof(double));
        if (!ctx[i].a || !ctx[i].b || !ctx[i].c) { perror("malloc"); return 1; }
        for (size_t j = 0; j < per; j++) { ctx[i].a[j] = 1.0; ctx[i].b[j] = 2.0; ctx[i].c[j] = 0.5; }
        pthread_create(&th[i], NULL, worker, &ctx[i]);
    }

    const char *names[3] = {"read", "copy", "triad"};
    // bytes moved per element (application-level; write-allocate traffic excluded)
    const double bpe[3] = {8.0, 16.0, 24.0};
    printf("{\"threads\":%d,\"array_bytes_per_thread\":%zu,\"kernels\":{", threads, per * 8 * 3);
    for (int k = 0; k < 3; k++) {
        double best = 0, sum = 0;
        for (int r = 0; r < reps; r++) {
            g_kernel = k;
            pthread_barrier_wait(&g_bar);
            pthread_barrier_wait(&g_bar);
            double t0 = now();
            pthread_barrier_wait(&g_bar);
            double dt = now() - t0;
            double gbps = (double)per * threads * bpe[k] / dt / 1e9;
            if (r > 0) { sum += gbps; if (gbps > best) best = gbps; } // skip warmup rep
        }
        printf("%s\"%s\":{\"best_GBps\":%.2f,\"avg_GBps\":%.2f}", k ? "," : "", names[k], best,
               sum / (reps - 1));
        fflush(stdout);
    }
    printf("}}\n");
    g_kernel = -1;
    pthread_barrier_wait(&g_bar);
    for (int i = 0; i < threads; i++) pthread_join(th[i], NULL);
    if (g_sink == 42.0) printf("#\n");
    return 0;
}
