#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "veclite.h"

#define VECLITE_E_GPU (-30)

int32_t veclite_gpu_open(const void *data, int32_t dtype, int64_t n, int64_t dim, const int64_t *tags, const uint8_t *live, int64_t *out_h);
int32_t veclite_gpu_update(int64_t h, const int64_t *tags, const uint8_t *live);
int32_t veclite_gpu_close(int64_t h);
int32_t veclite_gpu_search(int64_t h, const float *queries, int64_t nq, const uint8_t *filter_on, const int64_t *filter_tags,
                       int64_t k, int32_t metric, int64_t *out_ids, float *out_scores, int64_t *out_counts, float *out_full);

static int failures = 0, checks = 0;

#define CHECK(cond, ...)                                          \
    do {                                                          \
        checks++;                                                 \
        if (!(cond)) {                                            \
            if (failures++ < 12) {                                \
                fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                fprintf(stderr, __VA_ARGS__);                     \
                fprintf(stderr, "\n");                            \
            }                                                     \
        }                                                         \
    } while (0)

static uint64_t rng = 88172645463325252ull;
static uint64_t rnd(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
}
static int64_t below(int64_t n) { return (int64_t)(rnd() % (uint64_t)n); }

typedef struct {
    float s;
    int64_t id;
} cand;

static int cand_cmp(const void *a, const void *b) {
    const cand *x = a, *y = b;
    if (x->s != y->s) return x->s < y->s ? -1 : 1;
    return (x->id > y->id) - (x->id < y->id);
}

static void test_errors(void) {
    float x[8] = {1, 2, 3, 4, 5, 6, 7, 8}, q[2] = {1, 1}, sc[4];
    int64_t h = 0, ids[4], cnt[2] = {9, 9}, tags[4] = {0};
    uint8_t live[4] = {1, 1, 1, 1}, bad_fo[1] = {2}, fo[1] = {1};
    CHECK(veclite_gpu_open(x, 0, 4, 2, tags, live, NULL) == VECLITE_E_NULL, "open null out");
    CHECK(veclite_gpu_open(NULL, 0, 4, 2, tags, live, &h) == VECLITE_E_NULL, "open null data");
    CHECK(veclite_gpu_open(x, 0, 4, 2, NULL, live, &h) == VECLITE_E_NULL, "open null tags");
    CHECK(veclite_gpu_open(x, 0, 4, 2, tags, NULL, &h) == VECLITE_E_NULL, "open null live");
    CHECK(veclite_gpu_open(x, 2, 4, 2, tags, live, &h) == VECLITE_E_DTYPE, "open bad dtype");
    CHECK(veclite_gpu_open(x, 0, 4, 0, tags, live, &h) == VECLITE_E_DIM, "open dim 0");
    CHECK(veclite_gpu_open(x, 0, -1, 2, tags, live, &h) == VECLITE_E_N, "open neg n");
    CHECK(veclite_gpu_open(x, 0, 4, 2, tags, live, &h) == VECLITE_OK && h != 0, "open");
    CHECK(veclite_gpu_search(h, q, 1, NULL, NULL, 2, 0, ids, sc, NULL, NULL) == VECLITE_E_NULL, "null counts");
    CHECK(veclite_gpu_search(h, q, 0, NULL, NULL, 2, 0, ids, sc, cnt, NULL) == VECLITE_E_BATCH, "batch 0");
    CHECK(veclite_gpu_search(0, q, 1, NULL, NULL, 2, 0, ids, sc, cnt, NULL) == VECLITE_E_NULL && cnt[0] == 0, "null handle resets count");
    CHECK(veclite_gpu_search(h, NULL, 1, NULL, NULL, 2, 0, ids, sc, cnt, NULL) == VECLITE_E_NULL, "null queries");
    CHECK(veclite_gpu_search(h, q, 1, NULL, NULL, 0, 0, ids, sc, cnt, NULL) == VECLITE_E_K, "k 0");
    CHECK(veclite_gpu_search(h, q, 1, NULL, NULL, VECLITE_MAX_K + 1, 0, ids, sc, cnt, NULL) == VECLITE_E_K, "k max");
    CHECK(veclite_gpu_search(h, q, 1, NULL, NULL, 2, 3, ids, sc, cnt, NULL) == VECLITE_E_METRIC, "metric");
    CHECK(veclite_gpu_search(h, q, 1, bad_fo, tags, 2, 0, ids, sc, cnt, NULL) == VECLITE_E_FILTER, "filter 2");
    CHECK(veclite_gpu_search(h, q, 1, fo, NULL, 2, 0, ids, sc, cnt, NULL) == VECLITE_E_NULL, "filter needs tags");
    float nanq[2] = {NAN, 1}, zq[2] = {0, 0};
    CHECK(veclite_gpu_search(h, nanq, 1, NULL, NULL, 2, 0, ids, sc, cnt, NULL) == VECLITE_E_QUERY_NONFINITE, "nan query");
    CHECK(veclite_gpu_search(h, zq, 1, NULL, NULL, 2, 2, ids, sc, cnt, NULL) == VECLITE_E_QUERY_ZERO, "zero cosine query");
    CHECK(veclite_gpu_update(h, NULL, live) == VECLITE_E_NULL, "update null");
    CHECK(veclite_gpu_close(h) == VECLITE_OK, "close");
    CHECK(veclite_gpu_close(0) == VECLITE_E_NULL, "close null");
    float nx[8] = {1, 2, NAN, 4, 5, 6, 7, 8};
    uint8_t dead1[4] = {1, 0, 1, 1};
    CHECK(veclite_gpu_open(nx, 0, 4, 2, tags, live, &h) == VECLITE_OK, "open nan data");
    CHECK(veclite_gpu_search(h, q, 1, NULL, NULL, 2, 0, ids, sc, cnt, NULL) == VECLITE_E_SCORE_NONFINITE && cnt[0] == 0, "nan row");
    CHECK(veclite_gpu_update(h, tags, dead1) == VECLITE_OK, "update");
    CHECK(veclite_gpu_search(h, q, 1, NULL, NULL, 2, 0, ids, sc, cnt, NULL) == VECLITE_OK && cnt[0] == 2, "deleted nan row skipped after update");
    veclite_gpu_close(h);
}

static void test_random(int trials) {
    for (int t = 0; t < trials; t++) {
        int64_t n = below(4) == 0 ? 4096 + below(9000) : 1 + below(600);
        int64_t dim = 1 + below(70), k = 1 + below(20), nq = 1 + below(20);
        int32_t metric = (int32_t)below(3), dtype = (int32_t)below(2);
        float *x = malloc(sizeof(float) * (size_t)(n * dim)), *qs = malloc(sizeof(float) * (size_t)(nq * dim));
        float *full = malloc(sizeof(float) * (size_t)(n * nq)), *cpu_full = malloc(sizeof(float) * (size_t)(n * nq));
        _Float16 *h16 = malloc(sizeof(_Float16) * (size_t)(n * dim));
        int64_t *tags = malloc(sizeof(int64_t) * (size_t)n), bad;
        uint8_t *live = malloc((size_t)n), fo[20];
        int64_t ft[20], gi[20 * 20], ci[20 * 20], gc[20], cc[20], h;
        float gs[20 * 20], cs[20 * 20];
        cand *ref = malloc(sizeof(cand) * (size_t)n);
        for (int64_t i = 0; i < n * dim; i++) x[i] = (float)below(3);
        for (int64_t i = 0; i < n; i++) {
            x[i * dim] += 1;
            tags[i] = below(3);
            live[i] = below(6) != 0;
        }
        for (int64_t i = 0; i < nq * dim; i++) qs[i] = (float)below(3);
        for (int64_t q = 0; q < nq; q++) {
            qs[q * dim] += 1;
            fo[q] = (uint8_t)below(2);
            ft[q] = below(3);
        }
        CHECK(veclite_encode(VECLITE_F16, x, n, dim, h16, &bad) == VECLITE_OK, "encode");
        const void *data = dtype ? (const void *)h16 : (const void *)x;
        uint8_t *all_live = malloc((size_t)n);
        memset(all_live, 1, (size_t)n);
        CHECK(veclite_gpu_open(data, dtype, n, dim, tags, all_live, &h) == VECLITE_OK, "gpu open");
        CHECK(veclite_gpu_update(h, tags, live) == VECLITE_OK, "gpu update");
        free(all_live);
        int rc = veclite_gpu_search(h, qs, nq, fo, ft, k, metric, gi, gs, gc, full);
        CHECK(rc == VECLITE_OK, "trial %d gpu rc %d", t, rc);
        CHECK(veclite_search_ex(data, dtype, n, dim, tags, live, qs, nq, fo, ft, k, metric, 1 + below(8), ci, cs, cc) == VECLITE_OK, "cpu rc");
        CHECK(veclite_scores(x, n, dim, qs, nq, metric, cpu_full) == VECLITE_OK, "cpu scores");
        for (int64_t q = 0; q < nq; q++) {
            int64_t m = 0;
            for (int64_t i = 0; i < n; i++)
                if (live[i] && (!fo[q] || tags[i] == ft[q])) ref[m++] = (cand){full[q * n + i], i};
            qsort(ref, (size_t)m, sizeof(cand), cand_cmp);
            int64_t want = m < k ? m : k;
            CHECK(gc[q] == want, "trial %d q %lld count %lld want %lld (n %lld)", t, (long long)q, (long long)gc[q], (long long)want, (long long)n);
            for (int64_t j = 0; j < want && j < gc[q]; j++)
                CHECK(gi[q * k + j] == ref[j].id && gs[q * k + j] == ref[j].s, "trial %d q %lld pos %lld gpu id %lld want %lld (n %lld metric %d)", t,
                      (long long)q, (long long)j, (long long)gi[q * k + j], (long long)ref[j].id, (long long)n, metric);
            if (metric != 2) {
                for (int64_t i = 0; i < n; i++)
                    CHECK(full[q * n + i] == cpu_full[q * n + i], "trial %d integer-data score must be exact q %lld row %lld gpu %g cpu %g", t,
                          (long long)q, (long long)i, full[q * n + i], cpu_full[q * n + i]);
                CHECK(gc[q] == cc[q] && memcmp(gi + q * k, ci + q * k, sizeof(int64_t) * (size_t)gc[q]) == 0, "trial %d gpu != cpu ids q %lld", t, (long long)q);
            } else {
                for (int64_t i = 0; i < n; i++)
                    CHECK(fabsf(full[q * n + i] - cpu_full[q * n + i]) <= 1e-5f, "trial %d cosine score drift q %lld row %lld", t, (long long)q, (long long)i);
            }
        }
        CHECK(veclite_gpu_close(h) == VECLITE_OK, "close");
        free(x);
        free(qs);
        free(full);
        free(cpu_full);
        free(h16);
        free(tags);
        free(live);
        free(ref);
    }
}

int main(int argc, char **argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 150;
    rng = argc > 2 ? (uint64_t)atoll(argv[2]) : rng;
    test_errors();
    test_random(trials);
    printf("gpu_test: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
