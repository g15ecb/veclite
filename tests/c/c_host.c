#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "veclite.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...)                                          \
    do {                                                          \
        checks++;                                                 \
        if (!(cond)) {                                            \
            failures++;                                           \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);  \
            fprintf(stderr, __VA_ARGS__);                         \
            fprintf(stderr, "\n");                                \
        }                                                         \
    } while (0)

typedef struct {
    float score;
    int64_t id;
} cand;

static int cand_cmp(const void *a, const void *b) {
    const cand *x = a, *y = b;
    if (x->score != y->score) return x->score < y->score ? -1 : 1;
    return (x->id > y->id) - (x->id < y->id);
}

static int64_t reference(const float *scores, const int64_t *tags, const uint8_t *live, int64_t n, int64_t k,
                         int filter_on, int64_t tag, cand *out) {
    cand *all = malloc(sizeof(cand) * (size_t)(n + 1));
    int64_t m = 0;
    for (int64_t i = 0; i < n; i++)
        if (live[i] && (!filter_on || tags[i] == tag)) all[m++] = (cand){scores[i], i};
    qsort(all, (size_t)m, sizeof(cand), cand_cmp);
    int64_t r = m < k ? m : k;
    memcpy(out, all, sizeof(cand) * (size_t)r);
    free(all);
    return r;
}

static int32_t search1(const float *d, int64_t n, int64_t dim, const int64_t *tags, const uint8_t *live,
                       const float *q, int64_t k, int32_t metric, int fo, int64_t ft, int64_t nc, int64_t *ids,
                       float *sc, int64_t *cnt) {
    uint8_t f = (uint8_t)fo;
    return veclite_search(d, n, dim, tags, live, q, 1, &f, &ft, k, metric, nc, ids, sc, cnt);
}

static void test_errors(void) {
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int64_t tags[4] = {0, 1, 0, 1};
    uint8_t live[4] = {1, 1, 1, 1};
    float q[4] = {1, 1, 2, 2};
    uint8_t fo[2] = {1, 0}, bad_fo[2] = {0, 2};
    int64_t ft[2] = {0, 0};
    int64_t ids[16];
    float sc[16];
    int64_t cnt[2] = {99, 99};
    int64_t bad = 0;
    float nanq[2] = {NAN, 1}, infq[2] = {INFINITY, 1}, zeroq[2] = {0, 0};

    CHECK(veclite_abi_version() == 4, "abi");
    CHECK(veclite_search(NULL, 4, 2, tags, live, q, 2, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_NULL, "null data");
    CHECK(cnt[0] == 0 && cnt[1] == 0, "counts reset before validation");
    CHECK(veclite_search(data, 4, 2, tags, NULL, q, 2, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_NULL, "null live");
    CHECK(veclite_search(data, 4, 2, NULL, live, q, 2, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_NULL, "null tags with filter");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, NULL, 2, 0, 1, ids, sc, cnt) == VECLITE_E_NULL, "null filter tags");
    CHECK(veclite_search(data, 4, 2, NULL, live, q, 2, NULL, NULL, 2, 0, 1, ids, sc, cnt) == VECLITE_OK, "no filters ok");
    CHECK(veclite_search(data, 4, 2, tags, live, NULL, 2, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_NULL, "null queries");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, 2, 0, 1, NULL, sc, cnt) == VECLITE_E_NULL, "null ids");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, 2, 0, 1, ids, NULL, cnt) == VECLITE_E_NULL, "null scores");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, 2, 0, 1, ids, sc, NULL) == VECLITE_E_NULL, "null counts");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 0, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_BATCH, "batch 0");
    CHECK(veclite_search(data, 4, 2, tags, live, q, VECLITE_MAX_BATCH + 1, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_BATCH, "batch max");
    CHECK(veclite_search(data, -1, 2, tags, live, q, 2, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_N, "neg n");
    CHECK(veclite_search(data, 4, 0, tags, live, q, 2, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_DIM, "dim 0");
    CHECK(veclite_search(data, 4, VECLITE_MAX_DIM + 1, tags, live, q, 2, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_DIM, "dim max");
    CHECK(veclite_search(data, (int64_t)1 << 40, 65536, tags, live, q, 2, fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_OVERFLOW, "overflow");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, 0, 0, 1, ids, sc, cnt) == VECLITE_E_K, "k 0");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, VECLITE_MAX_K + 1, 0, 1, ids, sc, cnt) == VECLITE_E_K, "k max");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, 2, 3, 1, ids, sc, cnt) == VECLITE_E_METRIC, "metric 3");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, 2, -1, 1, ids, sc, cnt) == VECLITE_E_METRIC, "metric -1");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, bad_fo, ft, 2, 0, 1, ids, sc, cnt) == VECLITE_E_FILTER, "filter 2");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, 2, 0, 0, ids, sc, cnt) == VECLITE_E_CHUNKS, "chunks 0");
    CHECK(veclite_search(data, 4, 2, tags, live, q, 2, fo, ft, 2, 0, VECLITE_MAX_CHUNKS + 1, ids, sc, cnt) == VECLITE_E_CHUNKS, "chunks max");
    CHECK(search1(data, 4, 2, tags, live, nanq, 2, 0, 0, 0, 1, ids, sc, cnt) == VECLITE_E_QUERY_NONFINITE, "nan query");
    CHECK(search1(data, 4, 2, tags, live, infq, 2, 0, 0, 0, 1, ids, sc, cnt) == VECLITE_E_QUERY_NONFINITE, "inf query");
    CHECK(search1(data, 4, 2, tags, live, zeroq, 2, 2, 0, 0, 1, ids, sc, cnt) == VECLITE_E_QUERY_ZERO, "zero query cosine");
    CHECK(search1(data, 4, 2, tags, live, zeroq, 2, 0, 0, 0, 1, ids, sc, cnt) == VECLITE_OK, "zero query l2 ok");

    float nandata[8] = {1, 2, NAN, 4, 5, 6, 7, 8};
    uint8_t dead1[4] = {1, 0, 1, 1};
    cnt[0] = 99;
    CHECK(search1(nandata, 4, 2, tags, live, q, 2, 0, 0, 0, 2, ids, sc, cnt) == VECLITE_E_SCORE_NONFINITE, "nan row");
    CHECK(cnt[0] == 0, "count reset on error");
    CHECK(search1(nandata, 4, 2, tags, dead1, q, 2, 0, 0, 0, 2, ids, sc, cnt) == VECLITE_OK, "dead nan row skipped");
    CHECK(veclite_validate_rows(nandata, 4, 2, 0, &bad) == VECLITE_E_DATA_NONFINITE && bad == 1, "validate nan row");
    float zrow[4] = {1, 1, 0, 0}, huge[2] = {3e38f, 3e38f};
    CHECK(veclite_validate_rows(zrow, 2, 2, 2, &bad) == VECLITE_E_DATA_ZERO && bad == 1, "validate zero cosine row");
    CHECK(veclite_validate_rows(zrow, 2, 2, 0, &bad) == VECLITE_OK && bad == -1, "validate zero l2 row ok");
    CHECK(veclite_validate_rows(huge, 1, 2, 0, &bad) == VECLITE_E_DATA_NONFINITE, "validate overflow norm");
    CHECK(veclite_validate_rows(NULL, 1, 2, 0, &bad) == VECLITE_E_NULL, "validate null");
    CHECK(veclite_validate_rows(data, 4, 2, 7, &bad) == VECLITE_E_METRIC, "validate metric");
    CHECK(veclite_scores(data, 4, 2, q, 2, 0, NULL) == VECLITE_E_NULL, "scores null");
    CHECK(veclite_scores(data, 4, 2, q, 0, 0, sc) == VECLITE_E_BATCH, "scores batch 0");
    CHECK(veclite_search(data, 0, 2, tags, live, q, 2, fo, ft, 3, 0, 4, ids, sc, cnt) == VECLITE_OK && cnt[0] == 0 && cnt[1] == 0, "empty table");
}

static void test_random(void) {
    srand(7);
    for (int trial = 0; trial < 300; trial++) {
        int64_t n = rand() % 200, dim = 1 + rand() % 40, k = 1 + rand() % 30, nc = 1 + rand() % 12;
        int64_t nq = 1 + rand() % 9;
        int32_t metric = trial % 3;
        float *data = malloc(sizeof(float) * (size_t)(n * dim + 1));
        int64_t *tags = malloc(sizeof(int64_t) * (size_t)(n + 1));
        uint8_t *live = malloc((size_t)(n + 1));
        float *qs = malloc(sizeof(float) * (size_t)(nq * dim));
        float *scores = malloc(sizeof(float) * (size_t)(nq * n + 1));
        float *one = malloc(sizeof(float) * (size_t)(n + 1));
        uint8_t fo[16];
        int64_t ft[16];
        int64_t ids[16 * 32], ids1[32];
        float sc[16 * 32], sc1[32];
        int64_t cnt[16], cnt1;
        cand ref[64];
        for (int64_t i = 0; i < n * dim; i++) data[i] = (float)(rand() % 3);
        for (int64_t i = 0; i < n; i++) {
            tags[i] = rand() % 3;
            live[i] = rand() % 5 != 0;
            data[i * dim] = data[i * dim] == 0 ? 1 : data[i * dim];
        }
        for (int64_t q = 0; q < nq; q++) {
            for (int64_t d = 0; d < dim; d++) qs[q * dim + d] = (float)(rand() % 3);
            qs[q * dim] = qs[q * dim] == 0 ? 1 : qs[q * dim];
            fo[q] = (uint8_t)(rand() % 2);
            ft[q] = rand() % 3;
        }
        int rc = veclite_search(data, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ids, sc, cnt);
        CHECK(rc == VECLITE_OK, "trial %d rc %d", trial, rc);
        CHECK(veclite_scores(data, n, dim, qs, nq, metric, scores) == VECLITE_OK || n == 0, "scores rc");
        for (int64_t q = 0; q < nq; q++) {
            CHECK(n == 0 || veclite_scores(data, n, dim, qs + q * dim, 1, metric, one) == VECLITE_OK, "scores single rc");
            for (int64_t i = 0; i < n; i++)
                CHECK(memcmp(&one[i], &scores[q * n + i], sizeof(float)) == 0, "trial %d batch score differs q %lld row %lld", trial, (long long)q, (long long)i);
            int64_t r = reference(scores + q * n, tags, live, n, k, fo[q], ft[q], ref);
            CHECK(cnt[q] == r, "trial %d q %lld count %lld want %lld", trial, (long long)q, (long long)cnt[q], (long long)r);
            for (int64_t j = 0; j < r && j < cnt[q]; j++)
                CHECK(ids[q * k + j] == ref[j].id && sc[q * k + j] == ref[j].score, "trial %d q %lld pos %lld id %lld want %lld",
                      trial, (long long)q, (long long)j, (long long)ids[q * k + j], (long long)ref[j].id);
            CHECK(search1(data, n, dim, tags, live, qs + q * dim, k, metric, fo[q], ft[q], nc, ids1, sc1, &cnt1) == VECLITE_OK, "single rc");
            CHECK(cnt1 == cnt[q] && memcmp(ids1, ids + q * k, sizeof(int64_t) * (size_t)cnt1) == 0, "trial %d q %lld batch != single", trial, (long long)q);
        }
        free(data);
        free(tags);
        free(live);
        free(qs);
        free(scores);
        free(one);
    }
}

int main(void) {
    test_errors();
    test_random();
    printf("c_host: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
