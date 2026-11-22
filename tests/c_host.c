#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vdb.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond, ...)                         \
    do {                                         \
        checks++;                                \
        if (!(cond)) {                           \
            failures++;                          \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);        \
            fprintf(stderr, "\n");               \
        }                                        \
    } while (0)

typedef struct {
    float score;
    int64_t id;
} cand;

static int cand_cmp(const void *a, const void *b) {
    const cand *x = a, *y = b;
    if (x->score < y->score) return -1;
    if (x->score > y->score) return 1;
    return (x->id > y->id) - (x->id < y->id);
}

static int64_t reference(const float *scores, const int64_t *tags, const uint8_t *live, int64_t n, int64_t k,
                         int filter_on, int64_t tag, cand *out) {
    cand *all = malloc(sizeof(cand) * (size_t)(n > 0 ? n : 1));
    int64_t m = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!live[i]) continue;
        if (filter_on && tags[i] != tag) continue;
        all[m].score = scores[i];
        all[m].id = i;
        m++;
    }
    qsort(all, (size_t)m, sizeof(cand), cand_cmp);
    int64_t r = m < k ? m : k;
    memcpy(out, all, sizeof(cand) * (size_t)r);
    free(all);
    return r;
}

static void test_errors(void) {
    float data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    int64_t tags[4] = {0, 1, 0, 1};
    uint8_t live[4] = {1, 1, 1, 1};
    float q[2] = {1, 1};
    int64_t ids[8];
    float sc[8];
    int64_t cnt = 99;
    int64_t bad = 0;
    float nanq[2] = {NAN, 1};
    float infq[2] = {INFINITY, 1};
    float zeroq[2] = {0, 0};

    CHECK(vdb_abi_version() == 1, "abi");
    CHECK(vdb_search(NULL, 4, 2, tags, live, q, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_NULL, "null data");
    CHECK(vdb_search(data, 4, 2, tags, NULL, q, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_NULL, "null live");
    CHECK(vdb_search(data, 4, 2, NULL, live, q, 2, 0, 1, 0, 1, ids, sc, &cnt) == VDB_E_NULL, "null tags with filter");
    CHECK(vdb_search(data, 4, 2, NULL, live, q, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_OK, "null tags without filter ok");
    CHECK(vdb_search(data, 4, 2, tags, live, NULL, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_NULL, "null query");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 2, 0, 0, 0, 1, NULL, sc, &cnt) == VDB_E_NULL, "null ids");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 2, 0, 0, 0, 1, ids, NULL, &cnt) == VDB_E_NULL, "null scores");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 2, 0, 0, 0, 1, ids, sc, NULL) == VDB_E_NULL, "null count");
    CHECK(vdb_search(data, -1, 2, tags, live, q, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_N, "neg n");
    CHECK(vdb_search(data, 4, 0, tags, live, q, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_DIM, "dim 0");
    CHECK(vdb_search(data, 4, VDB_MAX_DIM + 1, tags, live, q, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_DIM, "dim max");
    CHECK(vdb_search(data, (int64_t)1 << 40, 65536, tags, live, q, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_OVERFLOW, "overflow");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 0, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_K, "k 0");
    CHECK(vdb_search(data, 4, 2, tags, live, q, VDB_MAX_K + 1, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_K, "k max");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 2, 3, 0, 0, 1, ids, sc, &cnt) == VDB_E_METRIC, "metric 3");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 2, -1, 0, 0, 1, ids, sc, &cnt) == VDB_E_METRIC, "metric -1");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 2, 0, 2, 0, 1, ids, sc, &cnt) == VDB_E_FILTER, "filter 2");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 2, 0, 0, 0, 0, ids, sc, &cnt) == VDB_E_CHUNKS, "chunks 0");
    CHECK(vdb_search(data, 4, 2, tags, live, q, 2, 0, 0, 0, VDB_MAX_CHUNKS + 1, ids, sc, &cnt) == VDB_E_CHUNKS, "chunks max");
    CHECK(vdb_search(data, 4, 2, tags, live, nanq, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_QUERY_NONFINITE, "nan query");
    CHECK(vdb_search(data, 4, 2, tags, live, infq, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_E_QUERY_NONFINITE, "inf query");
    CHECK(vdb_search(data, 4, 2, tags, live, zeroq, 2, 2, 0, 0, 1, ids, sc, &cnt) == VDB_E_QUERY_ZERO, "zero query cosine");
    CHECK(vdb_search(data, 4, 2, tags, live, zeroq, 2, 0, 0, 0, 1, ids, sc, &cnt) == VDB_OK, "zero query l2 ok");

    float bigdata[8] = {1, 2, NAN, 4, 5, 6, 7, 8};
    cnt = 99;
    CHECK(vdb_search(bigdata, 4, 2, tags, live, q, 2, 0, 0, 0, 2, ids, sc, &cnt) == VDB_E_SCORE_NONFINITE, "nan row");
    CHECK(cnt == 0, "count reset on error, got %lld", (long long)cnt);
    uint8_t live_nan_dead[4] = {1, 0, 1, 1};
    CHECK(vdb_search(bigdata, 4, 2, tags, live_nan_dead, q, 2, 0, 0, 0, 2, ids, sc, &cnt) == VDB_OK, "dead nan row skipped");
    CHECK(vdb_validate_rows(bigdata, 4, 2, 0, &bad) == VDB_E_DATA_NONFINITE && bad == 1, "validate nan row %lld", (long long)bad);
    float zrow[4] = {1, 1, 0, 0};
    CHECK(vdb_validate_rows(zrow, 2, 2, 2, &bad) == VDB_E_DATA_ZERO && bad == 1, "validate zero cosine row");
    CHECK(vdb_validate_rows(zrow, 2, 2, 0, &bad) == VDB_OK && bad == -1, "validate zero l2 row ok");
    float hugerow[2] = {3e38f, 3e38f};
    CHECK(vdb_validate_rows(hugerow, 1, 2, 0, &bad) == VDB_E_DATA_NONFINITE, "validate overflow norm");
    CHECK(vdb_validate_rows(NULL, 1, 2, 0, &bad) == VDB_E_NULL, "validate null");
    CHECK(vdb_validate_rows(data, 4, 2, 7, &bad) == VDB_E_METRIC, "validate metric");
    CHECK(vdb_scores(data, 4, 2, q, 0, NULL) == VDB_E_NULL, "scores null");
    CHECK(vdb_search(data, 0, 2, tags, live, q, 3, 0, 0, 0, 4, ids, sc, &cnt) == VDB_OK && cnt == 0, "empty table");
}

static void test_random(void) {
    srand(7);
    int metrics[3] = {VDB_METRIC_L2, VDB_METRIC_DOT, VDB_METRIC_COSINE};
    for (int trial = 0; trial < 300; trial++) {
        int64_t n = rand() % 200;
        int64_t dim = 1 + rand() % 40;
        int64_t k = 1 + rand() % 30;
        int64_t nchunks = 1 + rand() % 12;
        int metric = metrics[trial % 3];
        int filter_on = rand() % 2;
        int64_t tag = rand() % 3;
        float *data = malloc(sizeof(float) * (size_t)(n * dim + 1));
        int64_t *tags = malloc(sizeof(int64_t) * (size_t)(n + 1));
        uint8_t *live = malloc((size_t)(n + 1));
        float *q = malloc(sizeof(float) * (size_t)dim);
        float *scores = malloc(sizeof(float) * (size_t)(n + 1));
        for (int64_t i = 0; i < n * dim; i++) data[i] = (float)(rand() % 3);
        for (int64_t i = 0; i < n; i++) {
            tags[i] = rand() % 3;
            live[i] = rand() % 5 != 0;
            int any = 0;
            for (int64_t d = 0; d < dim; d++) any |= data[i * dim + d] != 0;
            if (!any) data[i * dim] = 1;
        }
        for (int64_t d = 0; d < dim; d++) q[d] = (float)(rand() % 3);
        q[0] = q[0] == 0 ? 1 : q[0];
        int64_t ids[64];
        float sc[64];
        cand ref[64];
        int64_t cnt = -1;
        int rc = vdb_search(data, n, dim, tags, live, q, k, metric, filter_on, tag, nchunks, ids, sc, &cnt);
        CHECK(rc == VDB_OK, "trial %d rc %d", trial, rc);
        CHECK(vdb_scores(data, n, dim, q, metric, scores) == VDB_OK, "scores rc");
        int64_t r = reference(scores, tags, live, n, k, filter_on, tag, ref);
        CHECK(cnt == r, "trial %d count %lld want %lld", trial, (long long)cnt, (long long)r);
        for (int64_t j = 0; j < r && j < cnt; j++) {
            CHECK(ids[j] == ref[j].id && sc[j] == ref[j].score, "trial %d pos %lld id %lld want %lld", trial,
                  (long long)j, (long long)ids[j], (long long)ref[j].id);
        }
        free(data);
        free(tags);
        free(live);
        free(q);
        free(scores);
    }
}

int main(void) {
    test_errors();
    test_random();
    printf("c_host: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
