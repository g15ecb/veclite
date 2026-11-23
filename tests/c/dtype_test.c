#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "veclite.h"

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
static float unit(void) { return (float)((double)(rnd() >> 11) / 9007199254740992.0) * 2.0f - 1.0f; }

static void fill(float *x, int64_t n, int64_t dim, int mode) {
    for (int64_t i = 0; i < n; i++) {
        float *r = x + i * dim;
        for (int64_t d = 0; d < dim; d++) r[d] = unit() * (mode == 1 ? 1000.0f : (mode == 2 ? 1e-3f : 1.0f));
        if (mode == 3 && i > 0) {
            memcpy(r, x, sizeof(float) * (size_t)dim);
            r[below(dim)] += unit() * 1e-3f;
        }
        r[0] += r[0] == 0.0f ? 0.5f : 0.0f;
    }
}

static void test_encode_validation(void) {
    float x[8] = {1, 2, 3, 4, 5, 6, 7, 8}, big[4] = {1, 65505.0f, 0, 0}, neg[4] = {1, -65505.0f, 0, 0};
    float nanv[4] = {1, NAN, 0, 0}, edge[4] = {65504.0f, -65504.0f, 0, 1};
    _Float16 h[8];
    float f[8];
    int64_t bad = 0;
    CHECK(veclite_encode(2, x, 2, 4, h, &bad) == VECLITE_E_DTYPE, "int8 removed");
    CHECK(veclite_encode(-1, x, 2, 4, h, &bad) == VECLITE_E_DTYPE, "neg dtype");
    CHECK(veclite_encode(VECLITE_F16, x, 2, 4, NULL, &bad) == VECLITE_E_NULL, "null dst");
    CHECK(veclite_encode(VECLITE_F16, x, 2, 4, h, NULL) == VECLITE_E_NULL, "null bad_row");
    CHECK(veclite_encode(VECLITE_F16, NULL, 2, 4, h, &bad) == VECLITE_E_NULL, "null src");
    CHECK(veclite_encode(VECLITE_F16, x, 2, 0, h, &bad) == VECLITE_E_DIM, "dim 0");
    CHECK(veclite_encode(VECLITE_F16, x, -1, 4, h, &bad) == VECLITE_E_N, "neg n");
    CHECK(veclite_encode(VECLITE_F16, big, 1, 4, h, &bad) == VECLITE_E_RANGE && bad == 0, "f16 range +");
    CHECK(veclite_encode(VECLITE_F16, neg, 1, 4, h, &bad) == VECLITE_E_RANGE && bad == 0, "f16 range -");
    CHECK(veclite_encode(VECLITE_F16, nanv, 1, 4, h, &bad) == VECLITE_E_DATA_NONFINITE && bad == 0, "f16 nan");
    CHECK(veclite_encode(VECLITE_F16, edge, 1, 4, h, &bad) == VECLITE_OK && (float)h[0] == 65504.0f && (float)h[1] == -65504.0f, "f16 max ok");
    CHECK(veclite_encode(VECLITE_F32, x, 2, 4, f, &bad) == VECLITE_OK && memcmp(f, x, sizeof f) == 0 && bad == -1, "f32 copy");
    float q[4] = {1, 0, 0, 0}, sc[1];
    int64_t ids[1], cnt[1];
    uint8_t live[2] = {1, 1};
    CHECK(veclite_search_ex(h, 2, 2, 4, NULL, live, q, 1, NULL, NULL, 1, 0, 1, ids, sc, cnt) == VECLITE_E_DTYPE, "search int8 removed");
}

static void compare(const char *ctx, int64_t nq, int64_t k, const int64_t *ca, const int64_t *ia, const float *sa,
                    const int64_t *cb, const int64_t *ib, const float *sb) {
    for (int64_t q = 0; q < nq; q++) {
        CHECK(ca[q] == cb[q], "%s q %lld count %lld want %lld", ctx, (long long)q, (long long)ca[q], (long long)cb[q]);
        for (int64_t j = 0; j < ca[q] && j < cb[q]; j++)
            CHECK(ia[q * k + j] == ib[q * k + j] && memcmp(&sa[q * k + j], &sb[q * k + j], 4) == 0,
                  "%s q %lld pos %lld id %lld want %lld", ctx, (long long)q, (long long)j, (long long)ia[q * k + j], (long long)ib[q * k + j]);
    }
}

static void test_random(int trials) {
    for (int t = 0; t < trials; t++) {
        int64_t n = 1 + below(300), dim = 1 + below(70), k = 1 + below(20), nq = 1 + below(9), nc = 1 + below(8);
        int32_t metric = (int32_t)below(3);
        float *x = malloc(sizeof(float) * (size_t)(n * dim)), *qs = malloc(sizeof(float) * (size_t)(nq * dim));
        float *dec = malloc(sizeof(float) * (size_t)(n * dim));
        _Float16 *h = malloc(sizeof(_Float16) * (size_t)(n * dim));
        int64_t *tags = malloc(sizeof(int64_t) * (size_t)n);
        uint8_t *live = malloc((size_t)n), fo[16];
        int64_t ft[16], bad, ia[16 * 20], ib[16 * 20], ca[16], cb[16];
        float sa[16 * 20], sb[16 * 20];
        fill(x, n, dim, (int)below(4));
        fill(qs, nq, dim, 0);
        for (int64_t i = 0; i < n; i++) {
            tags[i] = below(3);
            live[i] = below(6) != 0;
        }
        for (int64_t q = 0; q < nq; q++) {
            fo[q] = (uint8_t)below(2);
            ft[q] = below(3);
        }
        CHECK(veclite_encode(VECLITE_F16, x, n, dim, h, &bad) == VECLITE_OK, "encode f16");
        for (int64_t i = 0; i < n * dim; i++) {
            dec[i] = (float)h[i];
            CHECK(dec[i] == (float)(_Float16)x[i], "encode must round like the C compiler's float->half");
        }
        CHECK(veclite_search(dec, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ib, sb, cb) == VECLITE_OK, "f32 rc");
        CHECK(veclite_search_ex(h, VECLITE_F16, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ia, sa, ca) == VECLITE_OK, "f16 rc");
        compare("f16 == f32(decoded f16)", nq, k, ca, ia, sa, cb, ib, sb);
        CHECK(veclite_search_ex(x, VECLITE_F32, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ia, sa, ca) == VECLITE_OK, "f32 ex rc");
        CHECK(veclite_search(x, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ib, sb, cb) == VECLITE_OK, "f32 rc");
        compare("search_ex(f32) == search", nq, k, ca, ia, sa, cb, ib, sb);
        free(x);
        free(qs);
        free(dec);
        free(h);
        free(tags);
        free(live);
    }
}

int main(int argc, char **argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 400;
    test_encode_validation();
    test_random(trials);
    printf("dtype_test: %d checks, %d failures\n", checks, failures);
    return failures != 0;
}
