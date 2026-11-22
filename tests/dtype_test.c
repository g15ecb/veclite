#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vdb.h"

static int failures = 0, checks = 0;
static long bound_checks = 0, fallbacks = 0, cands_total = 0, searches = 0;
static double worst_ratio = 0, worst_eps = 0, worst_gap = 0, worst_approx = 0;
static int worst_metric = -1;
static float worst_err = 0;

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
        if (mode == 1 && i > 0 && below(3) == 0) {
            memcpy(r, x + below(i) * dim, sizeof(float) * (size_t)dim);
            r[below(dim)] += unit() * 1e-4f;
        } else if (mode == 2) {
            for (int64_t d = 0; d < dim; d++) r[d] = unit() * (d == 0 ? 50.0f : 0.01f);
        } else if (mode == 3 && i > 0) {
            memcpy(r, x, sizeof(float) * (size_t)dim);
            r[below(dim)] += unit() * 1e-3f;
        } else {
            for (int64_t d = 0; d < dim; d++) r[d] = unit();
        }
        r[0] += r[0] == 0.0f ? 0.5f : 0.0f;
    }
}

static void test_encode_validation(void) {
    float x[8] = {1, 2, 3, 4, 5, 6, 7, 8}, big[4] = {1, 65505.0f, 0, 0}, nanv[4] = {1, NAN, 0, 0};
    float aux[6];
    int8_t q8[8];
    _Float16 h[8];
    int64_t bad = 0;
    CHECK(vdb_encode(3, x, 2, 4, q8, aux, &bad) == VDB_E_DTYPE, "bad dtype");
    CHECK(vdb_encode(-1, x, 2, 4, q8, aux, &bad) == VDB_E_DTYPE, "neg dtype");
    CHECK(vdb_encode(VDB_I8, x, 2, 4, q8, NULL, &bad) == VDB_E_NULL, "i8 needs aux");
    CHECK(vdb_encode(VDB_F16, x, 2, 4, NULL, NULL, &bad) == VDB_E_NULL, "null dst");
    CHECK(vdb_encode(VDB_F16, x, 2, 4, h, NULL, NULL) == VDB_E_NULL, "null bad_row");
    CHECK(vdb_encode(VDB_F16, x, 2, 0, h, NULL, &bad) == VDB_E_DIM, "dim 0");
    CHECK(vdb_encode(VDB_F16, big, 1, 4, h, NULL, &bad) == VDB_E_RANGE && bad == 0, "f16 range");
    float edge[4] = {65504.0f, -65504.0f, 0, 1};
    CHECK(vdb_encode(VDB_F16, edge, 1, 4, h, NULL, &bad) == VDB_OK && (float)h[0] == 65504.0f && (float)h[1] == -65504.0f, "f16 max ok");
    CHECK(vdb_encode(VDB_I8, nanv, 1, 4, q8, aux, &bad) == VDB_E_DATA_NONFINITE && bad == 0, "i8 nan");
    CHECK(vdb_encode(VDB_I8, x, 2, 4, q8, aux, &bad) == VDB_OK && bad == -1, "i8 ok");
    CHECK(q8[3] == 127 && q8[7] == 127 && fabsf(aux[0] - 4.0f / 127) < 1e-7f, "i8 absmax scale");
    float zero[4] = {0};
    CHECK(vdb_encode(VDB_I8, zero, 1, 4, q8, aux, &bad) == VDB_OK && aux[0] == 0 && aux[2] == 0 && q8[0] == 0, "i8 zero row");
    float ids_dummy[4];
    int64_t ids[4], cnt[1];
    uint8_t live[2] = {1, 1};
    CHECK(vdb_search_ex(q8, VDB_I8, NULL, x, 2, 4, NULL, live, x, 1, NULL, NULL, 1, 0, 1, ids, ids_dummy, cnt, NULL) == VDB_E_NULL, "search i8 needs aux");
    CHECK(vdb_search_ex(q8, VDB_I8, aux, NULL, 2, 4, NULL, live, x, 1, NULL, NULL, 1, 0, 1, ids, ids_dummy, cnt, NULL) == VDB_E_NULL, "search i8 needs orig");
    CHECK(vdb_search_ex(q8, 7, aux, x, 2, 4, NULL, live, x, 1, NULL, NULL, 1, 0, 1, ids, ids_dummy, cnt, NULL) == VDB_E_DTYPE, "search bad dtype");
}

static void compare(const char *ctx, int64_t nq, int64_t k, const int64_t *ca, const int64_t *ia, const float *sa,
                    const int64_t *cb, const int64_t *ib, const float *sb) {
    for (int64_t q = 0; q < nq; q++) {
        CHECK(ca[q] == cb[q], "%s q %lld count %lld want %lld", ctx, (long long)q, (long long)ca[q], (long long)cb[q]);
        for (int64_t j = 0; j < ca[q] && j < cb[q]; j++)
            CHECK(ia[q * k + j] == ib[q * k + j] && memcmp(&sa[q * k + j], &sb[q * k + j], 4) == 0,
                  "%s q %lld pos %lld id %lld want %lld score %.9g want %.9g", ctx, (long long)q, (long long)j,
                  (long long)ia[q * k + j], (long long)ib[q * k + j], sa[q * k + j], sb[q * k + j]);
    }
}

static void test_random(int trials) {
    for (int t = 0; t < trials; t++) {
        int64_t n = below(300), dim = 1 + below(70), k = 1 + below(20), nq = 1 + below(9), nc = 1 + below(8);
        int32_t metric = (int32_t)below(3);
        int mode = (int)below(4);
        float *x = malloc(sizeof(float) * (size_t)(n * dim + 1)), *qs = malloc(sizeof(float) * (size_t)(nq * dim));
        float *dec = malloc(sizeof(float) * (size_t)(n * dim + 1)), *aux = malloc(sizeof(float) * (size_t)(3 * n + 3));
        _Float16 *h = malloc(sizeof(_Float16) * (size_t)(n * dim + 1));
        int8_t *q8 = malloc((size_t)(n * dim + 1));
        int64_t *tags = malloc(sizeof(int64_t) * (size_t)(n + 1));
        uint8_t *live = malloc((size_t)(n + 1)), fo[16];
        int64_t ft[16], bad, ia[16 * 20], ib[16 * 20], ca[16], cb[16], stats[2] = {0, 0};
        float sa[16 * 20], sb[16 * 20];
        fill(x, n, dim, mode);
        fill(qs, nq, dim, 0);
        for (int64_t i = 0; i < n; i++) {
            tags[i] = below(3);
            live[i] = below(6) != 0;
        }
        for (int64_t q = 0; q < nq; q++) {
            fo[q] = (uint8_t)below(2);
            ft[q] = below(3);
        }
        CHECK(vdb_encode(VDB_F16, x, n, dim, h, NULL, &bad) == VDB_OK || n == 0, "encode f16");
        CHECK(vdb_encode(VDB_I8, x, n, dim, q8, aux, &bad) == VDB_OK || n == 0, "encode i8");
        for (int64_t i = 0; i < n * dim; i++) dec[i] = (float)h[i];
        int rc;
        if (n > 0) {
            rc = vdb_search(dec, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ib, sb, cb);
            CHECK(rc == VDB_OK, "f32 rc %d", rc);
            rc = vdb_search_ex(h, VDB_F16, NULL, NULL, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ia, sa, ca, NULL);
            CHECK(rc == VDB_OK, "f16 rc %d", rc);
            compare("f16 == f32(decoded f16)", nq, k, ca, ia, sa, cb, ib, sb);
            rc = vdb_search(x, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ib, sb, cb);
            CHECK(rc == VDB_OK, "f32 orig rc %d", rc);
            rc = vdb_search_ex(q8, VDB_I8, aux, x, n, dim, tags, live, qs, nq, fo, ft, k, metric, nc, ia, sa, ca, stats);
            CHECK(rc == VDB_OK, "i8 rc %d", rc);
            compare("i8 rescored == f32(originals)", nq, k, ca, ia, sa, cb, ib, sb);
            fallbacks += stats[1];
            cands_total += stats[0];
            searches += nq;
            float *ap = malloc(sizeof(float) * (size_t)n), *ex = malloc(sizeof(float) * (size_t)n);
            double *ep = malloc(sizeof(double) * (size_t)n);
            for (int64_t q = 0; q < nq; q++) {
                CHECK(vdb_i8_bounds(q8, aux, x, n, dim, qs + q * dim, metric, ap, ep, ex) == VDB_OK, "bounds rc");
                for (int64_t i = 0; i < n; i++) {
                    double gap = fabs((double)ap[i] - (double)ex[i]);
                    bound_checks++;
                    CHECK(gap <= ep[i], "error bound violated metric %d row %lld approx %.9g exact %.9g eps %.3g", metric, (long long)i, ap[i], ex[i], ep[i]);
                    if (ep[i] > 0 && gap / ep[i] > worst_ratio) {
                        worst_ratio = gap / ep[i];
                        worst_eps = ep[i];
                        worst_gap = gap;
                        worst_metric = metric;
                        worst_err = aux[3 * i + 2];
                        worst_approx = ap[i];
                    }
                }
            }
            free(ap);
            free(ex);
            free(ep);
        }
        free(x);
        free(qs);
        free(dec);
        free(aux);
        free(h);
        free(q8);
        free(tags);
        free(live);
    }
}

static void test_overflow_fallback(void) {
    int64_t n = 400, dim = 32, k = 5, bad, ia[5], ib[5], ca[1], cb[1], stats[2] = {0, 0};
    float *x = malloc(sizeof(float) * (size_t)(n * dim)), *aux = malloc(sizeof(float) * (size_t)(3 * n)), q[32], sa[5], sb[5];
    int8_t *q8 = malloc((size_t)(n * dim));
    uint8_t *live = malloc((size_t)n);
    fill(x, n, dim, 3);
    fill(q, 1, dim, 0);
    memset(live, 1, (size_t)n);
    CHECK(vdb_encode(VDB_I8, x, n, dim, q8, aux, &bad) == VDB_OK, "encode");
    for (int32_t metric = 0; metric < 3; metric++) {
        CHECK(vdb_search(x, n, dim, NULL, live, q, 1, NULL, NULL, k, metric, 4, ib, sb, cb) == VDB_OK, "f32");
        CHECK(vdb_search_ex(q8, VDB_I8, aux, x, n, dim, NULL, live, q, 1, NULL, NULL, k, metric, 4, ia, sa, ca, stats) == VDB_OK, "i8");
        CHECK(stats[1] == 1, "near-duplicate rows must overflow the candidate buffer and fall back (metric %d, fallbacks %lld)", metric, (long long)stats[1]);
        compare("overflow fallback == f32", 1, k, ca, ia, sa, cb, ib, sb);
    }
    free(x);
    free(aux);
    free(q8);
    free(live);
}

int main(int argc, char **argv) {
    int trials = argc > 1 ? atoi(argv[1]) : 400;
    test_encode_validation();
    test_overflow_fallback();
    test_random(trials);
    printf("worst bound case: metric %d gap %.3g eps %.3g approx %.9g row err %.3g\n", worst_metric, worst_gap, worst_eps, worst_approx, worst_err);
    printf("dtype_test: %d checks, %d failures | bound checks %ld worst gap/eps %.3f | i8 candidates/query %.2f fallbacks %ld\n", checks, failures,
           bound_checks, worst_ratio, searches ? (double)cands_total / (double)searches : 0.0, fallbacks);
    return failures != 0;
}
