#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "veclite_store.h"

#define DIM 8
#define EXT 4
#define MAXOPS 4096
#define MAXROWS 1024
#define NREAD 3

static int failures = 0, checks = 0;
static int32_t storage = VECLITE_F32;
static long io_crashes = 0, idle_crashes = 0, restarts = 0, wal_over_ckpt = 0, lost_unacked = 0, searches = 0;

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

static void vec_of(int64_t id, float *v) {
    for (int d = 0; d < DIM; d++) v[d] = ((float)((id * 31 + d * 7) % 13) - 6.0f + (d == 0 ? 0.5f : 0.0f)) / 3.0f + (float)(id % 7) * 1e-4f;
}

static void stored_form(float *v) {
    if (storage == VECLITE_F16)
        for (int d = 0; d < DIM; d++) v[d] = (float)(_Float16)v[d];
}
static int64_t tag_of(int64_t id) { return id % 5; }

typedef struct {
    int64_t txn, id;
    int del;
} op;

static op ops[MAXOPS];
static int nops = 0;
static int64_t written = 0, acked = 0;

typedef struct {
    int64_t n;
    uint8_t live[MAXROWS];
} state;

static void state_at(int64_t m, state *s) {
    s->n = 0;
    memset(s->live, 0, sizeof s->live);
    for (int i = 0; i < nops && ops[i].txn <= m; i++) {
        if (ops[i].del) s->live[ops[i].id] = 0;
        else {
            s->live[ops[i].id] = 1;
            if (ops[i].id + 1 > s->n) s->n = ops[i].id + 1;
        }
    }
}

static int cand_cmp(const void *a, const void *b) {
    const float *x = a, *y = b;
    if (x[0] != y[0]) return x[0] < y[0] ? -1 : 1;
    return (x[1] > y[1]) - (x[1] < y[1]);
}

static void check_view(int64_t db, int64_t r, int64_t mark, const char *ctx) {
    static uint8_t live[MAXROWS];
    static int64_t tags[MAXROWS];
    static float vecs[MAXROWS * DIM], want[DIM];
    int64_t n = -1;
    state s;
    state_at(mark, &s);
    int rc = veclite_reader_view(db, r, MAXROWS, live, tags, vecs, &n);
    CHECK(rc == VECLITE_OK, "%s view rc %d", ctx, rc);
    CHECK(n == s.n, "%s readerSnapshot rows %lld want %lld (mark %lld)", ctx, (long long)n, (long long)s.n, (long long)mark);
    if (rc != VECLITE_OK || n != s.n) return;
    for (int64_t i = 0; i < n; i++) {
        vec_of(i, want);
        stored_form(want);
        CHECK(live[i] == s.live[i], "%s readerSnapshot live row %lld got %d want %d (mark %lld)", ctx, (long long)i, live[i], s.live[i], (long long)mark);
        CHECK(tags[i] == tag_of(i) && memcmp(vecs + i * DIM, want, sizeof want) == 0, "%s readerSnapshot content row %lld", ctx, (long long)i);
    }
    if (n == 0 || below(3) != 0) return;
    float q[2 * DIM];
    for (int i = 0; i < 2 * DIM; i++) q[i] = (float)(below(9) - 4);
    q[0] = q[DIM] = 1.0f;
    uint8_t fo[2] = {(uint8_t)below(2), (uint8_t)below(2)};
    int64_t ft[2] = {below(5), below(5)}, k = 1 + below(6), ids[16], cnt[2];
    float sc[16], scores[2 * MAXROWS], cands[2 * MAXROWS];
    int32_t metric = (int32_t)below(2);
    rc = veclite_reader_search(db, r, q, 2, fo, ft, k, metric, 1 + below(4), ids, sc, cnt);
    searches++;
    CHECK(rc == VECLITE_OK, "%s search rc %d", ctx, rc);
    CHECK(veclite_scores(vecs, n, DIM, q, 2, metric, scores) == VECLITE_OK, "%s scores", ctx);
    for (int qi = 0; qi < 2 && rc == VECLITE_OK; qi++) {
        int m = 0;
        for (int64_t i = 0; i < n; i++)
            if (live[i] && (!fo[qi] || tags[i] == ft[qi])) {
                cands[2 * m] = scores[qi * n + i];
                cands[2 * m + 1] = (float)i;
                m++;
            }
        qsort(cands, (size_t)m, 2 * sizeof(float), cand_cmp);
        int64_t want_n = m < k ? m : k;
        CHECK(cnt[qi] == want_n, "%s search count %lld want %lld", ctx, (long long)cnt[qi], (long long)want_n);
        for (int64_t j = 0; j < want_n && j < cnt[qi]; j++)
            CHECK(ids[qi * k + j] == (int64_t)cands[2 * j + 1] && sc[qi * k + j] == cands[2 * j], "%s search pos %lld", ctx, (long long)j);
    }
}

typedef struct {
    int64_t sim, db;
    int64_t active[NREAD], mark[NREAD];
    int64_t pend_id[64];
    int pend_del[64], npend;
    state cur;
} world;

static void reopen(world *w, const char *ctx) {
    veclite_close(w->db);
    w->db = 0;
    CHECK(veclite_sim_crash(w->sim, rnd()) == VECLITE_OK, "crash");
    int rc = veclite_open(NULL, DIM, EXT, storage, w->sim, &w->db);
    CHECK(rc == VECLITE_OK, "%s reopen rc %d", ctx, rc);
    if (rc != VECLITE_OK) {
        w->db = 0;
        return;
    }
    int64_t info[VECLITE_INFO_LEN];
    veclite_info(w->db, info);
    int64_t rec = info[VECLITE_INFO_NTX];
    if (getenv("TRACE")) fprintf(stderr, "REOPEN %s rows %lld ntx %lld hdr %lld frames %lld salt %lld main %lld\n", ctx, (long long)info[0], (long long)info[1], (long long)info[3], (long long)info[4], (long long)info[5], (long long)info[6]);
    wal_over_ckpt += info[VECLITE_INFO_HDR_TXN] > 0 && info[VECLITE_INFO_WAL_FRAMES] > 0;
    lost_unacked += rec < written;
    CHECK(rec >= acked, "%s ackedDurable recovered %lld < acked %lld", ctx, (long long)rec, (long long)acked);
    CHECK(rec <= written, "%s recovered %lld > written %lld", ctx, (long long)rec, (long long)written);
    CHECK(info[VECLITE_INFO_PUBLISHED] == rec, "%s published", ctx);
    while (nops > 0 && ops[nops - 1].txn > rec) nops--;
    written = rec < written ? rec : written;
    acked = written;
    for (int i = 0; i < NREAD; i++) w->active[i] = -1;
    w->npend = 0;
    state_at(written, &w->cur);
    int64_t r;
    if (veclite_reader_begin(w->db, &r) == VECLITE_OK) {
        check_view(w->db, r, written, "recoveryAtomic");
        veclite_reader_end(w->db, r);
    }
}

static int run_workload(uint64_t seed, int steps) {
    rng = seed * 2654435761u + 1;
    nops = 0;
    written = acked = 0;
    world w = {0};
    CHECK(veclite_sim_new(&w.sim) == VECLITE_OK, "sim");
    CHECK(veclite_open(NULL, DIM, EXT, storage, w.sim, &w.db) == VECLITE_OK, "open");
    for (int i = 0; i < NREAD; i++) w.active[i] = -1;
    float v[DIM];
    for (int step = 0; step < steps && w.db; step++) {
        if (below(5) == 0) veclite_sim_set_budget(w.sim, below(4) == 0 ? below(40) : below(4));
        int64_t act = below(100), rc = VECLITE_OK, id;
        if (act < 35 && w.cur.n + w.npend < MAXROWS - 1 && w.npend < 60) {
            int64_t nid = w.cur.n;
            vec_of(nid, v);
            rc = veclite_append(w.db, v, tag_of(nid), &id);
            CHECK(rc != VECLITE_OK || id == nid, "append id %lld want %lld", (long long)id, (long long)nid);
            if (rc == VECLITE_OK) {
                w.pend_id[w.npend] = nid;
                w.pend_del[w.npend++] = 0;
                w.cur.live[nid] = 1;
                w.cur.n++;
            }
        } else if (act < 50 && w.npend < 60) {
            int64_t cand = w.cur.n ? below(w.cur.n) : 0;
            if (w.cur.n && w.cur.live[cand]) {
                rc = veclite_delete(w.db, cand);
                if (rc == VECLITE_OK) {
                    w.pend_id[w.npend] = cand;
                    w.pend_del[w.npend++] = 1;
                    w.cur.live[cand] = 0;
                }
            } else if (w.cur.n) {
                CHECK(veclite_delete(w.db, cand) == VECLITE_E_ID, "delete dead row must fail");
            }
        } else if (act < 68) {
            if (w.npend) {
                written++;
                for (int i = 0; i < w.npend; i++) ops[nops++] = (op){written, w.pend_id[i], w.pend_del[i]};
                w.npend = 0;
            }
            int64_t txn = -1;
            rc = veclite_commit(w.db, &txn);
            if (rc == VECLITE_OK) {
                acked = written;
                CHECK(txn == written, "commit txn %lld want %lld", (long long)txn, (long long)written);
            }
        } else if (act < 78) {
            if (w.npend == 0) {
                int64_t before[VECLITE_INFO_LEN], after[VECLITE_INFO_LEN];
                veclite_info(w.db, before);
                rc = veclite_checkpoint(w.db, NULL);
                veclite_info(w.db, after);
                restarts += after[VECLITE_INFO_SALT] != before[VECLITE_INFO_SALT];
            }
        } else if (act < 88) {
            int slot = (int)below(NREAD);
            if (w.active[slot] < 0) {
                int64_t r;
                rc = veclite_reader_begin(w.db, &r);
                if (rc == VECLITE_OK) {
                    w.active[slot] = r;
                    w.mark[slot] = acked;
                }
            } else {
                veclite_reader_end(w.db, w.active[slot]);
                w.active[slot] = -1;
            }
        } else if (act < 92) {
            veclite_sim_set_budget(w.sim, -1);
            idle_crashes++;
            reopen(&w, "idle-crash");
            continue;
        }
        if (getenv("TRACE")) {
            int64_t ti[VECLITE_INFO_LEN];
            veclite_info(w.db, ti);
            fprintf(stderr, "step %d act %lld rc %lld npend %d cur.n %lld written %lld acked %lld | rows %lld ntx %lld pub %lld hdr %lld frames %lld salt %lld main %lld mut %lld\n", step, (long long)act, (long long)rc, w.npend, (long long)w.cur.n, (long long)written, (long long)acked,
                    (long long)ti[0], (long long)ti[1], (long long)ti[2], (long long)ti[3], (long long)ti[4], (long long)ti[5], (long long)ti[6], (long long)veclite_sim_mutations(w.sim));
        }
        if (rc == VECLITE_E_IO) {
            int64_t info[VECLITE_INFO_LEN];
            veclite_info(w.db, info);
            CHECK(info[VECLITE_INFO_POISONED] == 1, "io error must poison");
            CHECK(veclite_commit(w.db, NULL) == VECLITE_E_STATE, "poisoned db must refuse writes");
            io_crashes++;
            reopen(&w, "io-crash");
            continue;
        }
        CHECK(rc == VECLITE_OK, "step %d act %lld rc %lld", step, (long long)act, (long long)rc);
        veclite_sim_set_budget(w.sim, -1);
        for (int i = 0; i < NREAD; i++)
            if (w.active[i] >= 0) {
                int64_t mark = -1;
                veclite_reader_mark(w.db, w.active[i], &mark);
                CHECK(mark == w.mark[i], "reader mark %lld want %lld", (long long)mark, (long long)w.mark[i]);
                check_view(w.db, w.active[i], w.mark[i], "readerSnapshot");
            }
        int64_t info[VECLITE_INFO_LEN];
        veclite_info(w.db, info);
        CHECK(info[VECLITE_INFO_PUBLISHED] == acked && info[VECLITE_INFO_HDR_TXN] <= info[VECLITE_INFO_PUBLISHED], "publishedOrder");
    }
    if (w.db) veclite_close(w.db);
    veclite_sim_free(w.sim);
    return failures;
}

static void test_api(void) {
    int64_t sim, db, id, r;
    float v[DIM], nanv[DIM];
    vec_of(0, v);
    memcpy(nanv, v, sizeof v);
    nanv[3] = NAN;
    CHECK(veclite_open(NULL, DIM, EXT, storage, 0, &db) == VECLITE_E_NULL, "open null path");
    CHECK(veclite_sim_new(NULL) == VECLITE_E_NULL, "sim null");
    veclite_sim_new(&sim);
    CHECK(veclite_open(NULL, 0, EXT, storage, sim, &db) == VECLITE_E_ARG, "create needs dim");
    CHECK(veclite_open(NULL, -1, EXT, storage, sim, &db) == VECLITE_E_ARG, "neg dim");
    CHECK(veclite_open(NULL, DIM, -1, storage, sim, &db) == VECLITE_E_ARG, "neg extent");
    CHECK(veclite_open(NULL, DIM, EXT, storage, sim, NULL) == VECLITE_E_NULL, "null out");
    CHECK(veclite_open(NULL, DIM, EXT, storage, sim, &db) == VECLITE_OK, "open");
    CHECK(veclite_append(db, nanv, 0, &id) == VECLITE_E_DATA_NONFINITE, "nan row");
    CHECK(veclite_append(db, NULL, 0, &id) == VECLITE_E_NULL, "null vec");
    CHECK(veclite_append(db, v, 0, NULL) == VECLITE_E_NULL, "null id");
    CHECK(veclite_delete(db, 0) == VECLITE_E_ID, "delete missing");
    CHECK(veclite_delete(db, -1) == VECLITE_E_ID, "delete neg");
    CHECK(veclite_append(db, v, 3, &id) == VECLITE_OK && id == 0, "append");
    CHECK(veclite_delete(db, 0) == VECLITE_OK, "delete pending row");
    CHECK(veclite_delete(db, 0) == VECLITE_E_ID, "double delete");
    CHECK(veclite_checkpoint(db, NULL) == VECLITE_E_STATE, "checkpoint with pending");
    CHECK(veclite_commit(db, NULL) == VECLITE_OK, "commit");
    CHECK(veclite_reader_end(db, 0) == VECLITE_E_READER, "end inactive");
    CHECK(veclite_reader_end(db, 99) == VECLITE_E_READER, "end out of range");
    CHECK(veclite_reader_begin(db, NULL) == VECLITE_E_NULL, "begin null");
    for (int i = 0; i < 64; i++) CHECK(veclite_reader_begin(db, &r) == VECLITE_OK, "begin %d", i);
    CHECK(veclite_reader_begin(db, &r) == VECLITE_E_READER, "reader slots full");
    float q[DIM] = {1};
    int64_t ids[4], cnt;
    float sc[4];
    CHECK(veclite_reader_search(db, 0, q, 0, NULL, NULL, 2, 0, 1, ids, sc, &cnt) == VECLITE_E_BATCH, "search batch 0");
    CHECK(veclite_reader_search(db, 0, q, 1, NULL, NULL, 0, 0, 1, ids, sc, &cnt) == VECLITE_E_K, "search k 0");
    CHECK(veclite_reader_search(db, 0, NULL, 1, NULL, NULL, 2, 0, 1, ids, sc, &cnt) == VECLITE_E_NULL, "search null q");
    CHECK(veclite_reader_search(db, 0, q, 1, NULL, NULL, 2, 0, 1, ids, sc, &cnt) == VECLITE_OK && cnt == 0, "search deleted row");
    veclite_close(db);
    int32_t drc = veclite_open(NULL, DIM + 1, EXT, storage, sim, &db);
    CHECK(drc == VECLITE_E_DIM, "dim mismatch rc %d", drc);
    drc = veclite_open(NULL, 0, 0, -1, sim, &db);
    CHECK(drc == VECLITE_OK, "open existing with dim 0 rc %d", drc);
    int64_t dinfo[VECLITE_INFO_LEN];
    veclite_info(db, dinfo);
    CHECK(dinfo[VECLITE_INFO_DTYPE] == storage, "stored dtype %lld want %d", (long long)dinfo[VECLITE_INFO_DTYPE], storage);
    veclite_close(db);
    CHECK(veclite_open(NULL, 0, 0, 1 - storage, sim, &db) == VECLITE_E_DTYPE, "dtype mismatch");
    CHECK(veclite_open(NULL, 0, 0, 2, sim, &db) == VECLITE_E_DTYPE, "invalid storage (int8 removed)");
    CHECK(veclite_open(NULL, 0, 0, -2, sim, &db) == VECLITE_E_DTYPE, "invalid storage neg");
    CHECK(veclite_open(NULL, 0, 0, storage, sim, &db) == VECLITE_OK, "reopen same dtype");
    float huge[DIM] = {70000.0f};
    int64_t hid;
    CHECK(veclite_append(db, huge, 0, &hid) == (storage == VECLITE_F16 ? VECLITE_E_RANGE : VECLITE_OK), "f16 range on append");
    veclite_close(db);
    veclite_sim_free(sim);

    char path[] = "/tmp/vdb-store-test-XXXXXX";
    int fd = mkstemp(path);
    close(fd);
    unlink(path);
    char wal[64];
    snprintf(wal, sizeof wal, "%s-wal", path);
    CHECK(veclite_open(path, DIM, EXT, storage, 0, &db) == VECLITE_OK, "real open");
    for (int64_t i = 0; i < 10; i++) {
        vec_of(i, v);
        CHECK(veclite_append(db, v, tag_of(i), &id) == VECLITE_OK, "real append");
    }
    CHECK(veclite_delete(db, 3) == VECLITE_OK && veclite_commit(db, NULL) == VECLITE_OK, "real commit");
    int64_t info[VECLITE_INFO_LEN];
    CHECK(veclite_checkpoint(db, NULL) == VECLITE_OK, "real checkpoint");
    veclite_info(db, info);
    CHECK(info[VECLITE_INFO_HDR_TXN] == 1 && info[VECLITE_INFO_WAL_FRAMES] == 0, "real checkpoint restarted wal");
    veclite_close(db);
    CHECK(veclite_open(path, DIM, EXT, storage, 0, &db) == VECLITE_OK, "real reopen");
    veclite_info(db, info);
    CHECK(info[VECLITE_INFO_ROWS] == 10 && info[VECLITE_INFO_NTX] == 1, "real reopen rows %lld", (long long)info[VECLITE_INFO_ROWS]);
    veclite_reader_begin(db, &r);
    float q2[DIM];
    vec_of(3, q2);
    CHECK(veclite_reader_search(db, r, q2, 1, NULL, NULL, 3, 0, 2, ids, sc, &cnt) == VECLITE_OK && cnt == 3 && ids[0] != 3, "real search skips deleted");
    veclite_close(db);
    FILE *f = fopen(path, "r+b");
    fseek(f, 24, SEEK_SET);
    fputc(0x7f, f);
    fseek(f, 4096 + 24, SEEK_SET);
    fputc(0x7f, f);
    fclose(f);
    CHECK(veclite_open(path, DIM, EXT, storage, 0, &db) == VECLITE_E_CORRUPT, "corrupt headers");
    unlink(path);
    unlink(wal);
}

int main(int argc, char **argv) {
    int seeds = argc > 1 ? atoi(argv[1]) : 300;
    storage = argc > 4 ? atoi(argv[4]) : VECLITE_F32;
    int steps = argc > 2 ? atoi(argv[2]) : 200;
    test_api();
    int first = argc > 3 ? atoi(argv[3]) : 1;
    for (int s = first; s < first + seeds && failures == 0; s++) {
        run_workload((uint64_t)s, steps);
        if (failures) printf("failing seed %d\n", s);
    }
    printf("store_test[dtype %d]: %d checks, %d failures | io_crashes %ld idle_crashes %ld restarts %ld wal_replay_over_ckpt %ld lost_unacked_txns %ld searches %ld\n",
           storage, checks, failures, io_crashes, idle_crashes, restarts, wal_over_ckpt, lost_unacked, searches);
    return failures != 0;
}
