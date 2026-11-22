#ifndef VDB_STORE_H
#define VDB_STORE_H

#include <stdint.h>

#include "vdb.h"

#define VDB_E_IO (-20)
#define VDB_E_CORRUPT (-21)
#define VDB_E_STATE (-22)
#define VDB_E_ID (-23)
#define VDB_E_READER (-24)
#define VDB_E_ARG (-25)

enum { VDB_INFO_ROWS, VDB_INFO_NTX, VDB_INFO_PUBLISHED, VDB_INFO_HDR_TXN, VDB_INFO_WAL_FRAMES, VDB_INFO_SALT, VDB_INFO_MAIN_ROWS, VDB_INFO_POISONED, VDB_INFO_LEN };

int32_t vdb_sim_new(int64_t *out_sim);
int32_t vdb_sim_free(int64_t sim);
int32_t vdb_sim_set_budget(int64_t sim, int64_t budget);
int64_t vdb_sim_mutations(int64_t sim);
int32_t vdb_sim_crash(int64_t sim, uint64_t seed);

int32_t vdb_open(const char *path, int64_t dim, int64_t extent_rows, int64_t sim, int64_t *out_db);
int32_t vdb_close(int64_t db);
int32_t vdb_append(int64_t db, const float *vec, int64_t tag, int64_t *out_id);
int32_t vdb_delete(int64_t db, int64_t id);
int32_t vdb_commit(int64_t db, int64_t *out_txn);
int32_t vdb_checkpoint(int64_t db, int64_t *out_ckpt_txn);
int32_t vdb_reader_begin(int64_t db, int64_t *out_reader);
int32_t vdb_reader_end(int64_t db, int64_t reader);
int32_t vdb_reader_mark(int64_t db, int64_t reader, int64_t *out_mark);
int32_t vdb_reader_view(int64_t db, int64_t reader, int64_t cap, uint8_t *out_live, int64_t *out_tags, float *out_vecs,
                        int64_t *out_n);
int32_t vdb_reader_search(int64_t db, int64_t reader, const float *queries, int64_t nq, const uint8_t *filter_on,
                          const int64_t *filter_tags, int64_t k, int32_t metric, int64_t nchunks, int64_t *out_ids,
                          float *out_scores, int64_t *out_counts);
int32_t vdb_info(int64_t db, int64_t *out);

#endif
