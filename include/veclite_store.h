#ifndef VECLITE_STORE_H
#define VECLITE_STORE_H

#include <stdint.h>

#include "veclite.h"

#define VECLITE_E_IO (-20)
#define VECLITE_E_CORRUPT (-21)
#define VECLITE_E_STATE (-22)
#define VECLITE_E_ID (-23)
#define VECLITE_E_READER (-24)
#define VECLITE_E_ARG (-25)
#define VECLITE_E_GPU (-30)
#define VECLITE_E_NO_GPU (-31)

#define VECLITE_DEVICE_AUTO 0
#define VECLITE_DEVICE_CPU 1
#define VECLITE_DEVICE_GPU 2

enum { VECLITE_INFO_ROWS, VECLITE_INFO_NTX, VECLITE_INFO_PUBLISHED, VECLITE_INFO_HDR_TXN, VECLITE_INFO_WAL_FRAMES, VECLITE_INFO_SALT, VECLITE_INFO_MAIN_ROWS, VECLITE_INFO_POISONED, VECLITE_INFO_DTYPE, VECLITE_INFO_DIM, VECLITE_INFO_GPU_ROWS, VECLITE_INFO_LIVE_ROWS, VECLITE_INFO_LEN };

int32_t veclite_sim_new(int64_t *out_sim);
int32_t veclite_sim_free(int64_t sim);
int32_t veclite_sim_set_budget(int64_t sim, int64_t budget);
int64_t veclite_sim_mutations(int64_t sim);
int32_t veclite_sim_crash(int64_t sim, uint64_t seed);
int32_t veclite_sim_crash_range(int64_t sim, int64_t wal_lo, int64_t wal_hi);

int32_t veclite_open(const char *path, int64_t dim, int64_t extent_rows, int32_t storage, int64_t sim, int64_t *out_db);
int32_t veclite_close(int64_t db);
int32_t veclite_append(int64_t db, const float *vec, int64_t tag, int64_t *out_id);
int32_t veclite_delete(int64_t db, int64_t id);
int32_t veclite_commit(int64_t db, int64_t *out_txn);
int32_t veclite_pending(int64_t db, int64_t *out_n);
int32_t veclite_rollback(int64_t db, int64_t mark);
int32_t veclite_checkpoint(int64_t db, int64_t *out_ckpt_txn);
int32_t veclite_reader_begin(int64_t db, int64_t *out_reader);
int32_t veclite_reader_end(int64_t db, int64_t reader);
int32_t veclite_reader_mark(int64_t db, int64_t reader, int64_t *out_mark);
int32_t veclite_reader_view(int64_t db, int64_t reader, int64_t cap, uint8_t *out_live, int64_t *out_tags, float *out_vecs,
                        int64_t *out_n);
int32_t veclite_reader_search(int64_t db, int64_t reader, const float *queries, int64_t nq, const uint8_t *filter_on,
                          const int64_t *filter_tags, int64_t k, int32_t metric, int64_t nchunks, int64_t *out_ids,
                          float *out_scores, int64_t *out_counts);
int32_t veclite_info(int64_t db, int64_t *out);
int32_t veclite_gpu_available(void);
int32_t veclite_reader_search_dev(int64_t db, int64_t reader, const float *queries, int64_t nq, const uint8_t *filter_on,
                                  const int64_t *filter_tags, int64_t k, int32_t metric, int64_t nchunks, int32_t device,
                                  int64_t gpu_min_batch, int64_t *out_ids, float *out_scores, int64_t *out_counts,
                                  int32_t *out_used_gpu);

#endif
