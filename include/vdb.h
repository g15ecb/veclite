#ifndef VDB_H
#define VDB_H

#include <stdint.h>

#define VDB_OK 0
#define VDB_E_NULL (-1)
#define VDB_E_N (-2)
#define VDB_E_DIM (-3)
#define VDB_E_K (-4)
#define VDB_E_METRIC (-5)
#define VDB_E_OVERFLOW (-6)
#define VDB_E_QUERY_NONFINITE (-7)
#define VDB_E_SCORE_NONFINITE (-8)
#define VDB_E_QUERY_ZERO (-9)
#define VDB_E_CHUNKS (-10)
#define VDB_E_FILTER (-11)
#define VDB_E_DATA_NONFINITE (-12)
#define VDB_E_DATA_ZERO (-13)
#define VDB_E_BATCH (-14)

#define VDB_METRIC_L2 0
#define VDB_METRIC_DOT 1
#define VDB_METRIC_COSINE 2

#define VDB_MAX_DIM 65536
#define VDB_MAX_K 4096
#define VDB_MAX_CHUNKS 4096
#define VDB_MAX_BATCH 4096

int32_t vdb_abi_version(void);

int32_t vdb_validate_rows(const float *data, int64_t n, int64_t dim, int32_t metric, int64_t *bad_row);

int32_t vdb_scores(const float *data, int64_t n, int64_t dim, const float *queries, int64_t nq, int32_t metric,
                   float *out_scores);

int32_t vdb_search(const float *data, int64_t n, int64_t dim, const int64_t *tags, const uint8_t *live,
                   const float *queries, int64_t nq, const uint8_t *filter_on, const int64_t *filter_tags,
                   int64_t k, int32_t metric, int64_t nchunks, int64_t *out_ids, float *out_scores,
                   int64_t *out_counts);

#endif
