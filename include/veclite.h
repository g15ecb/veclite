#ifndef VECLITE_H
#define VECLITE_H

#include <stdint.h>

#define VECLITE_OK 0
#define VECLITE_E_NULL (-1)
#define VECLITE_E_N (-2)
#define VECLITE_E_DIM (-3)
#define VECLITE_E_K (-4)
#define VECLITE_E_METRIC (-5)
#define VECLITE_E_OVERFLOW (-6)
#define VECLITE_E_QUERY_NONFINITE (-7)
#define VECLITE_E_SCORE_NONFINITE (-8)
#define VECLITE_E_QUERY_ZERO (-9)
#define VECLITE_E_CHUNKS (-10)
#define VECLITE_E_FILTER (-11)
#define VECLITE_E_DATA_NONFINITE (-12)
#define VECLITE_E_DATA_ZERO (-13)
#define VECLITE_E_BATCH (-14)
#define VECLITE_E_DTYPE (-15)
#define VECLITE_E_RANGE (-16)

#define VECLITE_F32 0
#define VECLITE_F16 1

#define VECLITE_METRIC_L2 0
#define VECLITE_METRIC_DOT 1
#define VECLITE_METRIC_COSINE 2

#define VECLITE_MAX_DIM 65536
#define VECLITE_MAX_K 4096
#define VECLITE_MAX_CHUNKS 4096
#define VECLITE_MAX_BATCH 4096

int32_t veclite_abi_version(void);

int32_t veclite_validate_rows(const float *data, int64_t n, int64_t dim, int32_t metric, int64_t *bad_row);

int32_t veclite_scores(const float *data, int64_t n, int64_t dim, const float *queries, int64_t nq, int32_t metric,
                   float *out_scores);

int32_t veclite_encode(int32_t dtype, const float *src, int64_t n, int64_t dim, void *dst, int64_t *bad_row);

int32_t veclite_search_ex(const void *data, int32_t dtype, int64_t n, int64_t dim, const int64_t *tags, const uint8_t *live,
                      const float *queries, int64_t nq, const uint8_t *filter_on, const int64_t *filter_tags, int64_t k,
                      int32_t metric, int64_t nchunks, int64_t *out_ids, float *out_scores, int64_t *out_counts);

int32_t veclite_search(const float *data, int64_t n, int64_t dim, const int64_t *tags, const uint8_t *live,
                   const float *queries, int64_t nq, const uint8_t *filter_on, const int64_t *filter_tags,
                   int64_t k, int32_t metric, int64_t nchunks, int64_t *out_ids, float *out_scores,
                   int64_t *out_counts);

#endif
