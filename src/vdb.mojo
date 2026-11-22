from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys import simd_width_of, get_defined_bool
from std.utils.numerics import isfinite
from std.math import sqrt

comptime OK: Int32 = 0
comptime E_NULL: Int32 = -1
comptime E_N: Int32 = -2
comptime E_DIM: Int32 = -3
comptime E_K: Int32 = -4
comptime E_METRIC: Int32 = -5
comptime E_OVERFLOW: Int32 = -6
comptime E_QUERY_NONFINITE: Int32 = -7
comptime E_SCORE_NONFINITE: Int32 = -8
comptime E_QUERY_ZERO: Int32 = -9
comptime E_CHUNKS: Int32 = -10
comptime E_FILTER: Int32 = -11
comptime E_DATA_NONFINITE: Int32 = -12
comptime E_DATA_ZERO: Int32 = -13

comptime METRIC_L2: Int32 = 0
comptime METRIC_DOT: Int32 = 1
comptime METRIC_COSINE: Int32 = 2

comptime MAX_DIM: Int64 = 65536
comptime MAX_K: Int64 = 4096
comptime MAX_N: Int64 = 1 << 40
comptime MAX_ELEMS: Int64 = 1 << 42
comptime MAX_CHUNKS: Int64 = 4096
comptime ABI_VERSION: Int32 = 1

comptime MUTANT_NO_TIE_BREAK = get_defined_bool["VDB_MUTANT_NO_TIE_BREAK"]()
comptime MUTANT_POST_FILTER = get_defined_bool["VDB_MUTANT_POST_FILTER"]()
comptime MUTANT_LOCAL_K_MINUS_ONE = get_defined_bool["VDB_MUTANT_LOCAL_K_MINUS_ONE"]()

comptime W = simd_width_of[DType.float32]() * 4

comptime FPtr = Pointer[Float32, ImmUntrackedOrigin]
comptime FMut = Pointer[Float32, MutUntrackedOrigin]
comptime IPtr = Pointer[Int64, ImmUntrackedOrigin]
comptime IMut = Pointer[Int64, MutUntrackedOrigin]
comptime BPtr = Pointer[UInt8, ImmUntrackedOrigin]


@inline(.always)
def cand_le(sa: Float32, ia: Int64, sb: Float32, ib: Int64) -> Bool:
    comptime if MUTANT_NO_TIE_BREAK:
        return sa <= sb
    return sa < sb or (sa == sb and ia <= ib)


@inline(.always)
def bstep(
    scores: FMut, ids: IMut, length: Int, cap: Int, s: Float32, i: Int64
) -> Int:
    if cap <= 0:
        return 0
    if length == cap and not cand_le(s, i, scores[unsafe_offset=cap - 1], ids[unsafe_offset=cap - 1]):
        return length
    var p = 0
    while p < length and not cand_le(s, i, scores[unsafe_offset=p], ids[unsafe_offset=p]):
        p += 1
    if p >= cap:
        return length
    var new_len = length + 1 if length < cap else cap
    var j = new_len - 1
    while j > p:
        scores[unsafe_offset=j] = scores[unsafe_offset=j - 1]
        ids[unsafe_offset=j] = ids[unsafe_offset=j - 1]
        j -= 1
    scores[unsafe_offset=p] = s
    ids[unsafe_offset=p] = i
    return new_len


@inline(.always)
def dot_l2(a: FPtr, b: FPtr, dim: Int) -> Tuple[Float32, Float32, Float32]:
    var acc_dot = SIMD[DType.float32, W](0)
    var acc_l2 = SIMD[DType.float32, W](0)
    var acc_nb = SIMD[DType.float32, W](0)
    var d = 0
    while d + W <= dim:
        var va = a.unsafe_offset(d).unsafe_load[width=W]()
        var vb = b.unsafe_offset(d).unsafe_load[width=W]()
        var diff = va - vb
        acc_dot += va * vb
        acc_l2 += diff * diff
        acc_nb += vb * vb
        d += W
    var sd = acc_dot.reduce_add()
    var sl = acc_l2.reduce_add()
    var sn = acc_nb.reduce_add()
    while d < dim:
        var x = a[unsafe_offset=d]
        var y = b[unsafe_offset=d]
        sd += x * y
        sl += (x - y) * (x - y)
        sn += y * y
        d += 1
    return (sd, sl, sn)


@inline(.always)
def row_score(q: FPtr, row: FPtr, dim: Int, metric: Int32, qnorm: Float32) -> Float32:
    var r = dot_l2(q, row, dim)
    if metric == METRIC_L2:
        return r[1]
    if metric == METRIC_DOT:
        return -r[0]
    return Float32(1) - r[0] / (qnorm * sqrt(r[2]))


@inline(.always)
def eligible(live: BPtr, tags: IPtr, i: Int, filter_on: Int32, filter_tag: Int64) -> Bool:
    if live[unsafe_offset=i] == 0:
        return False
    if filter_on == 0:
        return True
    return tags[unsafe_offset=i] == filter_tag


@inline(.always)
def scan_eligible(live: BPtr, tags: IPtr, i: Int, filter_on: Int32, filter_tag: Int64) -> Bool:
    comptime if MUTANT_POST_FILTER:
        return live[unsafe_offset=i] != 0
    return eligible(live, tags, i, filter_on, filter_tag)


def validate_common(
    data: Int, n: Int64, dim: Int64, tags: Int, live: Int, filter_on: Int32
) -> Int32:
    if data == 0 or live == 0:
        return E_NULL
    if n < 0 or n > MAX_N:
        return E_N
    if dim <= 0 or dim > MAX_DIM:
        return E_DIM
    if n * dim > MAX_ELEMS:
        return E_OVERFLOW
    if filter_on != 0 and filter_on != 1:
        return E_FILTER
    if filter_on == 1 and tags == 0:
        return E_NULL
    return OK


def query_norm(query: FPtr, dim: Int, metric: Int32, mut status: Int32) -> Float32:
    var ss: Float32 = 0
    for d in range(dim):
        var x = query[unsafe_offset=d]
        if not isfinite(x):
            status = E_QUERY_NONFINITE
            return 0
        ss += x * x
    if not isfinite(ss):
        status = E_QUERY_NONFINITE
        return 0
    var nrm = sqrt(ss)
    if metric == METRIC_COSINE and not (nrm > 0):
        status = E_QUERY_ZERO
        return 0
    status = OK
    return nrm


@export("vdb_abi_version")
def vdb_abi_version() abi("C") -> Int32:
    return ABI_VERSION


@export("vdb_validate_rows")
def vdb_validate_rows(
    data_addr: Int, n: Int64, dim: Int64, metric: Int32, bad_row_addr: Int
) abi("C") -> Int32:
    if data_addr == 0 or bad_row_addr == 0:
        return E_NULL
    if n < 0 or n > MAX_N:
        return E_N
    if dim <= 0 or dim > MAX_DIM:
        return E_DIM
    if n * dim > MAX_ELEMS:
        return E_OVERFLOW
    if metric < METRIC_L2 or metric > METRIC_COSINE:
        return E_METRIC
    var data = FPtr(unsafe_from_address=data_addr)
    var bad_row = IMut(unsafe_from_address=bad_row_addr)
    bad_row[unsafe_offset=0] = -1
    var di = Int(dim)
    for r in range(Int(n)):
        var row = data.unsafe_offset(r * di)
        var ss: Float32 = 0
        for d in range(di):
            var x = row[unsafe_offset=d]
            if not isfinite(x):
                bad_row[unsafe_offset=0] = Int64(r)
                return E_DATA_NONFINITE
            ss += x * x
        if not isfinite(ss):
            bad_row[unsafe_offset=0] = Int64(r)
            return E_DATA_NONFINITE
        if metric == METRIC_COSINE and not (ss > 0):
            bad_row[unsafe_offset=0] = Int64(r)
            return E_DATA_ZERO
    return OK


@export("vdb_scores")
def vdb_scores(
    data_addr: Int, n: Int64, dim: Int64, query_addr: Int, metric: Int32, out_addr: Int
) abi("C") -> Int32:
    if data_addr == 0 or query_addr == 0 or out_addr == 0:
        return E_NULL
    if n < 0 or n > MAX_N:
        return E_N
    if dim <= 0 or dim > MAX_DIM:
        return E_DIM
    if n * dim > MAX_ELEMS:
        return E_OVERFLOW
    if metric < METRIC_L2 or metric > METRIC_COSINE:
        return E_METRIC
    var data = FPtr(unsafe_from_address=data_addr)
    var query = FPtr(unsafe_from_address=query_addr)
    var out_scores = FMut(unsafe_from_address=out_addr)
    var st: Int32 = OK
    var qn = query_norm(query, Int(dim), metric, st)
    if st != OK:
        return st
    var di = Int(dim)
    for r in range(Int(n)):
        out_scores[unsafe_offset=r] = row_score(query, data.unsafe_offset(r * di), di, metric, qn)
    return OK


@export("vdb_search")
def vdb_search(
    data_addr: Int,
    n: Int64,
    dim: Int64,
    tags_addr: Int,
    live_addr: Int,
    query_addr: Int,
    k: Int64,
    metric: Int32,
    filter_on: Int32,
    filter_tag: Int64,
    nchunks: Int64,
    out_ids_addr: Int,
    out_scores_addr: Int,
    out_count_addr: Int,
) abi("C") -> Int32:
    if out_count_addr != 0:
        IMut(unsafe_from_address=out_count_addr)[unsafe_offset=0] = 0
    var st = validate_common(data_addr, n, dim, tags_addr, live_addr, filter_on)
    if st != OK:
        return st
    if query_addr == 0 or out_ids_addr == 0 or out_scores_addr == 0 or out_count_addr == 0:
        return E_NULL
    if k <= 0 or k > MAX_K:
        return E_K
    if metric < METRIC_L2 or metric > METRIC_COSINE:
        return E_METRIC
    if nchunks <= 0 or nchunks > MAX_CHUNKS:
        return E_CHUNKS
    var data = FPtr(unsafe_from_address=data_addr)
    var tags = IPtr(unsafe_from_address=tags_addr if tags_addr != 0 else live_addr)
    var live = BPtr(unsafe_from_address=live_addr)
    var query = FPtr(unsafe_from_address=query_addr)
    var out_ids = IMut(unsafe_from_address=out_ids_addr)
    var out_scores = FMut(unsafe_from_address=out_scores_addr)
    var out_count = IMut(unsafe_from_address=out_count_addr)
    out_count[unsafe_offset=0] = 0
    var qn = query_norm(query, Int(dim), metric, st)
    if st != OK:
        return st

    initialize_runtime()

    var ni = Int(n)
    var di = Int(dim)
    var ki = Int(k)
    var nc = Int(nchunks)
    var local_cap = ki - 1 if MUTANT_LOCAL_K_MINUS_ONE else ki
    var loc_scores = List[Float32](length=nc * ki, fill=0)
    var loc_ids = List[Int64](length=nc * ki, fill=-1)
    var loc_len = List[Int64](length=nc, fill=0)
    var loc_status = List[Int32](length=nc, fill=OK)
    var ps = FMut(unsafe_from_address=Int(loc_scores.unsafe_ptr()))
    var pi = IMut(unsafe_from_address=Int(loc_ids.unsafe_ptr()))
    var pl = IMut(unsafe_from_address=Int(loc_len.unsafe_ptr()))
    var pst = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(loc_status.unsafe_ptr()))

    def scan_chunk(c: Int) {var}:
        var lo = c * ni // nc
        var hi = (c + 1) * ni // nc
        var cs = ps.unsafe_offset(c * ki)
        var cid = pi.unsafe_offset(c * ki)
        var length = 0
        for i in range(lo, hi):
            if not scan_eligible(live, tags, i, filter_on, filter_tag):
                continue
            var s = row_score(query, data.unsafe_offset(i * di), di, metric, qn)
            if not isfinite(s):
                pst[unsafe_offset=c] = E_SCORE_NONFINITE
                pl[unsafe_offset=c] = 0
                return
            length = bstep(cs, cid, length, local_cap, s, Int64(i))
        pl[unsafe_offset=c] = Int64(length)

    parallelize(scan_chunk, nc)

    for c in range(nc):
        if loc_status[c] != OK:
            return loc_status[c]

    var length = 0
    for c in range(nc):
        var cs = ps.unsafe_offset(c * ki)
        var cid = pi.unsafe_offset(c * ki)
        for j in range(Int(loc_len[c])):
            var id = cid[unsafe_offset=j]
            comptime if MUTANT_POST_FILTER:
                if not eligible(live, tags, Int(id), filter_on, filter_tag):
                    continue
            length = bstep(out_scores, out_ids, length, ki, cs[unsafe_offset=j], id)
    out_count[unsafe_offset=0] = Int64(length)
    _ = loc_scores^
    _ = loc_ids^
    _ = loc_len^
    _ = loc_status^
    return OK
