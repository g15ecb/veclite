from max.algorithm import parallelize
from std.runtime import initialize_runtime
from std.sys import simd_width_of, get_defined_bool, get_defined_int
from std.utils.numerics import isfinite, inf, nextafter
from std.math import sqrt, round

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
comptime E_BATCH: Int32 = -14
comptime E_DTYPE: Int32 = -15
comptime E_RANGE: Int32 = -16

comptime L2: Int32 = 0
comptime DOT: Int32 = 1
comptime COSINE: Int32 = 2

comptime F32: Int32 = 0
comptime F16: Int32 = 1
comptime I8: Int32 = 2
comptime F16_MAX: Float32 = 65504
comptime AUX = 3

comptime MAX_DIM: Int64 = 65536
comptime MAX_K: Int64 = 4096
comptime MAX_N: Int64 = 1 << 40
comptime MAX_ELEMS: Int64 = 1 << 42
comptime MAX_CHUNKS: Int64 = 4096
comptime MAX_BATCH: Int64 = 4096

comptime MUTANT_NO_TIE_BREAK = get_defined_bool["VDB_MUTANT_NO_TIE_BREAK"]()
comptime MUTANT_POST_FILTER = get_defined_bool["VDB_MUTANT_POST_FILTER"]()
comptime MUTANT_LOCAL_K_MINUS_ONE = get_defined_bool["VDB_MUTANT_LOCAL_K_MINUS_ONE"]()
comptime MUTANT_QUERY_MIX = get_defined_bool["VDB_MUTANT_QUERY_MIX"]()
comptime MUTANT_NO_EPS = get_defined_bool["VDB_MUTANT_NO_EPS"]()
comptime MUTANT_TAU_FROM_LOW = get_defined_bool["VDB_MUTANT_TAU_FROM_LOW"]()
comptime MUTANT_NO_FALLBACK = get_defined_bool["VDB_MUTANT_NO_FALLBACK"]()
comptime CAND_FACTOR = get_defined_int["VDB_CAND_FACTOR", 4]()
comptime CAND_MIN = get_defined_int["VDB_CAND_MIN", 64]()

comptime W = simd_width_of[DType.float32]()
comptime V = SIMD[DType.float32, W]
comptime QT = get_defined_int["VDB_QT", 4]()
comptime RT = get_defined_int["VDB_RT", 4]()
comptime BLOCK_BYTES = 1 << 18
comptime U_ROUND: Float64 = 5.9604644775390625e-08

comptime FPtr = Pointer[Float32, ImmUntrackedOrigin]
comptime FMut = Pointer[Float32, MutUntrackedOrigin]
comptime DMut = Pointer[Float64, MutUntrackedOrigin]
comptime IPtr = Pointer[Int64, ImmUntrackedOrigin]
comptime IMut = Pointer[Int64, MutUntrackedOrigin]
comptime BPtr = Pointer[UInt8, ImmUntrackedOrigin]


def fptr(a: Int) -> FPtr:
    return FPtr(unsafe_from_address=a)


def fmut(a: Int) -> FMut:
    return FMut(unsafe_from_address=a)


def dmut(a: Int) -> DMut:
    return DMut(unsafe_from_address=a)


def iptr(a: Int) -> IPtr:
    return IPtr(unsafe_from_address=a)


def imut(a: Int) -> IMut:
    return IMut(unsafe_from_address=a)


def bptr(a: Int) -> BPtr:
    return BPtr(unsafe_from_address=a)


def elem_bytes(dtype: Int32) -> Int:
    return 4 if dtype == F32 else (2 if dtype == F16 else 1)


@inline(.always)
def cand_le[D: DType](sa: Scalar[D], ia: Int64, sb: Scalar[D], ib: Int64) -> Bool:
    comptime if MUTANT_NO_TIE_BREAK:
        return sa <= sb
    return sa < sb or (sa == sb and ia <= ib)


@inline(.always)
def bstep[D: DType](
    scores: Pointer[Scalar[D], MutUntrackedOrigin], ids: IMut, length: Int, cap: Int, s: Scalar[D], i: Int64
) -> Int:
    if cap <= 0 or (length == cap and not cand_le(s, i, scores[unsafe_offset=cap - 1], ids[unsafe_offset=cap - 1])):
        return length
    var p = 0
    while p < length and not cand_le(s, i, scores[unsafe_offset=p], ids[unsafe_offset=p]):
        p += 1
    var j = min(length, cap - 1)
    while j > p:
        scores[unsafe_offset=j] = scores[unsafe_offset=j - 1]
        ids[unsafe_offset=j] = ids[unsafe_offset=j - 1]
        j -= 1
    scores[unsafe_offset=p] = s
    ids[unsafe_offset=p] = i
    return min(length + 1, cap)


@inline(.always)
def dots[NQ: Int, NR: Int, NORM: Bool, D: DType](q: FPtr, r: Array[Int, NR], dim: Int) -> Array[Float32, (NQ + 1) * NR]:
    var acc = Array[V, (NQ + 1) * NR](fill=V(0))
    var d = 0
    while d + W <= dim:
        comptime for j in range(NR):
            var b = Pointer[Scalar[D], ImmUntrackedOrigin](unsafe_from_address=r[j]).unsafe_offset(d).unsafe_load[width=W]().cast[DType.float32]()
            comptime for i in range(NQ):
                acc[i * NR + j] = q.unsafe_offset(i * dim + d).unsafe_load[width=W]().fma(b, acc[i * NR + j])
            comptime if NORM:
                acc[NQ * NR + j] = b.fma(b, acc[NQ * NR + j])
        d += W
    var out = Array[Float32, (NQ + 1) * NR](fill=0)
    comptime for t in range((NQ + 1) * NR):
        out[t] = acc[t].reduce_add()
    for e in range(d, dim):
        comptime for j in range(NR):
            var b = Pointer[Scalar[D], ImmUntrackedOrigin](unsafe_from_address=r[j])[unsafe_offset=e].cast[DType.float32]()
            comptime for i in range(NQ):
                out[i * NR + j] = q[unsafe_offset=i * dim + e].fma(b, out[i * NR + j])
            comptime if NORM:
                out[NQ * NR + j] = b.fma(b, out[NQ * NR + j])
    return out^


@inline(.always)
def sqnorm(p: FPtr, dim: Int) -> Float32:
    return dots[0, 1, True, DType.float32](p, Array[Int, 1](fill=Int(p)), dim)[0]


@inline(.always)
def finish[metric: Int32](dot: Float32, qn2: Float32, rn2: Float32) -> Float32:
    comptime if metric == L2:
        var v = qn2 + rn2 - 2 * dot
        return Float32(0) if v < 0 else v
    elif metric == DOT:
        return -dot
    return 1 - dot / (sqrt(qn2) * sqrt(rn2))


@inline(.always)
def eps_bound[metric: Int32](dim: Int, qn2: Float32, rn2: Float32, err: Float32) -> Float64:
    comptime if MUTANT_NO_EPS:
        return 0
    var g = Float64(dim + 2) * U_ROUND / (1 - Float64(dim + 2) * U_ROUND)
    var qn = sqrt(Float64(qn2) * (1 + g))
    var xn = sqrt(Float64(rn2) * (1 + g))
    var e = Float64(err)
    var xh = xn + e
    var delta = (g + 2 * U_ROUND) * qn * xh + g * qn * xn + qn * e
    var bound: Float64
    comptime if metric == DOT:
        bound = delta
    elif metric == L2:
        bound = 2 * delta + 4 * U_ROUND * (Float64(qn2) + Float64(rn2) + 2 * qn * xh)
    else:
        var df = sqrt(Float64(qn2)) * sqrt(Float64(rn2)) * (1 - 4 * U_ROUND)
        bound = delta / df * (1 + 4 * U_ROUND) + 4 * U_ROUND * (1 + qn * xh / df)
    return bound * 1.001


@fieldwise_init
struct Job(ImplicitlyCopyable, Movable):
    var data: Int
    var stride: Int
    var aux: FPtr
    var n: Int
    var dim: Int
    var tags: IPtr
    var live: BPtr
    var queries: FPtr
    var nq: Int
    var qn2: FPtr
    var filter_on: BPtr
    var filter_tags: IPtr
    var filtered: Bool
    var k: Int
    var nc: Int
    var buf_scores: FMut
    var buf_ids: IMut
    var buf_len: IMut
    var dst: FMut
    var mcap: Int
    var up: DMut
    var up_ids: IMut
    var up_len: IMut
    var lo: DMut
    var lo_ids: IMut
    var lo_len: IMut

    @inline(.always)
    def matches(self, f: Int, i: Int) -> Bool:
        return self.live[unsafe_offset=i] != 0 and (
            not self.filtered or self.filter_on[unsafe_offset=f] == 0 or self.tags[unsafe_offset=i] == self.filter_tags[unsafe_offset=f]
        )

    @inline(.always)
    def scan_ok[STORE: Bool](self, q: Int, i: Int) -> Bool:
        comptime if STORE:
            return True
        comptime if MUTANT_POST_FILTER:
            return self.live[unsafe_offset=i] != 0
        comptime if MUTANT_QUERY_MIX:
            return self.matches(0, i)
        return self.matches(q, i)

    @inline(.always)
    def row(self, i: Int) -> Int:
        return self.data + i * self.stride


@inline(.always)
def do_tile[metric: Int32, NQ: Int, NR: Int, STORE: Bool, S: Int32](job: Job, c: Int, q0: Int, rows: IPtr) -> Int32:
    comptime D = DType.float32 if S == F32 else (DType.float16 if S == F16 else DType.int8)
    var r = Array[Int, NR](fill=job.data)
    comptime for j in range(NR):
        r[j] = job.row(Int(rows[unsafe_offset=j]))
    var t = dots[NQ, NR, metric != DOT and S != I8, D](job.queries.unsafe_offset(q0 * job.dim), r, job.dim)
    var cap = job.k - 1 if MUTANT_LOCAL_K_MINUS_ONE else job.k
    comptime for i in range(NQ):
        comptime for j in range(NR):
            var q = q0 + i
            var id = Int(rows[unsafe_offset=j])
            if job.scan_ok[STORE](q, id):
                comptime if S == I8:
                    var m = job.aux.unsafe_offset(id * AUX)
                    var a = finish[metric](m[unsafe_offset=0] * t[i * NR + j], job.qn2[unsafe_offset=q], m[unsafe_offset=1])
                    if not isfinite(a):
                        return E_SCORE_NONFINITE
                    var eps = eps_bound[metric](job.dim, job.qn2[unsafe_offset=q], m[unsafe_offset=1], m[unsafe_offset=2])
                    var o = c * job.nq + q
                    job.up_len[unsafe_offset=o] = Int64(bstep(
                        job.up.unsafe_offset(o * job.k), job.up_ids.unsafe_offset(o * job.k),
                        Int(job.up_len[unsafe_offset=o]), job.k, Float64(a) + eps, Int64(id),
                    ))
                    job.lo_len[unsafe_offset=o] = Int64(bstep(
                        job.lo.unsafe_offset(o * job.mcap), job.lo_ids.unsafe_offset(o * job.mcap),
                        Int(job.lo_len[unsafe_offset=o]), job.mcap, Float64(a) - eps, Int64(id),
                    ))
                else:
                    var s = finish[metric](t[i * NR + j], job.qn2[unsafe_offset=q], t[NQ * NR + j])
                    if not isfinite(s):
                        return E_SCORE_NONFINITE
                    comptime if STORE:
                        job.dst[unsafe_offset=q * job.n + id] = s
                    else:
                        var o = c * job.nq + q
                        job.buf_len[unsafe_offset=o] = Int64(bstep(
                            job.buf_scores.unsafe_offset(o * job.k), job.buf_ids.unsafe_offset(o * job.k),
                            Int(job.buf_len[unsafe_offset=o]), cap, s, Int64(id),
                        ))
    return OK


def scan_qtile[metric: Int32, NQ: Int, STORE: Bool, S: Int32](job: Job, c: Int, q0: Int, b0: Int, b1: Int, idx: IMut) -> Int32:
    var m = 0
    for r in range(b0, b1):
        var hit = False
        comptime for i in range(NQ):
            hit = hit or job.scan_ok[STORE](q0 + i, r)
        if hit:
            idx[unsafe_offset=m] = Int64(r)
            m += 1
    var rows = iptr(Int(idx))
    var j = 0
    var st = OK
    while st == OK and j + RT <= m:
        st = do_tile[metric, NQ, RT, STORE, S](job, c, q0, rows.unsafe_offset(j))
        j += RT
    while st == OK and j < m:
        st = do_tile[metric, NQ, 1, STORE, S](job, c, q0, rows.unsafe_offset(j))
        j += 1
    return st


def run_chunk[metric: Int32, STORE: Bool, S: Int32](job: Job, c: Int) -> Int32:
    var rb = max(RT, BLOCK_BYTES // job.stride)
    var idx = List[Int64](length=rb, fill=0)
    var pidx = imut(Int(idx.unsafe_ptr()))
    var hi = (c + 1) * job.n // job.nc
    var st = OK
    for b0 in range(c * job.n // job.nc, hi, rb):
        var b1 = min(b0 + rb, hi)
        var q0 = 0
        while st == OK and q0 + QT <= job.nq:
            st = scan_qtile[metric, QT, STORE, S](job, c, q0, b0, b1, pidx)
            q0 += QT
        while st == OK and q0 < job.nq:
            st = scan_qtile[metric, 1, STORE, S](job, c, q0, b0, b1, pidx)
            q0 += 1
        if st != OK:
            break
    _ = idx^
    return st


def run_all[metric: Int32, STORE: Bool, S: Int32](job: Job) -> Int32:
    var status = List[Int32](length=job.nc, fill=OK)
    var ps = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(status.unsafe_ptr()))

    def body(c: Int) {var}:
        ps[unsafe_offset=c] = run_chunk[metric, STORE, S](job, c)

    parallelize(body, job.nc)
    for s in status:
        if s != OK:
            return s
    return OK


def run_typed[metric: Int32, STORE: Bool](job: Job, dtype: Int32) -> Int32:
    if dtype == F16:
        return run_all[metric, STORE, F16](job)
    if dtype == I8:
        return run_all[metric, STORE, I8](job)
    return run_all[metric, STORE, F32](job)


def run[STORE: Bool](job: Job, metric: Int32, dtype: Int32) -> Int32:
    if metric == L2:
        return run_typed[L2, STORE](job, dtype)
    if metric == DOT:
        return run_typed[DOT, STORE](job, dtype)
    return run_typed[COSINE, STORE](job, dtype)


def exact_score[metric: Int32](q: FPtr, qn2: Float32, row: Int, dim: Int) -> Float32:
    var t = dots[1, 1, metric != DOT, DType.float32](q, Array[Int, 1](fill=row), dim)
    return finish[metric](t[0], qn2, t[1])


def exact_score_m(metric: Int32, q: FPtr, qn2: Float32, row: Int, dim: Int) -> Float32:
    if metric == L2:
        return exact_score[L2](q, qn2, row, dim)
    if metric == DOT:
        return exact_score[DOT](q, qn2, row, dim)
    return exact_score[COSINE](q, qn2, row, dim)


def check_shape(data: Int, n: Int64, dim: Int64) -> Int32:
    if data == 0:
        return E_NULL
    if n < 0 or n > MAX_N:
        return E_N
    if dim <= 0 or dim > MAX_DIM:
        return E_DIM
    if n * dim > MAX_ELEMS:
        return E_OVERFLOW
    return OK


def check_metric(metric: Int32) -> Int32:
    return OK if metric >= L2 and metric <= COSINE else E_METRIC


def check_batch(nq: Int64) -> Int32:
    return OK if nq > 0 and nq <= MAX_BATCH else E_BATCH


def check_dtype(dtype: Int32) -> Int32:
    return OK if dtype >= F32 and dtype <= I8 else E_DTYPE


def query_norms(queries: FPtr, nq: Int, dim: Int, metric: Int32, mut st: Int32) -> List[Float32]:
    var qn2 = List[Float32](length=nq, fill=0)
    for q in range(nq):
        var ss = sqnorm(queries.unsafe_offset(q * dim), dim)
        if not isfinite(ss):
            st = E_QUERY_NONFINITE
        elif metric == COSINE and not (ss > 0):
            st = E_QUERY_ZERO
        if st != OK:
            return qn2^
        qn2[q] = ss
    return qn2^


def encode_row(dtype: Int32, src: FPtr, dim: Int, dst: Int, aux: FMut) -> Int32:
    var ss = sqnorm(src, dim)
    if not isfinite(ss):
        return E_DATA_NONFINITE
    if dtype == F32:
        for d in range(dim):
            fmut(dst)[unsafe_offset=d] = src[unsafe_offset=d]
        return OK
    if dtype == F16:
        var h = Pointer[Float16, MutUntrackedOrigin](unsafe_from_address=dst)
        for d in range(dim):
            var x = src[unsafe_offset=d]
            if x > F16_MAX or x < -F16_MAX:
                return E_RANGE
            h[unsafe_offset=d] = x.cast[DType.float16]()
        return OK
    var amax: Float32 = 0
    for d in range(dim):
        amax = max(amax, abs(src[unsafe_offset=d]))
    var scale = amax / 127
    var q8 = Pointer[Int8, MutUntrackedOrigin](unsafe_from_address=dst)
    var err2: Float64 = 0
    for d in range(dim):
        var x = src[unsafe_offset=d]
        var v: Float32 = 0 if scale == 0 else min(Float32(127), max(Float32(-127), round(x / scale)))
        q8[unsafe_offset=d] = Int8(Int(v))
        var diff = Float64(x) - Float64(scale) * Float64(v)
        err2 += diff * diff
    var err64 = sqrt(err2) * (1 + 1e-9)
    var err = Float32(err64)
    if Float64(err) < err64:
        err = nextafter(err, inf[DType.float32]())
    aux[unsafe_offset=0] = scale
    aux[unsafe_offset=1] = ss
    aux[unsafe_offset=2] = err
    return OK


@export("vdb_abi_version")
def vdb_abi_version() abi("C") -> Int32:
    return 3


@export("vdb_encode")
def vdb_encode(dtype: Int32, src: Int, n: Int64, dim: Int64, dst: Int, aux: Int, bad_row: Int) abi("C") -> Int32:
    var st = check_dtype(dtype)
    if st == OK:
        st = check_shape(src, n, dim)
    if st == OK and (dst == 0 or bad_row == 0 or (dtype == I8 and aux == 0)):
        st = E_NULL
    if st != OK:
        return st
    imut(bad_row)[unsafe_offset=0] = -1
    var di = Int(dim)
    var eb = elem_bytes(dtype)
    for r in range(Int(n)):
        st = encode_row(dtype, fptr(src).unsafe_offset(r * di), di, dst + r * di * eb, fmut(aux).unsafe_offset(r * AUX) if aux != 0 else fmut(dst))
        if st != OK:
            imut(bad_row)[unsafe_offset=0] = Int64(r)
            return st
    return OK


@export("vdb_i8_bounds")
def vdb_i8_bounds(
    data: Int, aux: Int, orig: Int, n: Int64, dim: Int64, query: Int, metric: Int32,
    out_approx: Int, out_eps: Int, out_exact: Int,
) abi("C") -> Int32:
    var st = check_shape(data, n, dim)
    if st == OK and (aux == 0 or orig == 0 or query == 0 or out_approx == 0 or out_eps == 0 or out_exact == 0):
        st = E_NULL
    if st == OK:
        st = check_metric(metric)
    var qn2 = query_norms(fptr(query), 1, Int(dim), metric, st)
    if st != OK:
        return st
    var di = Int(dim)
    for r in range(Int(n)):
        var m = fptr(aux).unsafe_offset(r * AUX)
        var t = dots[1, 1, False, DType.int8](fptr(query), Array[Int, 1](fill=data + r * di), di)
        var a: Float32
        var e: Float64
        if metric == L2:
            a = finish[L2](m[unsafe_offset=0] * t[0], qn2[0], m[unsafe_offset=1])
            e = eps_bound[L2](di, qn2[0], m[unsafe_offset=1], m[unsafe_offset=2])
        elif metric == DOT:
            a = finish[DOT](m[unsafe_offset=0] * t[0], qn2[0], m[unsafe_offset=1])
            e = eps_bound[DOT](di, qn2[0], m[unsafe_offset=1], m[unsafe_offset=2])
        else:
            a = finish[COSINE](m[unsafe_offset=0] * t[0], qn2[0], m[unsafe_offset=1])
            e = eps_bound[COSINE](di, qn2[0], m[unsafe_offset=1], m[unsafe_offset=2])
        fmut(out_approx)[unsafe_offset=r] = a
        dmut(out_eps)[unsafe_offset=r] = e
        fmut(out_exact)[unsafe_offset=r] = exact_score_m(metric, fptr(query), qn2[0], orig + r * di * 4, di)
    _ = qn2^
    return OK


@export("vdb_validate_rows")
def vdb_validate_rows(data_addr: Int, n: Int64, dim: Int64, metric: Int32, bad_row_addr: Int) abi("C") -> Int32:
    var st = check_shape(data_addr, n, dim)
    if st == OK and bad_row_addr == 0:
        st = E_NULL
    if st == OK:
        st = check_metric(metric)
    if st != OK:
        return st
    var bad_row = imut(bad_row_addr)
    bad_row[unsafe_offset=0] = -1
    for r in range(Int(n)):
        var ss = sqnorm(fptr(data_addr).unsafe_offset(r * Int(dim)), Int(dim))
        st = E_DATA_NONFINITE if not isfinite(ss) else (E_DATA_ZERO if metric == COSINE and not (ss > 0) else OK)
        if st != OK:
            bad_row[unsafe_offset=0] = Int64(r)
            return st
    return OK


def blank_job(data: Int, stride: Int, n: Int, dim: Int, queries: Int, nq: Int, qn2: Int, dst: Int) -> Job:
    return Job(
        data, stride, fptr(data), n, dim, iptr(data), bptr(data), fptr(queries), nq, fptr(qn2), bptr(data), iptr(data),
        False, 1, 1, fmut(dst), imut(dst), imut(dst), fmut(dst), 1, dmut(dst), imut(dst), imut(dst), dmut(dst), imut(dst), imut(dst),
    )


@export("vdb_scores")
def vdb_scores(
    data_addr: Int, n: Int64, dim: Int64, queries_addr: Int, nq: Int64, metric: Int32, out_addr: Int
) abi("C") -> Int32:
    var st = check_shape(data_addr, n, dim)
    if st == OK and (queries_addr == 0 or out_addr == 0):
        st = E_NULL
    if st == OK:
        st = check_batch(nq)
    if st == OK:
        st = check_metric(metric)
    if st != OK:
        return st
    var qn2 = query_norms(fptr(queries_addr), Int(nq), Int(dim), metric, st)
    if st != OK or n == 0:
        return st
    initialize_runtime()
    st = run[True](blank_job(data_addr, Int(dim) * 4, Int(n), Int(dim), queries_addr, Int(nq), Int(qn2.unsafe_ptr()), out_addr), metric, F32)
    _ = qn2^
    return st


@export("vdb_search")
def vdb_search(
    data_addr: Int, n: Int64, dim: Int64, tags_addr: Int, live_addr: Int,
    queries_addr: Int, nq: Int64, filter_on_addr: Int, filter_tags_addr: Int,
    k: Int64, metric: Int32, nchunks: Int64,
    out_ids_addr: Int, out_scores_addr: Int, out_counts_addr: Int,
) abi("C") -> Int32:
    return vdb_search_ex(
        data_addr, F32, 0, 0, n, dim, tags_addr, live_addr, queries_addr, nq, filter_on_addr, filter_tags_addr,
        k, metric, nchunks, out_ids_addr, out_scores_addr, out_counts_addr, 0,
    )


@export("vdb_search_ex")
def vdb_search_ex(
    data_addr: Int, dtype: Int32, aux_addr: Int, orig_addr: Int, n: Int64, dim: Int64, tags_addr: Int, live_addr: Int,
    queries_addr: Int, nq: Int64, filter_on_addr: Int, filter_tags_addr: Int,
    k: Int64, metric: Int32, nchunks: Int64,
    out_ids_addr: Int, out_scores_addr: Int, out_counts_addr: Int, stats_addr: Int,
) abi("C") -> Int32:
    if out_counts_addr == 0:
        return E_NULL
    var st = check_batch(nq)
    if st != OK:
        return st
    var counts = imut(out_counts_addr)
    for q in range(Int(nq)):
        counts[unsafe_offset=q] = 0
    st = check_shape(data_addr, n, dim)
    if st == OK:
        st = check_dtype(dtype)
    if st == OK and (live_addr == 0 or queries_addr == 0 or out_ids_addr == 0 or out_scores_addr == 0):
        st = E_NULL
    if st == OK and dtype == I8 and (aux_addr == 0 or orig_addr == 0):
        st = E_NULL
    if st == OK and (k <= 0 or k > MAX_K):
        st = E_K
    if st == OK:
        st = check_metric(metric)
    if st == OK and (nchunks <= 0 or nchunks > MAX_CHUNKS):
        st = E_CHUNKS
    var filtered = False
    if st == OK and filter_on_addr != 0:
        for q in range(Int(nq)):
            var f = bptr(filter_on_addr)[unsafe_offset=q]
            st = E_FILTER if f > 1 else st
            filtered = filtered or f == 1
    if st == OK and filtered and (tags_addr == 0 or filter_tags_addr == 0):
        st = E_NULL
    if st != OK:
        return st
    var qn2 = query_norms(fptr(queries_addr), Int(nq), Int(dim), metric, st)
    if st != OK:
        return st

    initialize_runtime()
    var ni = Int(nq)
    var ki = Int(k)
    var nc = Int(nchunks)
    var di = Int(dim)
    var mcap = max(CAND_MIN, CAND_FACTOR * ki) if dtype == I8 else 1
    var wide = dtype == I8
    var buf_scores = List[Float32](length=0 if wide else nc * ni * ki, fill=0)
    var buf_ids = List[Int64](length=0 if wide else nc * ni * ki, fill=-1)
    var buf_len = List[Int64](length=nc * ni, fill=0)
    var up = List[Float64](length=nc * ni * ki if wide else 0, fill=0)
    var up_ids = List[Int64](length=nc * ni * ki if wide else 0, fill=-1)
    var up_len = List[Int64](length=nc * ni, fill=0)
    var lo = List[Float64](length=nc * ni * mcap if wide else 0, fill=0)
    var lo_ids = List[Int64](length=nc * ni * mcap if wide else 0, fill=-1)
    var lo_len = List[Int64](length=nc * ni, fill=0)
    var job = Job(
        data_addr, di * elem_bytes(dtype), fptr(aux_addr if wide else data_addr), Int(n), di,
        iptr(tags_addr if filtered else live_addr), bptr(live_addr),
        fptr(queries_addr), ni, fptr(Int(qn2.unsafe_ptr())), bptr(filter_on_addr if filtered else live_addr),
        iptr(filter_tags_addr if filtered else live_addr), filtered, ki, nc,
        fmut(Int(buf_scores.unsafe_ptr())), imut(Int(buf_ids.unsafe_ptr())), imut(Int(buf_len.unsafe_ptr())),
        fmut(out_scores_addr), mcap,
        dmut(Int(up.unsafe_ptr())), imut(Int(up_ids.unsafe_ptr())), imut(Int(up_len.unsafe_ptr())),
        dmut(Int(lo.unsafe_ptr())), imut(Int(lo_ids.unsafe_ptr())), imut(Int(lo_len.unsafe_ptr())),
    )
    if n > 0:
        st = run[False](job, metric, dtype)
    if st != OK:
        return st

    var ids = imut(out_ids_addr)
    var scores = fmut(out_scores_addr)
    var cands = 0
    var fallbacks = 0
    for q in range(ni):
        var length = 0
        if not wide:
            for c in range(nc):
                var o = (c * ni + q) * ki
                for j in range(o, o + Int(buf_len[c * ni + q])):
                    comptime if MUTANT_POST_FILTER:
                        if not job.matches(q, Int(buf_ids[j])):
                            continue
                    length = bstep(scores.unsafe_offset(q * ki), ids.unsafe_offset(q * ki), length, ki, buf_scores[j], buf_ids[j])
            counts[unsafe_offset=q] = Int64(length)
            continue
        var mu = List[Float64](length=ki, fill=0)
        var mu_ids = List[Int64](length=ki, fill=-1)
        var ml = List[Float64](length=mcap, fill=0)
        var ml_ids = List[Int64](length=mcap, fill=-1)
        var nu = 0
        var nl = 0
        for c in range(nc):
            var o = c * ni + q
            for j in range(Int(up_len[o])):
                nu = bstep(dmut(Int(mu.unsafe_ptr())), imut(Int(mu_ids.unsafe_ptr())), nu, ki, up[o * ki + j], up_ids[o * ki + j])
            for j in range(Int(lo_len[o])):
                nl = bstep(dmut(Int(ml.unsafe_ptr())), imut(Int(ml_ids.unsafe_ptr())), nl, mcap, lo[o * mcap + j], lo_ids[o * mcap + j])
        var tau = inf[DType.float64]()
        comptime if MUTANT_TAU_FROM_LOW:
            if nl >= ki:
                tau = ml[0]
                for j in range(ki):
                    tau = max(tau, ml[j])
        else:
            if nu == ki:
                tau = mu[0]
                for j in range(nu):
                    tau = max(tau, mu[j])
        var lo_max = -inf[DType.float64]()
        for j in range(nl):
            lo_max = max(lo_max, ml[j])
        var overflow = nl == mcap and lo_max <= tau
        comptime if MUTANT_NO_FALLBACK:
            overflow = False
        if overflow:
            fallbacks += 1
            var one_on = List[UInt8](length=1, fill=UInt8(1) if job.filtered and job.filter_on[unsafe_offset=q] == 1 else 0)
            var one_tag = List[Int64](length=1, fill=job.filter_tags[unsafe_offset=q] if job.filtered else 0)
            var one_cnt = List[Int64](length=1, fill=0)
            st = vdb_search_ex(
                orig_addr, F32, 0, 0, n, dim, tags_addr if tags_addr != 0 else live_addr, live_addr,
                queries_addr + q * di * 4, 1, Int(one_on.unsafe_ptr()), Int(one_tag.unsafe_ptr()),
                k, metric, nchunks, out_ids_addr + q * ki * 8, out_scores_addr + q * ki * 4, Int(one_cnt.unsafe_ptr()), 0,
            )
            _ = one_on^
            _ = one_tag^
            if st != OK:
                return st
            counts[unsafe_offset=q] = one_cnt[0]
            continue
        var qp = job.queries.unsafe_offset(q * di)
        for j in range(nl):
            if ml[j] > tau:
                continue
            cands += 1
            var id = ml_ids[j]
            var s = exact_score_m(metric, qp, qn2[q], orig_addr + Int(id) * di * 4, di)
            if not isfinite(s):
                return E_SCORE_NONFINITE
            length = bstep(scores.unsafe_offset(q * ki), ids.unsafe_offset(q * ki), length, ki, s, id)
        counts[unsafe_offset=q] = Int64(length)
        _ = mu^
        _ = mu_ids^
        _ = ml^
        _ = ml_ids^
    if stats_addr != 0:
        imut(stats_addr)[unsafe_offset=0] = Int64(cands)
        imut(stats_addr)[unsafe_offset=1] = Int64(fallbacks)
    _ = qn2^
    _ = buf_scores^
    _ = buf_ids^
    _ = buf_len^
    _ = up^
    _ = up_ids^
    _ = up_len^
    _ = lo^
    _ = lo_ids^
    _ = lo_len^
    return OK
