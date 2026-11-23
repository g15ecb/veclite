from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.sync import barrier
from max.gpu.primitives import warp
from std.memory import unsafe_stack_allocation
from std.memory.alloc import unsafe_alloc
from std.sys import size_of, get_defined_int, get_defined_bool
from kernel import OK, E_NULL, E_K, E_SCORE_NONFINITE, E_FILTER, MAX_K, F16, cand_le, bstep, check_shape, check_metric, check_batch, check_dtype, query_norms, sqnorm, finish, elem_bytes, fptr, fmut, iptr, imut, bptr, L2, DOT, COSINE, MUTANT_POST_FILTER, MUTANT_LOCAL_K_MINUS_ONE

comptime MUTANT_GPU_STALE_LIVE = get_defined_bool["VECLITE_MUTANT_GPU_STALE_LIVE"]()
comptime MUTANT_GPU_SKIP_GAP = get_defined_bool["VECLITE_MUTANT_GPU_SKIP_GAP"]()
comptime GPU_REFRESH_FRACTION = get_defined_int["VECLITE_GPU_REFRESH_FRACTION", 4]()
comptime LANES = 32
comptime WARPS = 8
comptime PER_LANE = 64
comptime CHUNK = LANES * PER_LANE
comptime MAX_S_FLOATS = 1 << 26

comptime GF = Pointer[Float32, MutUntrackedOrigin]
comptime GI = Pointer[Int64, MutUntrackedOrigin]
comptime GU = Pointer[UInt8, MutUntrackedOrigin]
comptime GS = Pointer[Int32, MutUntrackedOrigin]


def score_kernel[metric: Int32, D: DType, BM: Int, BN: Int, RM: Int, RN: Int, BK: Int](
    x: Pointer[Scalar[D], MutUntrackedOrigin], rn2: GF, q: GF, qn2: GF, s: GF, n64: Int64, nq32: Int32, dim32: Int32,
):
    comptime TX = BM // RM
    comptime TY = BN // RN
    comptime NT = TX * TY
    comptime EX = (BM * BK + NT - 1) // NT
    comptime EQ = (BN * BK + NT - 1) // NT
    comptime LD = BK + 1
    var n = Int32(n64)
    var nq = nq32
    var dim = dim32
    var xs = unsafe_stack_allocation[BM * LD, Float32, address_space=.SHARED]()
    var qs = unsafe_stack_allocation[BN * LD, Float32, address_space=.SHARED]()
    var tx = Int32(thread_idx.x)
    var ty = Int32(thread_idx.y)
    var tid = ty * Int32(TX) + tx
    var r0 = Int32(block_idx.x) * Int32(BM)
    var q0 = Int32(block_idx.y) * Int32(BN)
    var xp = Array[Pointer[Scalar[D], MutUntrackedOrigin], EX](fill=x)
    var xcol = SIMD[DType.int32, EX](0)
    var xok = SIMD[DType.bool, EX](fill=False)
    var xdst = SIMD[DType.int32, EX](0)
    comptime for e in range(EX):
        var l = tid + Int32(e * NT)
        var row = l // Int32(BK)
        var col = l % Int32(BK)
        xok[e] = l < Int32(BM * BK) and r0 + row < n
        xcol[e] = col
        xdst[e] = row * Int32(LD) + col
        xp[e] = x.unsafe_offset(Int(r0 + row) * Int(dim) + Int(col)) if xok[e] else x
    var qp = Array[GF, EQ](fill=q)
    var qcol = SIMD[DType.int32, EQ](0)
    var qok = SIMD[DType.bool, EQ](fill=False)
    var qdst = SIMD[DType.int32, EQ](0)
    comptime for e in range(EQ):
        var l = tid + Int32(e * NT)
        var row = l // Int32(BK)
        var col = l % Int32(BK)
        qok[e] = l < Int32(BN * BK) and q0 + row < nq
        qcol[e] = col
        qdst[e] = row * Int32(LD) + col
        qp[e] = q.unsafe_offset(Int(q0 + row) * Int(dim) + Int(col)) if qok[e] else q
    var xa = xs.unsafe_offset(Int(tx * Int32(LD)))
    var qa = qs.unsafe_offset(Int(ty * Int32(LD)))
    var acc = SIMD[DType.float32, RM * RN](0)
    var kd: Int32 = 0
    while kd < dim:
        comptime for e in range(EX):
            if l_ok(xok[e], kd + xcol[e], dim):
                xs[unsafe_offset=Int(xdst[e])] = xp[e][unsafe_offset=Int(kd)].cast[DType.float32]()
            elif e * NT < BM * BK:
                if Int(xdst[e]) < BM * LD:
                    xs[unsafe_offset=Int(xdst[e])] = 0
        comptime for e in range(EQ):
            if l_ok(qok[e], kd + qcol[e], dim):
                qs[unsafe_offset=Int(qdst[e])] = qp[e][unsafe_offset=Int(kd)]
            elif Int(qdst[e]) < BN * LD and tid + Int32(e * NT) < Int32(BN * BK):
                qs[unsafe_offset=Int(qdst[e])] = 0
        barrier()
        comptime for kk in range(BK):
            var xv = SIMD[DType.float32, RM](0)
            comptime for m in range(RM):
                xv[m] = xa[unsafe_offset=TX * m * LD + kk]
            comptime for j in range(RN):
                var qv = qa[unsafe_offset=TY * j * LD + kk]
                comptime for m in range(RM):
                    acc[j * RM + m] = qv.fma(xv[m], acc[j * RM + m])
        barrier()
        kd += Int32(BK)
    comptime for j in range(RN):
        var qi = q0 + ty + Int32(TY * j)
        if qi < nq:
            var so = s.unsafe_offset(Int(qi) * Int(n))
            var qq = qn2[unsafe_offset=Int(qi)]
            comptime for m in range(RM):
                var r = r0 + tx + Int32(TX * m)
                if r < n:
                    so[unsafe_offset=Int(r)] = finish[metric](acc[j * RM + m], qq, rn2[unsafe_offset=Int(r)])


@inline(.always)
def l_ok(ok: Bool, d: Int32, dim: Int32) -> Bool:
    return ok and d < dim


def gemv_kernel[metric: Int32, D: DType, RN: Int](
    x: Pointer[Scalar[D], MutUntrackedOrigin], rn2: GF, q: GF, qn2: GF, s: GF, n64: Int64, nq32: Int32, dim32: Int32,
):
    var n = Int(n64)
    var nq = Int(nq32)
    var dim = Int(dim32)
    var lane = Int(thread_idx.x) % LANES
    var row = Int(block_idx.x) * WARPS + Int(thread_idx.x) // LANES
    var q0 = Int(block_idx.y) * RN
    var acc = SIMD[DType.float32, RN](0)
    if row < n:
        var xr = x.unsafe_offset(row * dim)
        var d = lane * 4
        while d + 4 <= dim:
            var xv = xr.unsafe_offset(d).unsafe_load[width=4, alignment=1]().cast[DType.float32]()
            comptime for j in range(RN):
                if q0 + j < nq:
                    var qv = q.unsafe_offset((q0 + j) * dim + d).unsafe_load[width=4, alignment=1]()
                    acc[j] += (qv * xv).reduce_add()
            d += LANES * 4
        if d < dim:
            for e in range(d, dim):
                var xe = xr[unsafe_offset=e].cast[DType.float32]()
                comptime for j in range(RN):
                    if q0 + j < nq:
                        acc[j] = q[unsafe_offset=(q0 + j) * dim + e].fma(xe, acc[j])
    comptime for j in range(RN):
        comptime for step in range(5):
            acc[j] += warp.shuffle_xor(acc[j], UInt32(1 << step))
    if row < n and lane == 0:
        comptime for j in range(RN):
            if q0 + j < nq:
                s[unsafe_offset=(q0 + j) * n + row] = finish[metric](acc[j], qn2[unsafe_offset=q0 + j], rn2[unsafe_offset=row])


@inline(.always)
def better(sa: Float32, ia: Int32, sb: Float32, ib: Int32) -> Bool:
    return ia >= 0 and (ib < 0 or cand_le(sa, Int64(ia), sb, Int64(ib)))


def select_kernel(
    s: GF, n64: Int64, live: GU, tags: GI, filter_on: GU, filter_tags: GI, filtered: Int32, k32: Int32,
    out_s: GF, out_i: GI, out_c: GS, status: GS,
):
    var n = Int(n64)
    var k = Int(k32)
    var chunk = Int(block_idx.x) * WARPS + Int(thread_idx.x) // LANES
    var qi = Int(block_idx.y)
    var nchunks = (n + CHUNK - 1) // CHUNK
    var lane = Int(thread_idx.x) % LANES
    if chunk >= nchunks:
        return
    var base = chunk * CHUNK
    var o = qi * nchunks + chunk
    var use_filter = filtered != 0 and filter_on[unsafe_offset=qi] != 0
    var ftag = filter_tags[unsafe_offset=qi]
    var vals = SIMD[DType.float32, PER_LANE](0)
    var ok = SIMD[DType.bool, PER_LANE](fill=False)
    var bad = False
    comptime for t in range(PER_LANE):
        var r = base + lane + LANES * t
        if r < n and live[unsafe_offset=r] != 0:
            var keep = True
            comptime if not MUTANT_POST_FILTER:
                keep = not use_filter or tags[unsafe_offset=r] == ftag
            if keep:
                var v = s[unsafe_offset=qi * n + r]
                if v != v or v - v != 0:
                    bad = True
                else:
                    vals[t] = v
                    ok[t] = True
    var ts: Float32 = 0
    var ti: Int32 = -1
    var count = 0
    for j in range(k):
        var bs: Float32 = 0
        var bi: Int32 = -1
        comptime for t in range(PER_LANE):
            var li = Int32(lane + LANES * t)
            if ok[t] and (ti < 0 or not cand_le(vals[t], Int64(li), ts, Int64(ti))) and better(vals[t], li, bs, bi):
                bs = vals[t]
                bi = li
        comptime for step in range(5):
            comptime off = UInt32(1 << step)
            var os = warp.shuffle_xor(bs, off)
            var oi = warp.shuffle_xor(bi, off)
            if better(os, oi, bs, bi):
                bs = os
                bi = oi
        if bi < 0:
            break
        ts = bs
        ti = bi
        if lane == 0:
            out_s[unsafe_offset=o * k + j] = bs
            out_i[unsafe_offset=o * k + j] = Int64(base) + Int64(bi)
        count = j + 1
    if bad:
        status[unsafe_offset=o] = 1
    if lane == 0:
        out_c[unsafe_offset=o] = Int32(count)


struct Gpu(Movable):
    var ctx: DeviceContext
    var x: DeviceBuffer[DType.uint8]
    var rn2: DeviceBuffer[DType.float32]
    var live: DeviceBuffer[DType.uint8]
    var tags: DeviceBuffer[DType.int64]
    var work: DeviceBuffer[DType.uint8]
    var work_len: Int
    var sbuf: DeviceBuffer[DType.float32]
    var s_len: Int
    var n: Int
    var dim: Int
    var dtype: Int32

    def __init__(out self, var ctx: DeviceContext, var x: DeviceBuffer[DType.uint8], var rn2: DeviceBuffer[DType.float32],
                 var live: DeviceBuffer[DType.uint8], var tags: DeviceBuffer[DType.int64], var work: DeviceBuffer[DType.uint8],
                 var sbuf: DeviceBuffer[DType.float32], n: Int, dim: Int, dtype: Int32):
        self.ctx = ctx^
        self.x = x^
        self.rn2 = rn2^
        self.live = live^
        self.tags = tags^
        self.work = work^
        self.work_len = 1
        self.sbuf = sbuf^
        self.s_len = 1
        self.n = n
        self.dim = dim
        self.dtype = dtype


def gp(h: Int) -> Pointer[Gpu, MutUntrackedOrigin]:
    return Pointer[Gpu, MutUntrackedOrigin](unsafe_from_address=h)


def write_into[D: DType](b: DeviceBuffer[D], src: Int, count: Int) raises:
    if count > 0:
        with b.map_to_host() as h:
            copy_bytes(Int(h.unsafe_ptr()), src, count * size_of[Scalar[D]]())


def upload[D: DType](ctx: DeviceContext, src: Int, count: Int) raises -> DeviceBuffer[D]:
    var b = ctx.enqueue_create_buffer[D](max(count, 1))
    write_into(b, src, count)
    return b^


def gpu_open_impl(data: Int, dtype: Int32, n: Int, dim: Int, tags: Int, live: Int, out_h: Int) raises -> Int32:
    var ctx = DeviceContext()
    var rn = List[Float32](length=max(n, 1), fill=0)
    var tmp = List[Float32](length=dim, fill=0)
    for r in range(n):
        if dtype == F16:
            var h = Pointer[Float16, MutUntrackedOrigin](unsafe_from_address=data + r * dim * 2)
            for d in range(dim):
                tmp[d] = h[unsafe_offset=d].cast[DType.float32]()
            rn[r] = sqnorm(fptr(Int(tmp.unsafe_ptr())), dim)
        else:
            rn[r] = sqnorm(fptr(data + r * dim * 4), dim)
    var xb = upload[DType.uint8](ctx, data, n * dim * elem_bytes(dtype))
    var rb = upload[DType.float32](ctx, Int(rn.unsafe_ptr()), n)
    var lb = upload[DType.uint8](ctx, live, n)
    var tb = upload[DType.int64](ctx, tags, n)
    var wb = ctx.enqueue_create_buffer[DType.uint8](1)
    var sb = ctx.enqueue_create_buffer[DType.float32](1)
    ctx.synchronize()
    var p = unsafe_alloc[Gpu](1)
    p.unsafe_write(Gpu(ctx^, xb^, rb^, lb^, tb^, wb^, sb^, n, dim, dtype))
    st64(out_h, UInt64(Int(p)))
    _ = rn^
    _ = tmp^
    return OK


@export("veclite_gpu_open")
def veclite_gpu_open(data: Int, dtype: Int32, n: Int64, dim: Int64, tags: Int, live: Int, out_h: Int) abi("C") -> Int32:
    if out_h == 0:
        return E_NULL
    st64(out_h, 0)
    var st = check_shape(data, n, dim)
    if st == OK:
        st = check_dtype(dtype)
    if st == OK and (tags == 0 or live == 0):
        st = E_NULL
    if st != OK:
        return st
    try:
        return gpu_open_impl(data, dtype, Int(n), Int(dim), tags, live, out_h)
    except:
        return E_GPU


@export("veclite_gpu_update")
def veclite_gpu_update(h: Int, tags: Int, live: Int) abi("C") -> Int32:
    if h == 0 or tags == 0 or live == 0:
        return E_NULL
    try:
        write_into(gp(h)[].live, live, gp(h)[].n)
        write_into(gp(h)[].tags, tags, gp(h)[].n)
    except:
        return E_GPU
    return OK


@export("veclite_gpu_close")
def veclite_gpu_close(h: Int) abi("C") -> Int32:
    if h == 0:
        return E_NULL
    var p = gp(h)
    _ = p.unsafe_take_pointee()
    p.unsafe_free()
    return OK


comptime CFG_SMALL = get_defined_int["VECLITE_GPU_SMALL", 0]()
comptime CFG_LARGE = get_defined_int["VECLITE_GPU_LARGE", 1]()


def gemm[metric: Int32, D: DType, BM: Int, BN: Int, RM: Int, RN: Int, BK: Int](
    g: Gpu, xp: Pointer[Scalar[D], MutUntrackedOrigin], rp: GF, q: GF, qn: GF, s: GF, nq: Int
) raises:
    g.ctx.enqueue_function[score_kernel[metric, D, BM, BN, RM, RN, BK]](
        xp, rp, q, qn, s, Int64(g.n), Int32(nq), Int32(g.dim),
        grid_dim=((g.n + BM - 1) // BM, (nq + BN - 1) // BN), block_dim=(BM // RM, BN // RN))


def launch_scores[metric: Int32, D: DType](g: Gpu, xp: Pointer[Scalar[D], MutUntrackedOrigin], q: GF, qn: GF, s: GF, nq: Int) raises:
    var rp = GF(unsafe_from_address=Int(g.rn2.unsafe_ptr()))
    if nq == 1:
        g.ctx.enqueue_function[gemv_kernel[metric, D, 1]](
            xp, rp, q, qn, s, Int64(g.n), Int32(nq), Int32(g.dim), grid_dim=((g.n + WARPS - 1) // WARPS, 1), block_dim=WARPS * LANES)
    elif nq <= 16:
        comptime if CFG_SMALL == 0:
            gemm[metric, D, 128, 16, 8, 1, 8](g, xp, rp, q, qn, s, nq)
        elif CFG_SMALL == 1:
            gemm[metric, D, 256, 16, 16, 1, 8](g, xp, rp, q, qn, s, nq)
        elif CFG_SMALL == 2:
            gemm[metric, D, 128, 16, 4, 2, 16](g, xp, rp, q, qn, s, nq)
        else:
            gemm[metric, D, 64, 16, 4, 4, 16](g, xp, rp, q, qn, s, nq)
    elif nq <= 64:
        gemm[metric, D, 128, 64, 8, 4, 8](g, xp, rp, q, qn, s, nq)
    else:
        comptime if CFG_LARGE == 0:
            gemm[metric, D, 64, 64, 4, 4, 16](g, xp, rp, q, qn, s, nq)
        elif CFG_LARGE == 1:
            gemm[metric, D, 128, 64, 8, 4, 8](g, xp, rp, q, qn, s, nq)
        else:
            gemm[metric, D, 128, 128, 8, 8, 8](g, xp, rp, q, qn, s, nq)


def run_scores[metric: Int32](g: Gpu, q: GF, qn: GF, s: GF, nq: Int) raises:
    if g.dtype == F16:
        launch_scores[metric, DType.float16](g, Pointer[Float16, MutUntrackedOrigin](unsafe_from_address=Int(g.x.unsafe_ptr())), q, qn, s, nq)
    else:
        launch_scores[metric, DType.float32](g, GF(unsafe_from_address=Int(g.x.unsafe_ptr())), q, qn, s, nq)


def align8(v: Int) -> Int:
    return (v + 7) // 8 * 8


def gpu_search_impl(
    h: Int, queries: Int, nq: Int, filter_on: Int, filter_tags: Int, filtered: Bool, k: Int, metric: Int32,
    qn2: Int, out_ids: Int, out_scores: Int, out_counts: Int, out_full: Int,
) raises -> Int32:
    var g = gp(h)
    var n = g[].n
    var dim = g[].dim
    var nchunks = (n + CHUNK - 1) // CHUNK
    var kloc = k - 1 if MUTANT_LOCAL_K_MINUS_ONE else k
    var kc = max(kloc, 1)
    var group = max(1, min(nq, MAX_S_FLOATS // max(n, 1)))
    var o_q = 0
    var o_qn = align8(o_q + group * dim * 4)
    var o_fo = align8(o_qn + group * 4)
    var o_ft = align8(o_fo + group)
    var o_os = align8(o_ft + group * 8)
    var o_oi = align8(o_os + group * nchunks * kc * 4)
    var o_oc = align8(o_oi + group * nchunks * kc * 8)
    var o_st = align8(o_oc + group * nchunks * 4)
    var total = align8(o_st + group * nchunks * 4)
    if g[].work_len < total:
        g[].work = g[].ctx.enqueue_create_buffer[DType.uint8](total)
        g[].work_len = total
    if g[].s_len < group * n:
        g[].sbuf = g[].ctx.enqueue_create_buffer[DType.float32](group * n)
        g[].s_len = group * n
    var wa = Int(g[].work.unsafe_ptr())
    var st = OK
    var g0 = 0
    while g0 < nq and st == OK:
        var gn = min(group, nq - g0)
        with g[].work.map_to_host() as hw:
            var a = Int(hw.unsafe_ptr())
            copy_bytes(a + o_q, queries + g0 * dim * 4, gn * dim * 4)
            copy_bytes(a + o_qn, qn2 + g0 * 4, gn * 4)
            if filtered:
                copy_bytes(a + o_fo, filter_on + g0, gn)
                copy_bytes(a + o_ft, filter_tags + g0 * 8, gn * 8)
            zero_bytes(a + o_oc, gn * nchunks * 4)
            zero_bytes(a + o_st, gn * nchunks * 4)
        var s = GF(unsafe_from_address=Int(g[].sbuf.unsafe_ptr()))
        if metric == L2:
            run_scores[L2](g[], GF(unsafe_from_address=wa + o_q), GF(unsafe_from_address=wa + o_qn), s, gn)
        elif metric == DOT:
            run_scores[DOT](g[], GF(unsafe_from_address=wa + o_q), GF(unsafe_from_address=wa + o_qn), s, gn)
        else:
            run_scores[COSINE](g[], GF(unsafe_from_address=wa + o_q), GF(unsafe_from_address=wa + o_qn), s, gn)
        if kloc > 0:
            g[].ctx.enqueue_function[select_kernel](
                s, Int64(n), GU(unsafe_from_address=Int(g[].live.unsafe_ptr())), GI(unsafe_from_address=Int(g[].tags.unsafe_ptr())), GU(unsafe_from_address=wa + o_fo),
                GI(unsafe_from_address=wa + o_ft), Int32(1 if filtered else 0), Int32(kloc),
                GF(unsafe_from_address=wa + o_os), GI(unsafe_from_address=wa + o_oi), GS(unsafe_from_address=wa + o_oc),
                GS(unsafe_from_address=wa + o_st), grid_dim=((nchunks + WARPS - 1) // WARPS, gn), block_dim=WARPS * LANES,
            )
        g[].ctx.synchronize()
        if out_full != 0:
            with g[].sbuf.map_to_host() as hf:
                copy_bytes(out_full + g0 * n * 4, Int(hf.unsafe_ptr()), gn * n * 4)
        with g[].work.map_to_host() as hw:
            var a = Int(hw.unsafe_ptr())
            var stat = GS(unsafe_from_address=a + o_st)
            for i in range(gn * nchunks):
                if stat[unsafe_offset=i] != 0:
                    st = E_SCORE_NONFINITE
            if st == OK:
                var cs = GF(unsafe_from_address=a + o_os)
                var ci = GI(unsafe_from_address=a + o_oi)
                var cc = GS(unsafe_from_address=a + o_oc)
                for qq in range(gn):
                    var length = 0
                    for c in range(nchunks):
                        var o = qq * nchunks + c
                        for j in range(Int(cc[unsafe_offset=o])):
                            length = bstep(fmut(out_scores).unsafe_offset((g0 + qq) * k), imut(out_ids).unsafe_offset((g0 + qq) * k),
                                           length, k, cs[unsafe_offset=o * kloc + j], ci[unsafe_offset=o * kloc + j])
                    imut(out_counts)[unsafe_offset=g0 + qq] = Int64(length)
        g0 += gn
    return st


@export("veclite_gpu_search")
def veclite_gpu_search(
    h: Int, queries: Int, nq: Int64, filter_on: Int, filter_tags: Int,
    k: Int64, metric: Int32, out_ids: Int, out_scores: Int, out_counts: Int, out_full: Int,
) abi("C") -> Int32:
    if out_counts == 0:
        return E_NULL
    var st = check_batch(nq)
    if st != OK:
        return st
    for q in range(Int(nq)):
        imut(out_counts)[unsafe_offset=q] = 0
    if h == 0 or queries == 0 or out_ids == 0 or out_scores == 0:
        return E_NULL
    if k <= 0 or k > MAX_K:
        return E_K
    st = check_metric(metric)
    if st != OK:
        return st
    var filtered = False
    if filter_on != 0:
        for q in range(Int(nq)):
            var f = bptr(filter_on)[unsafe_offset=q]
            if f > 1:
                return E_FILTER
            filtered = filtered or f == 1
    if filtered and filter_tags == 0:
        return E_NULL
    var qn2 = query_norms(fptr(queries), Int(nq), gp(h)[].dim, metric, st)
    if st != OK:
        return st
    if gp(h)[].n == 0:
        return OK
    try:
        st = gpu_search_impl(h, queries, Int(nq), filter_on, filter_tags, filtered, Int(k), metric,
                             Int(qn2.unsafe_ptr()), out_ids, out_scores, out_counts, out_full)
    except:
        st = E_GPU
    if st != OK:
        for q in range(Int(nq)):
            imut(out_counts)[unsafe_offset=q] = 0
    _ = qn2^
    return st

def check_device(device: Int32, min_batch: Int64, nq: Int64, filter_on: Int, mut filtered: Bool) -> Int32:
    if device < DEV_AUTO or device > DEV_GPU or min_batch < 1:
        return E_ARG
    filtered = False
    if filter_on != 0:
        for q in range(Int(nq)):
            var f = bptr(filter_on)[unsafe_offset=q]
            if f > 1:
                return E_FILTER
            filtered = filtered or f == 1
    return OK


def gpu_detect() -> Bool:
    try:
        var c = DeviceContext()
        _ = c^
        return True
    except:
        return False


@export("veclite_gpu_available")
def veclite_gpu_available() abi("C") -> Int32:
    return 1 if gpu_detect() else 0


@export("veclite_close")
def veclite_close(h: Int) abi("C") -> Int32:
    if h != 0 and dbp(h)[].gpu > 0:
        _ = veclite_gpu_close(dbp(h)[].gpu)
        dbp(h)[].gpu = 0
    return close_db(h)


def gpu_refresh(mut db: Db) raises:
    var rows = db.main_rows
    var vec = List[UInt8](length=max(rows * db.enc_bytes(), 1), fill=0)
    var tags = List[Int64](length=max(rows, 1), fill=0)
    var live = List[UInt8](length=max(rows, 1), fill=1)
    for i in range(rows):
        live[i] = db.live[i]
    if not copy_main_rows(db, rows, Int(vec.unsafe_ptr()), Int(tags.unsafe_ptr())):
        raise Error("read main rows")
    if db.gpu > 0:
        _ = veclite_gpu_close(db.gpu)
        db.gpu = 0
        db.gpu_rows = 0
    var out = List[Int64](length=1, fill=0)
    if gpu_open_impl(Int(vec.unsafe_ptr()), db.dtype, rows, db.dim, Int(tags.unsafe_ptr()), Int(live.unsafe_ptr()), Int(out.unsafe_ptr())) != OK:
        raise Error("gpu open")
    db.gpu = Int(out[0])
    db.gpu_rows = rows
    _ = vec^
    _ = tags^
    _ = live^


@export("veclite_reader_search_dev")
def veclite_reader_search_dev(
    h: Int, r: Int64, queries: Int, nq: Int64, filter_on: Int, filter_tags: Int, k: Int64, metric: Int32, nchunks: Int64,
    device: Int32, min_batch: Int64, out_ids: Int, out_scores: Int, out_counts: Int, out_used: Int,
) abi("C") -> Int32:
    if out_used != 0:
        GS(unsafe_from_address=out_used)[unsafe_offset=0] = 0
    var st = check_search(h, r, queries, nq, k, out_ids, out_scores, out_counts)
    var filtered = False
    if st == OK:
        st = check_device(device, min_batch, nq, filter_on, filtered)
    if st != OK:
        return st
    var db = dbp(h)
    var want = device == DEV_GPU or (device == DEV_AUTO and nq >= min_batch and not filtered and db[].main_rows > 0)
    if want and db[].gpu_avail == 0:
        db[].gpu_avail = 1 if gpu_detect() else 2
    if want and db[].gpu_avail != 1:
        if device == DEV_GPU:
            return E_NO_GPU
        want = False
    if not want or db[].main_rows == 0:
        return search_cpu(h, r, queries, nq, filter_on, filter_tags, k, metric, nchunks, out_ids, out_scores, out_counts)
    try:
        if db[].gpu == 0 or (db[].main_rows - db[].gpu_rows) * GPU_REFRESH_FRACTION > db[].gpu_rows:
            gpu_refresh(db[])
        var v = make_view(db[], Int(r))
        var split = min(db[].gpu_rows, v.main_rows)
        comptime if not MUTANT_GPU_STALE_LIVE:
            write_into(gp(db[].gpu)[].live, Int(v.live.unsafe_ptr()), split)
        st = veclite_gpu_search(db[].gpu, queries, nq, filter_on, filter_tags, k, metric, out_ids, out_scores, out_counts, 0)
        if st == OK:
            var from_row = v.main_rows if MUTANT_GPU_SKIP_GAP else split
            st = search_rows(db[], Int(r), v, from_row, queries, nq, filter_on, filter_tags, k, metric, nchunks, out_ids, out_scores, out_counts)
        _ = v^
    except:
        st = E_GPU
    if st != OK:
        for q in range(Int(nq)):
            imut(out_counts)[unsafe_offset=q] = 0
        return st
    if out_used != 0:
        GS(unsafe_from_address=out_used)[unsafe_offset=0] = 1
    return OK
