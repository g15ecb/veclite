from std.ffi import external_call, c_int
from std.sys import get_defined_bool, CompilationTarget
from std.utils.numerics import isfinite
from std.memory.alloc import unsafe_alloc
from kernel import veclite_scores, veclite_validate_rows, veclite_abi_version, OK, E_NULL, E_DIM, E_K, E_BATCH, E_DATA_NONFINITE, MAX_DIM, MAX_K, MAX_BATCH, bstep, veclite_search, fptr, fmut, iptr, imut, bptr

comptime E_IO: Int32 = -20
comptime E_CORRUPT: Int32 = -21
comptime E_STATE: Int32 = -22
comptime E_ID: Int32 = -23
comptime E_READER: Int32 = -24
comptime E_ARG: Int32 = -25
comptime E_GPU: Int32 = -30
comptime E_NO_GPU: Int32 = -31
comptime DEV_AUTO: Int32 = 0
comptime DEV_CPU: Int32 = 1
comptime DEV_GPU: Int32 = 2

comptime MUTANT_ACK_BEFORE_SYNC = get_defined_bool["VECLITE_MUTANT_ACK_BEFORE_SYNC"]()
comptime MUTANT_CKPT_IGNORES_READERS = get_defined_bool["VECLITE_MUTANT_CKPT_IGNORES_READERS"]()
comptime MUTANT_NO_CHAIN = get_defined_bool["VECLITE_MUTANT_NO_CHAIN"]()
comptime MUTANT_RESTART_EARLY = get_defined_bool["VECLITE_MUTANT_RESTART_EARLY"]()
comptime MUTANT_SKIP_DATA_SYNC = get_defined_bool["VECLITE_MUTANT_SKIP_DATA_SYNC"]()
comptime MUTANT_UNSYNCED_RESTART_HDR = get_defined_bool["VECLITE_MUTANT_UNSYNCED_RESTART_HDR"]()
comptime MUTANT_NO_RECOVERY_TRUNCATE = get_defined_bool["VECLITE_MUTANT_NO_RECOVERY_TRUNCATE"]()
comptime MUTANT_ROLLBACK_KEEP_PENDING = get_defined_bool["VECLITE_MUTANT_ROLLBACK_KEEP_PENDING"]()
comptime MUTANT_ROLLBACK_KEEP_ROWS = get_defined_bool["VECLITE_MUTANT_ROLLBACK_KEEP_ROWS"]()

comptime MAIN_MAGIC: UInt64 = 0x314E49414D424456
comptime WAL_MAGIC: UInt64 = 0x31304C4157424456
comptime VERSION: UInt64 = 2
comptime SLOT_BYTES = 4096
comptime DATA_OFF = 2 * SLOT_BYTES
comptime WAL_HDR = 64
comptime FRAME_HDR = 64
comptime SECTOR = 512
comptime MAX_READERS = 64
comptime MAX_EXTENT_ROWS: Int64 = 1 << 24
comptime APPEND: UInt64 = 1
comptime DELETE: UInt64 = 2
comptime MAIN = 0
comptime O_RDWR_CREAT = 0x202 if CompilationTarget.is_macos() else 0x42
comptime F_FULLFSYNC = 51
comptime WALF = 1

comptime U64Mut = Pointer[UInt64, MutUntrackedOrigin]
comptime U8Mut = Pointer[UInt8, MutUntrackedOrigin]


def ld64(a: Int) -> UInt64:
    return U64Mut(unsafe_from_address=a).unsafe_load[alignment=1]()


def st64(a: Int, v: UInt64):
    U64Mut(unsafe_from_address=a).unsafe_store[alignment=1](0, v)


def ld8(a: Int) -> UInt8:
    return U8Mut(unsafe_from_address=a)[unsafe_offset=0]


def st8(a: Int, v: UInt8):
    U8Mut(unsafe_from_address=a)[unsafe_offset=0] = v


def copy_bytes(dst: Int, src: Int, n: Int):
    if n > 0:
        _ = external_call["memmove", Int](dst, src, n)


def same_bytes(a: Int, b: Int, n: Int) -> Bool:
    return n == 0 or external_call["memcmp", c_int](a, b, n) == 0


def zero_bytes(dst: Int, n: Int):
    if n > 0:
        _ = external_call["memset", Int](dst, c_int(0), n)


def mix(h: UInt64, w: UInt64) -> UInt64:
    var x = (h ^ w) * 0x9E3779B97F4A7C15
    return x ^ (x >> 29)


def hash_bytes(h0: UInt64, a: Int, n: Int) -> UInt64:
    var h = h0
    var i = 0
    while i + 8 <= n:
        h = mix(h, ld64(a + i))
        i += 8
    var tail = UInt64(n)
    for j in range(i, n):
        tail ^= UInt64(ld8(a + j)) << UInt64(8 * (j - i + 1))
    return mix(h, tail)


def seed_of(salt: UInt64) -> UInt64:
    return mix(0x5EED5EED5EED5EED, salt)


def grow(mut b: List[UInt8], n: Int):
    if len(b) < n:
        b.resize(n, 0)


@fieldwise_init
struct SimFile(Movable):
    var cache: List[UInt8]
    var disk: List[UInt8]
    var w_off: List[Int]
    var w_len: List[Int]
    var w_src: List[Int]
    var log: List[UInt8]

    def write(mut self, off: Int, src: Int, n: Int):
        grow(self.cache, off + n)
        copy_bytes(Int(self.cache.unsafe_ptr()) + off, src, n)
        self.w_off.append(off)
        self.w_len.append(n)
        self.w_src.append(len(self.log))
        var at = len(self.log)
        grow(self.log, at + n)
        copy_bytes(Int(self.log.unsafe_ptr()) + at, src, n)

    def truncate(mut self, n: Int):
        self.cache.resize(n, 0)
        self.w_off.append(n)
        self.w_len.append(-1)
        self.w_src.append(0)

    def sync(mut self):
        self.disk = self.cache.copy()
        self.forget()

    def forget(mut self):
        self.w_off.clear()
        self.w_len.clear()
        self.w_src.clear()
        self.log.clear()

    def crash(mut self, mut rng: UInt64):
        var d = self.disk.copy()
        for i in range(len(self.w_off)):
            var off = self.w_off[i]
            var n = self.w_len[i]
            if n < 0:
                if next_rand(rng) % 2 == 0:
                    d.resize(off, 0)
                continue
            var s = 0
            while s < n:
                var e = min(n, (((off + s) // SECTOR) + 1) * SECTOR - off)
                var pick = next_rand(rng) % 3
                var m = 0 if pick == 0 else (e - s if pick == 1 else Int(next_rand(rng) % UInt64(e - s)))
                if m > 0:
                    grow(d, off + s + m)
                    copy_bytes(Int(d.unsafe_ptr()) + off + s, Int(self.log.unsafe_ptr()) + self.w_src[i] + s, m)
                s = e
        self.disk = d.copy()
        self.cache = d^
        self.forget()

    def crash_range(mut self, lo: Int, hi: Int):
        var d = self.disk.copy()
        for i in range(len(self.w_off)):
            var n = self.w_len[i]
            var a = max(0, lo)
            var b = min(n, hi)
            if n > 0 and b > a:
                grow(d, self.w_off[i] + b)
                copy_bytes(Int(d.unsafe_ptr()) + self.w_off[i] + a, Int(self.log.unsafe_ptr()) + self.w_src[i] + a, b - a)
        self.disk = d.copy()
        self.cache = d^
        self.forget()


def next_rand(mut s: UInt64) -> UInt64:
    s ^= s << 13
    s ^= s >> 7
    s ^= s << 17
    return s


@fieldwise_init
struct Sim(Movable):
    var main: SimFile
    var wal: SimFile
    var budget: Int
    var mutations: Int

    def write(mut self, f: Int, off: Int, src: Int, n: Int):
        if f == MAIN:
            self.main.write(off, src, n)
        else:
            self.wal.write(off, src, n)

    def truncate(mut self, f: Int, n: Int):
        if f == MAIN:
            self.main.truncate(n)
        else:
            self.wal.truncate(n)

    def sync(mut self, f: Int):
        if f == MAIN:
            self.main.sync()
        else:
            self.wal.sync()

    def cache(self, f: Int) -> Tuple[Int, Int]:
        if f == MAIN:
            return (Int(self.main.cache.unsafe_ptr()), len(self.main.cache))
        return (Int(self.wal.cache.unsafe_ptr()), len(self.wal.cache))


def new_file() -> SimFile:
    return SimFile(List[UInt8](), List[UInt8](), List[Int](), List[Int](), List[Int](), List[UInt8]())


@fieldwise_init
struct Db(Movable):
    var sim: Int
    var fds: List[Int]
    var dim: Int
    var dtype: Int32
    var extent_rows: Int
    var extent_bytes: Int
    var hdr_seq: UInt64
    var hdr_slot: Int
    var hdr_rows: Int
    var hdr_txn: Int
    var hdr_durable_txn: Int
    var main_rows: Int
    var salt: UInt64
    var salt_synced: Bool
    var chain_tail: UInt64
    var f_txn: List[Int]
    var f_kind: List[UInt64]
    var f_id: List[Int]
    var f_salt: List[UInt64]
    var f_vec: List[Int]
    var a_vec: List[Float32]
    var a_tag: List[Int64]
    var a_enc: List[UInt8]
    var ntx: Int
    var published: Int
    var pub_slot: Int
    var rows: Int
    var live: List[UInt8]
    var p_kind: List[UInt64]
    var p_id: List[Int]
    var p_tag: List[Int64]
    var p_vec: List[Float32]
    var r_active: List[Bool]
    var r_mark: List[Int]
    var r_salt: List[UInt64]
    var r_max_slot: List[Int]
    var poisoned: Bool
    var map_addr: Int
    var map_len: Int
    var gpu: Int
    var gpu_rows: Int
    var gpu_avail: Int

    def frame_bytes(self) -> Int:
        return FRAME_HDR + self.dim * 4

    def row_off(self, id: Int) -> Int:
        return DATA_OFF + (id // self.extent_rows) * self.extent_bytes

    def enc_bytes(self) -> Int:
        return self.dim * elem_bytes(self.dtype)

    def set_extent_bytes(mut self):
        self.extent_bytes = ((self.extent_rows * (self.enc_bytes() + 9) + SLOT_BYTES - 1) // SLOT_BYTES) * SLOT_BYTES

    def vec_off(self, id: Int) -> Int:
        return self.row_off(id) + (id % self.extent_rows) * self.enc_bytes()

    def tag_off(self, id: Int) -> Int:
        return self.row_off(id) + self.extent_rows * self.enc_bytes() + (id % self.extent_rows) * 8

    def live_off(self, id: Int) -> Int:
        return self.row_off(id) + self.extent_rows * (self.enc_bytes() + 8) + (id % self.extent_rows)

    def format_word(self) -> UInt64:
        return VERSION | (UInt64(Int(self.dtype)) << 16) | (UInt64(self.dim) << 32)

    def simp(self) -> Pointer[Sim, MutUntrackedOrigin]:
        return Pointer[Sim, MutUntrackedOrigin](unsafe_from_address=self.sim)

    def mutation(mut self) -> Bool:
        if self.sim == 0:
            return True
        var s = self.simp()
        if s[].budget == 0:
            self.poisoned = True
            return False
        if s[].budget > 0:
            s[].budget -= 1
        s[].mutations += 1
        return True

    def wr(mut self, f: Int, off: Int, src: Int, n: Int) -> Bool:
        if not self.mutation():
            return False
        if self.sim != 0:
            self.simp()[].write(f, off, src, n)
            return True
        var done = 0
        while done < n:
            var w = external_call["pwrite", Int](c_int(self.fds[f]), src + done, n - done, off + done)
            if w <= 0:
                self.poisoned = True
                return False
            done += w
        return True

    def sync(mut self, f: Int) -> Bool:
        if not self.mutation():
            return False
        if self.sim != 0:
            self.simp()[].sync(f)
            return True
        comptime if CompilationTarget.is_macos():
            if external_call["fcntl", c_int, num_fixed_args=2](c_int(self.fds[f]), c_int(F_FULLFSYNC)) == 0:
                return True
        if external_call["fsync", c_int](c_int(self.fds[f])) == 0:
            return True
        self.poisoned = True
        return False

    def trunc(mut self, f: Int, n: Int) -> Bool:
        if not self.mutation():
            return False
        if self.sim != 0:
            self.simp()[].truncate(f, n)
            return True
        if external_call["ftruncate", c_int](c_int(self.fds[f]), n) == 0:
            return True
        self.poisoned = True
        return False

    def fsize(self, f: Int) -> Int:
        if self.sim != 0:
            return self.simp()[].cache(f)[1]
        return external_call["lseek", Int](c_int(self.fds[f]), 0, c_int(2))

    def rd(self, f: Int, off: Int, dst: Int, n: Int) -> Bool:
        if off < 0 or n < 0 or off + n > self.fsize(f):
            return False
        if self.sim != 0:
            copy_bytes(dst, self.simp()[].cache(f)[0] + off, n)
            return True
        var done = 0
        while done < n:
            var r = external_call["pread", Int](c_int(self.fds[f]), dst + done, n - done, off + done)
            if r <= 0:
                return False
            done += r
        return True

    def map_main(mut self, size: Int) -> Int:
        if self.sim != 0:
            return self.simp()[].cache(MAIN)[0]
        if size <= self.map_len:
            return self.map_addr
        self.unmap_main()
        var want = max(size * 2, 1 << 30)
        var a = external_call["mmap", Int](0, want, c_int(1), c_int(1), c_int(self.fds[MAIN]), 0)
        if a != -1:
            self.map_addr = a
            self.map_len = want
        return a

    def unmap_main(mut self):
        if self.map_len > 0:
            _ = external_call["munmap", c_int](self.map_addr, self.map_len)
        self.map_addr = 0
        self.map_len = 0

    def write_main_header(mut self, rows: Int, txn: Int) -> Bool:
        var b = List[UInt8](length=64, fill=0)
        var a = Int(b.unsafe_ptr())
        st64(a, MAIN_MAGIC)
        st64(a + 8, self.format_word())
        st64(a + 16, UInt64(self.extent_rows))
        st64(a + 24, UInt64(rows))
        st64(a + 32, UInt64(txn))
        st64(a + 40, self.hdr_seq + 1)
        st64(a + 48, hash_bytes(MAIN_MAGIC, a, 48))
        var slot = 1 - self.hdr_slot
        var ok = self.wr(MAIN, slot * SLOT_BYTES, a, 64)
        _ = b^
        if not ok:
            return False
        self.hdr_slot = slot
        self.hdr_seq += 1
        self.hdr_rows = rows
        self.hdr_txn = txn
        return True

    def write_wal_header(mut self) -> Bool:
        var b = List[UInt8](length=WAL_HDR, fill=0)
        var a = Int(b.unsafe_ptr())
        st64(a, WAL_MAGIC)
        st64(a + 8, self.format_word())
        st64(a + 16, self.salt)
        st64(a + 24, hash_bytes(WAL_MAGIC, a, 24))
        var ok = self.wr(WALF, 0, a, WAL_HDR)
        _ = b^
        return ok

    def reset_wal_index(mut self):
        self.f_txn.clear()
        self.f_kind.clear()
        self.f_id.clear()
        self.f_salt.clear()
        self.f_vec.clear()
        self.a_vec.clear()
        self.a_tag.clear()
        self.a_enc.clear()
        self.chain_tail = seed_of(self.salt)
        self.pub_slot = 0

    def index_frame(mut self, txn: Int, kind: UInt64, id: Int, tag: Int64, vec: Int, chain: UInt64) -> Int32:
        if kind == APPEND:
            var ea = len(self.a_enc)
            self.a_enc.resize(ea + self.enc_bytes(), 0)
            var st = encode_row(self.dtype, fptr(vec), self.dim, Int(self.a_enc.unsafe_ptr()) + ea)
            if st != OK:
                self.a_enc.resize(ea, 0)
                return st
        self.f_txn.append(txn)
        self.f_kind.append(kind)
        self.f_id.append(id)
        self.f_salt.append(self.salt)
        self.f_vec.append(len(self.a_tag) if kind == APPEND else -1)
        if kind == APPEND:
            var at = len(self.a_vec)
            self.a_vec.resize(at + self.dim, 0)
            copy_bytes(Int(self.a_vec.unsafe_ptr()) + at * 4, vec, self.dim * 4)
            self.a_tag.append(tag)
        self.chain_tail = chain
        return OK

    def encode_frame(self, a: Int, txn: Int, kind: UInt64, commit: Bool, id: Int, tag: Int64, vec: Int, prev: UInt64) -> UInt64:
        zero_bytes(a, FRAME_HDR)
        st64(a, UInt64(txn))
        st64(a + 8, kind | (UInt64(1 if commit else 0) << 32))
        st64(a + 16, UInt64(id))
        st64(a + 24, UInt64(tag))
        st64(a + 32, self.salt)
        if kind == APPEND:
            copy_bytes(a + FRAME_HDR, vec, self.dim * 4)
        else:
            zero_bytes(a + FRAME_HDR, self.dim * 4)
        var h = seed_of(self.salt) if MUTANT_NO_CHAIN else prev
        var c = hash_bytes(hash_bytes(h, a, 40), a + FRAME_HDR, self.dim * 4)
        st64(a + 40, c)
        return c


def row_ok(vec: Int, dim: Int) -> Bool:
    var p = fptr(vec)
    var ss: Float32 = 0
    for d in range(dim):
        var x = p[unsafe_offset=d]
        if not isfinite(x):
            return False
        ss += x * x
    return isfinite(ss)


def dbp(h: Int) -> Pointer[Db, MutUntrackedOrigin]:
    return Pointer[Db, MutUntrackedOrigin](unsafe_from_address=h)


def usable(h: Int) -> Int32:
    if h == 0:
        return E_NULL
    return E_STATE if dbp(h)[].poisoned else OK


@export("veclite_sim_new")
def veclite_sim_new(out_addr: Int) abi("C") -> Int32:
    if out_addr == 0:
        return E_NULL
    var p = unsafe_alloc[Sim](1)
    p.unsafe_write(Sim(new_file(), new_file(), -1, 0))
    st64(out_addr, UInt64(Int(p)))
    return OK


@export("veclite_sim_free")
def veclite_sim_free(sim: Int) abi("C") -> Int32:
    if sim == 0:
        return E_NULL
    var p = Pointer[Sim, MutUntrackedOrigin](unsafe_from_address=sim)
    _ = p.unsafe_take_pointee()
    p.unsafe_free()
    return OK


@export("veclite_sim_set_budget")
def veclite_sim_set_budget(sim: Int, budget: Int64) abi("C") -> Int32:
    if sim == 0:
        return E_NULL
    Pointer[Sim, MutUntrackedOrigin](unsafe_from_address=sim)[].budget = Int(budget)
    return OK


@export("veclite_sim_mutations")
def veclite_sim_mutations(sim: Int) abi("C") -> Int64:
    return -1 if sim == 0 else Int64(Pointer[Sim, MutUntrackedOrigin](unsafe_from_address=sim)[].mutations)


@export("veclite_sim_crash")
def veclite_sim_crash(sim: Int, seed: UInt64) abi("C") -> Int32:
    if sim == 0:
        return E_NULL
    var s = Pointer[Sim, MutUntrackedOrigin](unsafe_from_address=sim)
    var rng = seed | 1
    s[].main.crash(rng)
    s[].wal.crash(rng)
    s[].budget = -1
    return OK


@export("veclite_sim_crash_range")
def veclite_sim_crash_range(sim: Int, wal_lo: Int64, wal_hi: Int64) abi("C") -> Int32:
    if sim == 0:
        return E_NULL
    if wal_lo < 0 or wal_hi < wal_lo:
        return E_ARG
    var s = Pointer[Sim, MutUntrackedOrigin](unsafe_from_address=sim)
    s[].main.crash_range(0, 0)
    s[].wal.crash_range(Int(wal_lo), Int(wal_hi))
    s[].budget = -1
    return OK


def open_fd(path: Int, suffix: Bool) -> Int:
    var n = Int(external_call["strlen", UInt](path))
    var b = List[UInt8](length=n + 5, fill=0)
    copy_bytes(Int(b.unsafe_ptr()), path, n)
    if suffix:
        copy_bytes(Int(b.unsafe_ptr()) + n, Int("-wal".ptr()), 4)
    var fd = Int(external_call["open", c_int, num_fixed_args=2](Int(b.unsafe_ptr()), c_int(O_RDWR_CREAT), c_int(0o644)))
    _ = b^
    return fd


def load_main(mut db: Db, want_dim: Int, want_extent: Int, want_dtype: Int32) -> Int32:
    var size = db.fsize(MAIN)
    if size == 0:
        if want_dim <= 0 or want_extent <= 0:
            return E_ARG
        db.dim = want_dim
        db.dtype = F32 if want_dtype < 0 else want_dtype
        db.extent_rows = want_extent
        db.hdr_slot = 1
        if not db.trunc(MAIN, DATA_OFF) or not db.write_main_header(0, 0) or not db.sync(MAIN):
            return E_IO
        return OK
    if size < DATA_OFF:
        return E_CORRUPT
    var b = List[UInt8](length=64, fill=0)
    var a = Int(b.unsafe_ptr())
    var best = -1
    for slot in range(2):
        if not db.rd(MAIN, slot * SLOT_BYTES, a, 64):
            return E_IO
        if ld64(a) != MAIN_MAGIC or ld64(a + 48) != hash_bytes(MAIN_MAGIC, a, 48) or (ld64(a + 8) & 0xFFFF) != VERSION:
            continue
        var seq = ld64(a + 40)
        if best >= 0 and seq <= db.hdr_seq:
            continue
        var dim = Int(ld64(a + 8) >> 32)
        var ext = Int(ld64(a + 16))
        var dt = Int32(Int((ld64(a + 8) >> 16) & 0xFFFF))
        if dim <= 0 or dim > Int(MAX_DIM) or ext <= 0 or ext > Int(MAX_EXTENT_ROWS) or check_dtype(dt) != OK:
            return E_CORRUPT
        db.dtype = dt
        best = slot
        db.hdr_seq = seq
        db.dim = dim
        db.extent_rows = ext
        db.hdr_rows = Int(ld64(a + 24))
        db.hdr_txn = Int(ld64(a + 32))
    _ = b^
    if best < 0:
        return E_CORRUPT
    if want_dim > 0 and want_dim != db.dim:
        return E_DIM
    if want_dtype >= 0 and want_dtype != db.dtype:
        return E_DTYPE
    db.hdr_slot = best
    db.set_extent_bytes()
    if db.hdr_rows < 0 or db.hdr_txn < 0 or (db.hdr_rows > 0 and db.live_off(db.hdr_rows - 1) >= size):
        return E_CORRUPT
    return OK


def load_live(mut db: Db) -> Int32:
    db.rows = db.hdr_rows
    db.main_rows = db.hdr_rows
    db.live = List[UInt8](length=db.rows, fill=0)
    var e = 0
    while e * db.extent_rows < db.rows:
        var n = min(db.extent_rows, db.rows - e * db.extent_rows)
        if not db.rd(MAIN, db.live_off(e * db.extent_rows), Int(db.live.unsafe_ptr()) + e * db.extent_rows, n):
            return E_IO
        e += 1
    for i in range(db.rows):
        if db.live[i] > 1:
            return E_CORRUPT
    return OK


def same_as_main(db: Db, id: Int, vec: Int) -> Bool:
    var enc = List[UInt8](length=db.enc_bytes(), fill=0)
    var old = List[UInt8](length=db.enc_bytes(), fill=0)
    var o = Int(old.unsafe_ptr())
    var ok = encode_row(db.dtype, fptr(vec), db.dim, Int(enc.unsafe_ptr())) == OK
    ok = ok and db.rd(MAIN, db.vec_off(id), o, db.enc_bytes()) and same_bytes(o, Int(enc.unsafe_ptr()), db.enc_bytes())
    _ = enc^
    _ = old^
    return ok


def recover_wal(mut db: Db) -> Int32:
    var size = db.fsize(WALF)
    var hb = List[UInt8](length=WAL_HDR, fill=0)
    var ha = Int(hb.unsafe_ptr())
    var have = size >= WAL_HDR and db.rd(WALF, 0, ha, WAL_HDR)
    var good = have and ld64(ha) == WAL_MAGIC and ld64(ha + 24) == hash_bytes(WAL_MAGIC, ha, 24) and ld64(ha + 8) == db.format_word()
    db.salt = ld64(ha + 16) if good else UInt64(db.hdr_seq) + 1
    _ = hb^
    db.reset_wal_index()
    db.ntx = db.hdr_txn
    if not good:
        if not db.write_wal_header() or not db.sync(WALF):
            return E_IO
        db.salt_synced = True
        db.published = db.ntx
        return OK
    db.salt_synced = True
    var fs = db.frame_bytes()
    var nslots = (size - WAL_HDR) // fs if size > WAL_HDR else 0
    var fb = List[UInt8](length=fs, fill=0)
    var fa = Int(fb.unsafe_ptr())
    var prev = seed_of(db.salt)
    var last = -1
    var txns = List[Int]()
    var chains = List[UInt64]()
    for i in range(nslots):
        if not db.rd(WALF, WAL_HDR + i * fs, fa, fs):
            return E_IO
        var h = seed_of(db.salt) if MUTANT_NO_CHAIN else prev
        if ld64(fa + 32) != db.salt or ld64(fa + 40) != hash_bytes(hash_bytes(h, fa, 40), fa + FRAME_HDR, db.dim * 4):
            break
        prev = ld64(fa + 40)
        chains.append(prev)
        txns.append(Int(ld64(fa)))
        if (ld64(fa + 8) >> 32) == 1:
            last = i
    var expect_txn = -1
    for i in range(last + 1):
        if not db.rd(WALF, WAL_HDR + i * fs, fa, fs):
            return E_IO
        var txn = Int(ld64(fa))
        var kind = ld64(fa + 8) & 0xFFFFFFFF
        var commit = ld64(fa + 8) >> 32
        var id = Int(ld64(fa + 16))
        var tag = Int64(ld64(fa + 24))
        if (kind != APPEND and kind != DELETE) or commit > 1 or txn < 1 or (expect_txn >= 0 and txn != expect_txn):
            return E_CORRUPT
        expect_txn = txn + 1 if commit == 1 else txn
        if kind == APPEND:
            if id > db.rows or id < 0 or not row_ok(fa + FRAME_HDR, db.dim):
                return E_CORRUPT
            if id < db.rows:
                if id >= db.main_rows or not same_as_main(db, id, fa + FRAME_HDR):
                    return E_CORRUPT
                db.live[id] = 1
            else:
                db.rows += 1
                db.live.append(1)
        else:
            if id < 0 or id >= db.rows:
                return E_CORRUPT
            db.live[id] = 0
        if db.index_frame(txn, kind, id, tag, fa + FRAME_HDR, chains[i]) != OK:
            return E_CORRUPT
        if commit == 1:
            db.ntx = max(db.ntx, txn)
    _ = fb^
    comptime if not MUTANT_NO_RECOVERY_TRUNCATE:
        var keep = WAL_HDR + (last + 1) * fs
        if size > keep and (not db.trunc(WALF, keep) or not db.sync(WALF)):
            return E_IO
    if last < 0:
        db.chain_tail = seed_of(db.salt)
    db.published = db.ntx
    db.pub_slot = last + 1
    return OK


@export("veclite_open")
def veclite_open(path: Int, dim: Int64, extent_rows: Int64, storage: Int32, sim: Int, out_db: Int) abi("C") -> Int32:
    if out_db == 0 or (path == 0 and sim == 0):
        return E_NULL
    st64(out_db, 0)
    if dim < 0 or dim > MAX_DIM or extent_rows < 0 or extent_rows > MAX_EXTENT_ROWS:
        return E_ARG
    if storage != -1 and check_dtype(storage) != OK:
        return E_DTYPE
    var fds = List[Int](length=2, fill=-1)
    if sim == 0:
        fds[MAIN] = open_fd(path, False)
        fds[WALF] = open_fd(path, True)
        if fds[MAIN] < 0 or fds[WALF] < 0:
            for f in fds:
                if f >= 0:
                    _ = external_call["close", c_int](c_int(f))
            return E_IO
    var db = Db(
        sim, fds^, 0, F32, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, 0,
        List[Int](), List[UInt64](), List[Int](), List[UInt64](), List[Int](), List[Float32](), List[Int64](),
        List[UInt8](),
        0, 0, 0, 0, List[UInt8](), List[UInt64](), List[Int](), List[Int64](), List[Float32](),
        List[Bool](length=MAX_READERS, fill=False), List[Int](length=MAX_READERS, fill=0),
        List[UInt64](length=MAX_READERS, fill=0), List[Int](length=MAX_READERS, fill=0), False, 0, 0, 0, 0, 0,
    )
    var st = load_main(db, Int(dim), Int(extent_rows), storage)
    if st == OK:
        db.set_extent_bytes()
        db.hdr_durable_txn = db.hdr_txn
        st = load_live(db)
    if st == OK:
        st = recover_wal(db)
    if st != OK:
        close_fds(db)
        return st
    var p = unsafe_alloc[Db](1)
    p.unsafe_write(db^)
    st64(out_db, UInt64(Int(p)))
    return OK


def close_fds(db: Db):
    for f in db.fds:
        if f >= 0:
            _ = external_call["close", c_int](c_int(f))


def close_db(h: Int) -> Int32:
    if h == 0:
        return E_NULL
    var p = dbp(h)
    p[].unmap_main()
    close_fds(p[])
    _ = p.unsafe_take_pointee()
    p.unsafe_free()
    return OK


@export("veclite_append")
def veclite_append(h: Int, vec: Int, tag: Int64, out_id: Int) abi("C") -> Int32:
    var st = usable(h)
    if st != OK:
        return st
    if vec == 0 or out_id == 0:
        return E_NULL
    var db = dbp(h)
    var enc = List[UInt8](length=db[].enc_bytes(), fill=0)
    st = encode_row(db[].dtype, fptr(vec), db[].dim, Int(enc.unsafe_ptr()))
    _ = enc^
    if st != OK:
        return st
    var id = db[].rows
    var at = len(db[].p_vec)
    db[].p_vec.resize(at + db[].dim, 0)
    copy_bytes(Int(db[].p_vec.unsafe_ptr()) + at * 4, vec, db[].dim * 4)
    db[].p_kind.append(APPEND)
    db[].p_id.append(id)
    db[].p_tag.append(tag)
    db[].rows += 1
    db[].live.append(1)
    st64(out_id, UInt64(id))
    return OK


@export("veclite_delete")
def veclite_delete(h: Int, id: Int64) abi("C") -> Int32:
    var st = usable(h)
    if st != OK:
        return st
    var db = dbp(h)
    if id < 0 or Int(id) >= db[].rows or db[].live[Int(id)] == 0:
        return E_ID
    db[].p_kind.append(DELETE)
    db[].p_id.append(Int(id))
    db[].p_tag.append(0)
    db[].live[Int(id)] = 0
    return OK


@export("veclite_pending")
def veclite_pending(h: Int, out_n: Int) abi("C") -> Int32:
    var st = usable(h)
    if st != OK:
        return st
    if out_n == 0:
        return E_NULL
    st64(out_n, UInt64(len(dbp(h)[].p_kind)))
    return OK


@export("veclite_rollback")
def veclite_rollback(h: Int, mark: Int64) abi("C") -> Int32:
    var st = usable(h)
    if st != OK:
        return st
    var db = dbp(h)
    var n = len(db[].p_kind)
    if mark < 0 or Int(mark) > n:
        return E_ARG
    while len(db[].p_kind) > Int(mark):
        var kind = db[].p_kind.pop()
        var id = db[].p_id.pop()
        _ = db[].p_tag.pop()
        comptime if not MUTANT_ROLLBACK_KEEP_ROWS:
            if kind == APPEND:
                db[].rows -= 1
                _ = db[].live.pop()
                db[].p_vec.resize(len(db[].p_vec) - db[].dim, 0)
            else:
                db[].live[id] = 1
        else:
            if kind == APPEND:
                db[].p_vec.resize(len(db[].p_vec) - db[].dim, 0)
        comptime if MUTANT_ROLLBACK_KEEP_PENDING:
            db[].p_kind.append(kind)
            db[].p_id.append(id)
            db[].p_tag.append(0)
            break
    return OK


@export("veclite_commit")
def veclite_commit(h: Int, out_txn: Int) abi("C") -> Int32:
    var st = usable(h)
    if st != OK:
        return st
    var db = dbp(h)
    var n = len(db[].p_kind)
    if n == 0:
        if out_txn != 0:
            st64(out_txn, UInt64(db[].published))
        return OK
    if not db[].salt_synced and not MUTANT_UNSYNCED_RESTART_HDR:
        return E_STATE
    var fs = db[].frame_bytes()
    var buf = List[UInt8](length=n * fs, fill=0)
    var base = Int(buf.unsafe_ptr())
    var txn = db[].ntx + 1
    var chains = List[UInt64]()
    var prev = db[].chain_tail
    var vi = 0
    for j in range(n):
        var vec = Int(db[].p_vec.unsafe_ptr()) + vi * db[].dim * 4
        prev = db[].encode_frame(base + j * fs, txn, db[].p_kind[j], j == n - 1, db[].p_id[j], db[].p_tag[j], vec, prev)
        chains.append(prev)
        if db[].p_kind[j] == APPEND:
            vi += 1
    var slot = len(db[].f_txn)
    if not db[].wr(WALF, WAL_HDR + slot * fs, base, n * fs):
        return E_IO
    for j in range(n):
        _ = db[].index_frame(txn, db[].p_kind[j], db[].p_id[j], db[].p_tag[j], base + j * fs + FRAME_HDR, chains[j])
    _ = buf^
    db[].ntx = txn
    db[].p_kind.clear()
    db[].p_id.clear()
    db[].p_tag.clear()
    db[].p_vec.clear()
    comptime if not MUTANT_ACK_BEFORE_SYNC:
        if not db[].sync(WALF):
            return E_IO
    db[].salt_synced = True
    db[].published = txn
    db[].pub_slot = len(db[].f_txn)
    if out_txn != 0:
        st64(out_txn, UInt64(txn))
    return OK


def ckpt_target(db: Db) -> Int:
    var t = db.published
    comptime if not MUTANT_CKPT_IGNORES_READERS:
        for r in range(MAX_READERS):
            if db.r_active[r]:
                t = min(t, db.r_mark[r])
    return t


@export("veclite_checkpoint")
def veclite_checkpoint(h: Int, out_txn: Int) abi("C") -> Int32:
    var st = usable(h)
    if st != OK:
        return st
    var db = dbp(h)
    if len(db[].p_kind) > 0:
        return E_STATE
    var target = ckpt_target(db[])
    if target > db[].hdr_txn:
        var one = List[UInt8](length=1, fill=1)
        var zero = List[UInt8](length=1, fill=0)
        var tagb = List[Int64](length=1, fill=0)
        var rows = db[].main_rows
        for i in range(len(db[].f_txn)):
            var t = db[].f_txn[i]
            if t <= db[].hdr_txn or t > target:
                continue
            var id = db[].f_id[i]
            if db[].f_kind[i] == APPEND:
                var a = db[].f_vec[i]
                tagb[0] = db[].a_tag[a]
                if not db[].wr(MAIN, db[].vec_off(id), Int(db[].a_enc.unsafe_ptr()) + a * db[].enc_bytes(), db[].enc_bytes()):
                    return E_IO
                if not db[].wr(MAIN, db[].tag_off(id), Int(tagb.unsafe_ptr()), 8):
                    return E_IO
                if not db[].wr(MAIN, db[].live_off(id), Int(one.unsafe_ptr()), 1):
                    return E_IO
                rows = max(rows, id + 1)
            elif not db[].wr(MAIN, db[].live_off(id), Int(zero.unsafe_ptr()), 1):
                return E_IO
        db[].main_rows = rows
        _ = one^
        _ = zero^
        _ = tagb^
        comptime if not MUTANT_SKIP_DATA_SYNC:
            if not db[].sync(MAIN):
                return E_IO
        if not db[].write_main_header(rows, target):
            return E_IO
        comptime if not MUTANT_RESTART_EARLY:
            if not db[].sync(MAIN):
                return E_IO
            db[].hdr_durable_txn = target
        else:
            db[].hdr_durable_txn = target
    var restart = len(db[].f_txn) > 0 and db[].published == db[].ntx and db[].hdr_durable_txn == db[].ntx
    for r in range(MAX_READERS):
        restart = restart and (not db[].r_active[r] or db[].r_mark[r] == db[].ntx)
    if restart:
        db[].salt += 1
        db[].reset_wal_index()
        db[].salt_synced = False
        if not db[].write_wal_header():
            return E_IO
        comptime if not MUTANT_UNSYNCED_RESTART_HDR:
            if not db[].sync(WALF):
                return E_IO
            db[].salt_synced = True
    comptime if MUTANT_RESTART_EARLY:
        if not db[].sync(MAIN):
            return E_IO
    if out_txn != 0:
        st64(out_txn, UInt64(db[].hdr_txn))
    return OK


@export("veclite_reader_begin")
def veclite_reader_begin(h: Int, out_r: Int) abi("C") -> Int32:
    var st = usable(h)
    if st != OK:
        return st
    if out_r == 0:
        return E_NULL
    var db = dbp(h)
    for r in range(MAX_READERS):
        if not db[].r_active[r]:
            db[].r_active[r] = True
            db[].r_mark[r] = db[].published
            db[].r_salt[r] = db[].salt
            db[].r_max_slot[r] = db[].pub_slot
            st64(out_r, UInt64(r))
            return OK
    return E_READER


def reader_ok(h: Int, r: Int64) -> Int32:
    var st = usable(h)
    if st != OK:
        return st
    return OK if r >= 0 and r < MAX_READERS and dbp(h)[].r_active[Int(r)] else E_READER


@export("veclite_reader_end")
def veclite_reader_end(h: Int, r: Int64) abi("C") -> Int32:
    var st = reader_ok(h, r)
    if st == OK:
        dbp(h)[].r_active[Int(r)] = False
    return st


@export("veclite_reader_mark")
def veclite_reader_mark(h: Int, r: Int64, out_mark: Int) abi("C") -> Int32:
    var st = reader_ok(h, r)
    if st == OK and out_mark == 0:
        st = E_NULL
    if st == OK:
        st64(out_mark, UInt64(dbp(h)[].r_mark[Int(r)]))
    return st


@fieldwise_init
struct View(Movable):
    var live: List[UInt8]
    var main_rows: Int
    var tail: Int
    var base: Int
    var size: Int


def make_view(db: Db, r: Int) -> View:
    var n = db.main_rows
    var live = List[UInt8](length=n, fill=0)
    var e = 0
    while e * db.extent_rows < n:
        var cnt = min(db.extent_rows, n - e * db.extent_rows)
        _ = db.rd(MAIN, db.live_off(e * db.extent_rows), Int(live.unsafe_ptr()) + e * db.extent_rows, cnt)
        e += 1
    var tail = 0
    for i in range(min(db.r_max_slot[r], len(db.f_txn))):
        if db.f_salt[i] != db.r_salt[r]:
            continue
        var id = db.f_id[i]
        if db.f_kind[i] == APPEND:
            if id >= n:
                live.append(1)
                tail += 1
            else:
                live[id] = 1
        elif id < len(live):
            live[id] = 0
    return View(live^, n, tail, 0, 0)


@export("veclite_reader_view")
def veclite_reader_view(h: Int, r: Int64, cap: Int64, out_live: Int, out_tags: Int, out_vecs: Int, out_n: Int) abi("C") -> Int32:
    var st = reader_ok(h, r)
    if st != OK:
        return st
    if out_n == 0 or (cap > 0 and (out_live == 0 or out_tags == 0 or out_vecs == 0)):
        return E_NULL
    var db = dbp(h)
    var v = make_view(db[], Int(r))
    var n = len(v.live)
    st64(out_n, UInt64(n))
    if Int(cap) < n:
        return E_ARG
    var dim = db[].dim
    var first_tail = tail_start(db[], Int(r))
    var h16 = List[Float16](length=dim, fill=0)
    var hp = Int(h16.unsafe_ptr())
    for id in range(n):
        st8(out_live + id, v.live[id])
        var dst = out_vecs + id * dim * 4
        if id < v.main_rows:
            if db[].dtype == F32:
                _ = db[].rd(MAIN, db[].vec_off(id), dst, dim * 4)
            else:
                _ = db[].rd(MAIN, db[].vec_off(id), hp, dim * 2)
            _ = db[].rd(MAIN, db[].tag_off(id), out_tags + id * 8, 8)
        else:
            var a = first_tail + id - v.main_rows
            if db[].dtype == F16:
                copy_bytes(hp, Int(db[].a_enc.unsafe_ptr()) + a * dim * 2, dim * 2)
            else:
                copy_bytes(dst, Int(db[].a_vec.unsafe_ptr()) + a * dim * 4, dim * 4)
            st64(out_tags + id * 8, UInt64(db[].a_tag[a]))
        if db[].dtype == F16:
            for d in range(dim):
                fmut(dst)[unsafe_offset=d] = h16[d].cast[DType.float32]()
    _ = h16^
    return OK


def tail_start(db: Db, r: Int) -> Int:
    for i in range(min(db.r_max_slot[r], len(db.f_txn))):
        if db.f_salt[i] == db.r_salt[r] and db.f_kind[i] == APPEND and db.f_id[i] >= db.main_rows:
            return db.f_vec[i]
    return len(db.a_tag)


def check_search(h: Int, r: Int64, queries: Int, nq: Int64, k: Int64, out_ids: Int, out_scores: Int, out_counts: Int) -> Int32:
    var st = reader_ok(h, r)
    if st != OK:
        return st
    if queries == 0 or out_ids == 0 or out_scores == 0 or out_counts == 0:
        return E_NULL
    if nq <= 0 or nq > MAX_BATCH:
        return E_BATCH
    if k <= 0 or k > MAX_K:
        return E_K
    for q in range(Int(nq)):
        imut(out_counts)[unsafe_offset=q] = 0
    return OK


def search_rows(
    mut db: Db, r: Int, v: View, from_row: Int, queries: Int, nq: Int64, filter_on: Int, filter_tags: Int,
    k: Int64, metric: Int32, nchunks: Int64, out_ids: Int, out_scores: Int, out_counts: Int,
) -> Int32:
    var ni = Int(nq)
    var ki = Int(k)
    var size = db.fsize(MAIN)
    var m = db.map_main(size)
    if size > 0 and m == -1:
        return E_IO
    var e_ids = List[Int64](length=ni * ki, fill=0)
    var e_sc = List[Float32](length=ni * ki, fill=0)
    var e_cnt = List[Int64](length=ni, fill=0)
    var nseg = (v.main_rows + db.extent_rows - 1) // db.extent_rows + 1
    var st = OK
    for seg in range(nseg):
        var lo = seg * db.extent_rows
        var n: Int
        var data: Int
        var tags: Int
        if seg < nseg - 1:
            var hi = min(lo + db.extent_rows, v.main_rows)
            lo = max(lo, from_row)
            n = hi - lo
            data = m + db.vec_off(lo)
            tags = m + db.tag_off(lo)
        else:
            lo = v.main_rows
            n = v.tail
            var a = tail_start(db, r)
            data = Int(db.a_enc.unsafe_ptr()) + a * db.enc_bytes()
            tags = Int(db.a_tag.unsafe_ptr()) + a * 8
        if n <= 0:
            continue
        st = veclite_search_ex(
            data, db.dtype, Int64(n), Int64(db.dim), tags, Int(v.live.unsafe_ptr()) + lo, queries, nq,
            filter_on, filter_tags, k, metric, nchunks, Int(e_ids.unsafe_ptr()), Int(e_sc.unsafe_ptr()), Int(e_cnt.unsafe_ptr()),
        )
        if st != OK:
            break
        for q in range(ni):
            var length = Int(imut(out_counts)[unsafe_offset=q])
            for j in range(Int(e_cnt[q])):
                length = bstep(fmut(out_scores).unsafe_offset(q * ki), imut(out_ids).unsafe_offset(q * ki), length, ki, e_sc[q * ki + j], e_ids[q * ki + j] + Int64(lo))
            imut(out_counts)[unsafe_offset=q] = Int64(length)
    _ = e_ids^
    _ = e_sc^
    _ = e_cnt^
    return st


def copy_main_rows(db: Db, rows: Int, dst_vec: Int, dst_tags: Int) -> Bool:
    var eb = db.enc_bytes()
    var e = 0
    while e * db.extent_rows < rows:
        var lo = e * db.extent_rows
        var n = min(db.extent_rows, rows - lo)
        if not db.rd(MAIN, db.vec_off(lo), dst_vec + lo * eb, n * eb) or not db.rd(MAIN, db.tag_off(lo), dst_tags + lo * 8, n * 8):
            return False
        e += 1
    return True


def search_cpu(h: Int, r: Int64, queries: Int, nq: Int64, filter_on: Int, filter_tags: Int,
               k: Int64, metric: Int32, nchunks: Int64, out_ids: Int, out_scores: Int, out_counts: Int) -> Int32:
    var st = check_search(h, r, queries, nq, k, out_ids, out_scores, out_counts)
    if st != OK:
        return st
    var v = make_view(dbp(h)[], Int(r))
    st = search_rows(dbp(h)[], Int(r), v, 0, queries, nq, filter_on, filter_tags, k, metric, nchunks, out_ids, out_scores, out_counts)
    _ = v^
    if st != OK:
        for q in range(Int(nq)):
            imut(out_counts)[unsafe_offset=q] = 0
    return st


@export("veclite_reader_search")
def veclite_reader_search(
    h: Int, r: Int64, queries: Int, nq: Int64, filter_on: Int, filter_tags: Int,
    k: Int64, metric: Int32, nchunks: Int64, out_ids: Int, out_scores: Int, out_counts: Int,
) abi("C") -> Int32:
    return search_cpu(h, r, queries, nq, filter_on, filter_tags, k, metric, nchunks, out_ids, out_scores, out_counts)


def live_count(db: Db) -> Int:
    var c = 0
    for i in range(db.rows):
        c += Int(db.live[i])
    return c


@export("veclite_info")
def veclite_info(h: Int, dst: Int) abi("C") -> Int32:
    if h == 0 or dst == 0:
        return E_NULL
    var db = dbp(h)
    var vals: List[Int] = [db[].rows, db[].ntx, db[].published, db[].hdr_txn, len(db[].f_txn), Int(db[].salt), db[].main_rows, 1 if db[].poisoned else 0, Int(db[].dtype), db[].dim, db[].gpu_rows, live_count(db[])]
    for i in range(len(vals)):
        st64(dst + i * 8, UInt64(vals[i]))
    return OK
