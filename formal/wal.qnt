module wal {
  const NIDS: int
  const MAXTXN: int
  const SLOTS: int
  const NREADERS: int
  const ACK_BEFORE_SYNC: bool
  const CKPT_IGNORES_READERS: bool
  const NO_CHAIN: bool
  const RESTART_EARLY: bool
  const SKIP_DATA_SYNC: bool
  const READER_NO_SLOT_BOUND: bool
  const UNSYNCED_RESTART_HDR: bool
  const TRUNCATE_ON_RECOVERY: bool
  const ROLLBACK_KEEP_FRAMES: bool
  const ROLLBACK_KEEP_PENDING: bool

  type Op = { del: bool, id: int, v: int }
  type Row = { present: bool, v: int, live: bool }
  type Frame = { wid: int, prev: int, salt: int, txn: int, op: Op, commit: bool }
  type Slot = { hdr: Frame, payload: Op }
  type Reader = { active: bool, mark: int, salt: int, maxSlot: int }
  type Wal = { cache: int -> Slot, disk: int -> Slot, saltC: int, saltD: int, len: int, nextWid: int }
  type Main = { cache: int -> Row, disk: int -> Row, hdrC: int, hdrD: int, phase: str, target: int }
  type Log = { committed: List[List[Op]], pending: List[Op], published: int, pubSlot: int, acked: int }
  type Ghost = { rollbacks: int, crashes: int, restarts: int, lostAcked: bool, recoveryBad: bool, tornRecovered: bool, replayedCheckpointed: bool }

  var wal: Wal
  var main: Main
  var log: Log
  var readers: int -> Reader
  var ghost: Ghost

  pure val NOOP: Op = { del: true, id: -1, v: 0 }
  pure val GARBAGE: Op = { del: false, id: -2, v: -1 }
  pure val NOHDR: Frame = { wid: -1, prev: -1, salt: -1, txn: -1, op: NOOP, commit: false }
  pure val EMPTY: Slot = { hdr: NOHDR, payload: GARBAGE }
  pure val NOROW: Row = { present: false, v: 0, live: false }
  pure val IDLE: Reader = { active: false, mark: 0, salt: 0, maxSlot: 0 }

  pure def seed(salt: int): int = -1 - salt

  pure def minInt(a: int, b: int): int = if (a < b) a else b

  pure def applyOp(s: int -> Row, o: Op): int -> Row =
    if (not(o.del)) s.set(o.id, { present: true, v: o.v, live: true })
    else if (s.get(o.id).present) s.set(o.id, { ...s.get(o.id), live: false })
    else s

  pure def applyAll(s: int -> Row, ops: List[Op]): int -> Row = ops.foldl(s, (acc, o) => applyOp(acc, o))

  def IDS: Set[int] = 0.to(NIDS - 1)

  def SLOTSET: Set[int] = 0.to(SLOTS - 1)

  def emptyRows: int -> Row = IDS.mapBy(i => NOROW)

  def stateAt(m: int): int -> Row = log.committed.slice(0, m).foldl(emptyRows, (s, t) => applyAll(s, t))

  def ntx: int = log.committed.length()

  def current: int -> Row = applyAll(stateAt(ntx), log.pending)

  def nextId: int = IDS.filter(i => current.get(i).present).size()

  def candidateOps: Set[Op] =
    (if (nextId < NIDS) Set(1, 2).map(x => { del: false, id: nextId, v: x }) else Set())
      .union(IDS.filter(i => current.get(i).present and current.get(i).live).map(i => { del: true, id: i, v: 0 }))

  def activeReaders: Set[int] = readers.keys().filter(r => readers.get(r).active)

  def ckptTarget: int =
    if (CKPT_IGNORES_READERS) log.published
    else activeReaders.fold(log.published, (a, r) => minInt(a, readers.get(r).mark))

  def frameVisible(r: Reader, i: int): bool = {
    val f = wal.cache.get(i).hdr
    f.wid >= 0 and f.salt == r.salt and (if (READER_NO_SLOT_BOUND) f.txn <= r.mark else i < r.maxSlot)
  }

  def view(r: Reader): int -> Row =
    range(0, SLOTS).foldl(main.cache, (s, i) => if (frameVisible(r, i)) applyOp(s, wal.cache.get(i).hdr.op) else s)

  def lastCommitSlot(frames: int -> Slot, len: int): int =
    range(0, len).foldl(0, (a, i) => if (frames.get(i).hdr.commit) i + 1 else a)

  pure def half(old: Frame, new: Frame, pick: int): Frame = if (pick == 0) old else if (pick == 1) new else NOHDR

  pure def halfOp(old: Op, new: Op, pick: int): Op = if (pick == 0) old else if (pick == 1) new else GARBAGE

  action init = all {
    wal' = { cache: SLOTSET.mapBy(i => EMPTY), disk: SLOTSET.mapBy(i => EMPTY), saltC: 0, saltD: 0, len: 0, nextWid: 0 },
    main' = { cache: emptyRows, disk: emptyRows, hdrC: 0, hdrD: 0, phase: "idle", target: 0 },
    log' = { committed: List(), pending: List(), published: 0, pubSlot: 0, acked: 0 },
    readers' = 0.to(NREADERS - 1).mapBy(r => IDLE),
    ghost' = { rollbacks: 0, crashes: 0, restarts: 0, lostAcked: false, recoveryBad: false, tornRecovered: false, replayedCheckpointed: false },
  }

  action writeFrameWith(o: Op, commit: bool): bool = {
    val txn = ntx + 1
    val prev = if (wal.len == 0) seed(wal.saltC) else wal.cache.get(wal.len - 1).hdr.wid
    val f = { wid: wal.nextWid, prev: prev, salt: wal.saltC, txn: txn, op: o, commit: commit }
    all {
      main.phase == "idle",
      UNSYNCED_RESTART_HDR or wal.saltD == wal.saltC,
      candidateOps.contains(o),
      commit or log.pending.length() == 0,
      wal.len < SLOTS,
      txn <= MAXTXN,
      wal' = { ...wal, cache: wal.cache.set(wal.len, { hdr: f, payload: o }), len: wal.len + 1, nextWid: wal.nextWid + 1 },
      log' =
        if (not(commit)) { ...log, pending: log.pending.append(o) }
        else if (ACK_BEFORE_SYNC) { committed: log.committed.append(log.pending.append(o)), pending: List(), published: txn, pubSlot: wal.len + 1, acked: txn }
        else { ...log, committed: log.committed.append(log.pending.append(o)), pending: List() },
      main' = main,
      readers' = readers,
      ghost' = ghost,
    }
  }

  action writeFrame = {
    nondet o = candidateOps.oneOf()
    nondet commit = Set(true, false).oneOf()
    writeFrameWith(o, commit)
  }

  action syncWal = all {
    wal.disk != wal.cache or wal.saltD != wal.saltC or log.published < ntx,
    wal' = { ...wal, disk: wal.cache, saltD: wal.saltC },
    log' = { ...log, published: ntx, acked: ntx, pubSlot: lastCommitSlot(wal.cache, wal.len) },
    main' = main,
    readers' = readers,
    ghost' = ghost,
  }

  action ckptApply = all {
    main.phase == "idle",
    log.pending == List(),
    ckptTarget > main.hdrC,
    main' = {
      ...main,
      cache: range(0, wal.len).foldl(main.cache, (s, i) => {
        val t = wal.cache.get(i).hdr.txn
        if (t > main.hdrC and t <= ckptTarget) applyOp(s, wal.cache.get(i).hdr.op) else s
      }),
      phase: "applied",
      target: ckptTarget,
    },
    wal' = wal,
    log' = log,
    readers' = readers,
    ghost' = ghost,
  }

  action ckptSyncData = all {
    main.phase == "applied",
    main' = { ...main, disk: main.cache, phase: "synced" },
    wal' = wal,
    log' = log,
    readers' = readers,
    ghost' = ghost,
  }

  action ckptWriteHdr = all {
    main.phase == "synced" or (SKIP_DATA_SYNC and main.phase == "applied"),
    main' = { ...main, hdrC: main.target, phase: "hdr" },
    wal' = wal,
    log' = log,
    readers' = readers,
    ghost' = ghost,
  }

  action ckptSyncHdr = all {
    main.phase == "hdr",
    main' = { ...main, hdrD: main.hdrC, phase: "idle" },
    wal' = wal,
    log' = log,
    readers' = readers,
    ghost' = ghost,
  }

  action walRestart = all {
    log.pending == List(),
    wal.len > 0,
    log.published == ntx,
    if (RESTART_EARLY) main.hdrC == ntx and (main.phase == "idle" or main.phase == "hdr")
    else main.phase == "idle" and main.hdrD == ntx,
    activeReaders.forall(r => readers.get(r).mark == ntx),
    wal' = { ...wal, saltC: wal.saltC + 1, len: 0 },
    log' = { ...log, pubSlot: 0 },
    main' = main,
    readers' = readers,
    ghost' = { ...ghost, restarts: ghost.restarts + 1 },
  }

  action syncWalHeader = all {
    wal.saltD != wal.saltC,
    wal' = { ...wal, saltD: wal.saltC },
    main' = main,
    log' = log,
    readers' = readers,
    ghost' = ghost,
  }

  action rollbackTo(m: int): bool = all {
    m >= 0,
    m < log.pending.length(),
    wal' = if (ROLLBACK_KEEP_FRAMES) wal else { ...wal, len: wal.len - (log.pending.length() - m) },
    log' = if (ROLLBACK_KEEP_PENDING) log else { ...log, pending: log.pending.slice(0, m) },
    main' = main,
    readers' = readers,
    ghost' = { ...ghost, rollbacks: ghost.rollbacks + 1 },
  }

  action rollback = {
    nondet m = 0.to(MAXTXN).oneOf()
    rollbackTo(m)
  }

  action readerBegin = {
    nondet r = readers.keys().filter(r => not(readers.get(r).active)).oneOf()
    all {
      readers' = readers.set(r, { active: true, mark: log.published, salt: wal.saltC, maxSlot: log.pubSlot }),
      wal' = wal,
      main' = main,
      log' = log,
      ghost' = ghost,
    }
  }

  action readerEnd = {
    nondet r = activeReaders.oneOf()
    all {
      readers' = readers.set(r, IDLE),
      wal' = wal,
      main' = main,
      log' = log,
      ghost' = ghost,
    }
  }

  action crashWith(hdrHalf: int -> int, payHalf: int -> int, saltPick: int, rowPick: int -> bool, hdrPick: int): bool = {
    val wd = SLOTSET.mapBy(i => {
      val c = wal.cache.get(i)
      val d = wal.disk.get(i)
      {
        hdr: if (c.hdr == d.hdr) d.hdr else half(d.hdr, c.hdr, hdrHalf.get(i)),
        payload: if (c.payload == d.payload) d.payload else halfOp(d.payload, c.payload, payHalf.get(i)),
      }
    })
    val md = IDS.mapBy(i => if (rowPick.get(i)) main.cache.get(i) else main.disk.get(i))
    val scan = range(0, SLOTS).foldl({ ok: true, last: -1 }, (acc, i) => {
      val f = wd.get(i).hdr
      val chained = NO_CHAIN or f.prev == (if (i == 0) seed(saltPick) else wd.get(i - 1).hdr.wid)
      val good = acc.ok and f.wid >= 0 and f.salt == saltPick and chained and wd.get(i).payload == f.op
      { ok: good, last: if (good and f.commit) i else acc.last }
    })
    val walTxn = if (scan.last >= 0) wd.get(scan.last).hdr.txn else -1
    val recCount = if (walTxn > hdrPick) walTxn else hdrPick
    val recState = range(0, scan.last + 1).foldl(md, (s, i) => applyOp(s, wd.get(i).hdr.op))
    val keptWal = if (TRUNCATE_ON_RECOVERY) SLOTSET.mapBy(i => if (i <= scan.last) wd.get(i) else EMPTY) else wd
    val recOk = recCount <= ntx and (if (recCount <= ntx) recState == stateAt(recCount) else false)
    val kept = minInt(recCount, ntx)
    all {
      Set(wal.saltD, wal.saltC).contains(saltPick),
      Set(main.hdrD, main.hdrC).contains(hdrPick),
      wal' = { cache: keptWal, disk: keptWal, saltC: saltPick, saltD: saltPick, len: scan.last + 1, nextWid: wal.nextWid },
      main' = { cache: md, disk: md, hdrC: hdrPick, hdrD: hdrPick, phase: "idle", target: 0 },
      log' = { committed: log.committed.slice(0, kept), pending: List(), published: kept, pubSlot: scan.last + 1, acked: kept },
      readers' = readers.keys().mapBy(r => IDLE),
      ghost' = {
        ...ghost,
        crashes: ghost.crashes + 1,
        lostAcked: ghost.lostAcked or recCount < log.acked,
        recoveryBad: ghost.recoveryBad or not(recOk),
        tornRecovered: ghost.tornRecovered or SLOTSET.exists(i => wd.get(i).hdr.wid >= 0 and wd.get(i).payload != wd.get(i).hdr.op),
        replayedCheckpointed: ghost.replayedCheckpointed or (hdrPick > 0 and walTxn > hdrPick and wd.get(0).hdr.txn <= hdrPick),
      },
    }
  }

  action crash = {
    nondet hdrHalf = SLOTSET.setOfMaps(0.to(2)).oneOf()
    nondet payHalf = SLOTSET.setOfMaps(0.to(2)).oneOf()
    nondet saltPick = Set(wal.saltD, wal.saltC).oneOf()
    nondet rowPick = IDS.setOfMaps(Set(true, false)).oneOf()
    nondet hdrPick = Set(main.hdrD, main.hdrC).oneOf()
    crashWith(hdrHalf, payHalf, saltPick, rowPick, hdrPick)
  }

  action step = any { writeFrame, rollback, syncWal, syncWalHeader, ckptApply, ckptSyncData, ckptWriteHdr, ckptSyncHdr, walRestart, readerBegin, readerEnd, crash }

  val readerSnapshot = activeReaders.forall(r => view(readers.get(r)) == stateAt(readers.get(r).mark))

  val ackedDurable = not(ghost.lostAcked)

  val recoveryAtomic = not(ghost.recoveryBad)

  val publishedOrder = log.acked <= log.published and log.published <= ntx and main.hdrC <= log.published

  val pendingFramesMatch = wal.len == lastCommitSlot(wal.cache, wal.len) + log.pending.length() or ghost.crashes > 0

  val inv = and { readerSnapshot, ackedDurable, recoveryAtomic, publishedOrder }

  val witnessRollbackThenCommit = not(ghost.rollbacks > 0 and ntx > 0 and ghost.crashes > 0)

  val witnessTornRecovered = not(ghost.tornRecovered and ghost.crashes > 0 and ntx > 0)

  val witnessRestartThenCrash = not(ghost.restarts > 0 and ghost.crashes > 0 and ntx > 0)

  val witnessReaderBehindCheckpoint = not(activeReaders.exists(r => readers.get(r).mark < log.published and main.hdrD > 0))

  val witnessReplayOverCheckpoint = not(ghost.replayedCheckpointed)
}

module base {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantAckBeforeSync {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = true, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantCkptIgnoresReaders {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = true, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantNoChain {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = true, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantRestartEarly {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = true, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantSkipDataSync {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = true, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantReaderNoSlotBound {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = true, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantUnsyncedRestartHdr {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = true, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantNoRecoveryTruncate {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = false, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).*
}

module mutantRollbackKeepFrames {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = true, ROLLBACK_KEEP_PENDING = false).*
}

module mutantRollbackKeepPending {
  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = false, TRUNCATE_ON_RECOVERY = true, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = true).*
}
