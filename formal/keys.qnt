module keys {
  const NKEYS: int
  const MAXROWS: int
  const ROLLBACK_KEEPS_INDEX: bool
  const DELETE_KEEPS_KEY: bool
  const RECOVERY_IGNORES_DELETES: bool
  const DUP_CHECK_COMMITTED_ONLY: bool

  type Op = { del: bool, row: int, key: int }
  type Rows = { n: int, key: int -> int, live: int -> bool }

  var committed: List[Op]
  var pending: List[Op]
  var rows: Rows
  var index: int -> int
  var ghost: { dupAccepted: bool, crashes: int, rollbacks: int, reuses: int, rejects: int }

  pure val NONE = -1

  def KEYS: Set[int] = 0.to(NKEYS - 1)

  def RIDS: Set[int] = 0.to(MAXROWS - 1)

  pure def apply(r: Rows, o: Op): Rows =
    if (o.del) { ...r, live: r.live.set(o.row, false) }
    else { n: r.n + 1, key: r.key.set(o.row, o.key), live: r.live.set(o.row, true) }

  def empty: Rows = { n: 0, key: RIDS.mapBy(i => NONE), live: RIDS.mapBy(i => false) }

  def replay(ops: List[Op]): Rows = ops.foldl(empty, (r, o) => apply(r, o))

  def liveKeys(r: Rows): Set[int] = RIDS.filter(i => i < r.n and r.live.get(i)).map(i => r.key.get(i))

  def derived(r: Rows): int -> int =
    KEYS.mapBy(k => RIDS.fold(NONE, (a, i) => if (i < r.n and r.live.get(i) and r.key.get(i) == k) i else a))

  def rebuild(ops: List[Op]): int -> int =
    if (RECOVERY_IGNORES_DELETES)
      ops.foldl(KEYS.mapBy(k => NONE), (ix, o) => if (o.del) ix else ix.set(o.key, o.row))
    else derived(replay(ops))

  def taken(k: int): bool =
    if (DUP_CHECK_COMMITTED_ONLY) derived(replay(committed)).get(k) != NONE else index.get(k) != NONE

  action init = all {
    committed' = List(),
    pending' = List(),
    rows' = empty,
    index' = KEYS.mapBy(k => NONE),
    ghost' = { dupAccepted: false, crashes: 0, rollbacks: 0, reuses: 0, rejects: 0 },
  }

  action add = {
    nondet k = KEYS.oneOf()
    if (taken(k)) all {
      committed' = committed, pending' = pending, rows' = rows, index' = index,
      ghost' = { ...ghost, rejects: ghost.rejects + 1 },
    } else all {
      rows.n < MAXROWS,
      val o = { del: false, row: rows.n, key: k }
      all {
        pending' = pending.append(o),
        rows' = apply(rows, o),
        index' = index.set(k, rows.n),
        committed' = committed,
        ghost' = {
          ...ghost,
          dupAccepted: ghost.dupAccepted or index.get(k) != NONE,
          reuses: ghost.reuses + (if (RIDS.exists(i => i < rows.n and rows.key.get(i) == k)) 1 else 0),
        },
      }
    }
  }

  action deleteKey = {
    nondet k = KEYS.oneOf()
    all {
      index.get(k) != NONE,
      val r = index.get(k)
      all {
        pending' = pending.append({ del: true, row: r, key: k }),
        rows' = apply(rows, { del: true, row: r, key: k }),
        index' = if (DELETE_KEEPS_KEY) index else index.set(k, NONE),
        committed' = committed,
        ghost' = ghost,
      }
    }
  }

  action commit = all {
    pending.length() > 0,
    committed' = committed.concat(pending),
    pending' = List(),
    rows' = rows, index' = index, ghost' = ghost,
  }

  action rollbackTo(m: int): bool = all {
    m >= 0, m < pending.length(),
    val kept = committed.concat(pending.slice(0, m))
    all {
      pending' = pending.slice(0, m),
      rows' = replay(kept),
      index' = if (ROLLBACK_KEEPS_INDEX) index else derived(replay(kept)),
      committed' = committed,
      ghost' = { ...ghost, rollbacks: ghost.rollbacks + 1 },
    }
  }

  action rollback = {
    nondet m = 0.to(MAXROWS).oneOf()
    rollbackTo(m)
  }

  action crash = all {
    committed' = committed,
    pending' = List(),
    rows' = replay(committed),
    index' = rebuild(committed),
    ghost' = { ...ghost, crashes: ghost.crashes + 1 },
  }

  action step = any { add, deleteKey, commit, rollback, crash }

  val rowsMatchLog = rows == replay(committed.concat(pending))

  val indexExact = index == derived(rows)

  val uniqueLive = RIDS.forall(i => RIDS.forall(j =>
    not(i < j and j < rows.n and rows.live.get(i) and rows.live.get(j) and rows.key.get(i) == rows.key.get(j))))

  val noDupAccepted = not(ghost.dupAccepted)

  val inv = and { rowsMatchLog, indexExact, uniqueLive, noDupAccepted }

  val witnessKeyReused = ghost.reuses == 0

  val witnessRejected = ghost.rejects == 0

  val witnessRollbackCrash = not(ghost.rollbacks > 0 and ghost.crashes > 0 and committed.length() > 2)
}

module fixed {
  import keys(NKEYS = 3, MAXROWS = 6, ROLLBACK_KEEPS_INDEX = false, DELETE_KEEPS_KEY = false, RECOVERY_IGNORES_DELETES = false, DUP_CHECK_COMMITTED_ONLY = false).*
}

module mutantRollbackKeepsIndex {
  import keys(NKEYS = 3, MAXROWS = 6, ROLLBACK_KEEPS_INDEX = true, DELETE_KEEPS_KEY = false, RECOVERY_IGNORES_DELETES = false, DUP_CHECK_COMMITTED_ONLY = false).*
}

module mutantDeleteKeepsKey {
  import keys(NKEYS = 3, MAXROWS = 6, ROLLBACK_KEEPS_INDEX = false, DELETE_KEEPS_KEY = true, RECOVERY_IGNORES_DELETES = false, DUP_CHECK_COMMITTED_ONLY = false).*
}

module mutantRecoveryIgnoresDeletes {
  import keys(NKEYS = 3, MAXROWS = 6, ROLLBACK_KEEPS_INDEX = false, DELETE_KEEPS_KEY = false, RECOVERY_IGNORES_DELETES = true, DUP_CHECK_COMMITTED_ONLY = false).*
}

module mutantDupCheckCommittedOnly {
  import keys(NKEYS = 3, MAXROWS = 6, ROLLBACK_KEEPS_INDEX = false, DELETE_KEEPS_KEY = false, RECOVERY_IGNORES_DELETES = false, DUP_CHECK_COMMITTED_ONLY = true).*
}
