module scenarios {
  const UNSYNCED_HDR: bool
  const TRUNCATE: bool

  import wal(NIDS = 3, MAXTXN = 4, SLOTS = 5, NREADERS = 2, ACK_BEFORE_SYNC = false, CKPT_IGNORES_READERS = false, NO_CHAIN = false, RESTART_EARLY = false, SKIP_DATA_SYNC = false, READER_NO_SLOT_BOUND = false, UNSYNCED_RESTART_HDR = UNSYNCED_HDR, TRUNCATE_ON_RECOVERY = TRUNCATE, ROLLBACK_KEEP_FRAMES = false, ROLLBACK_KEEP_PENDING = false).* from "./wal"

  pure def ins(x: int): Op = { del: false, id: x, v: 1 }

  pure def rm(x: int): Op = { del: true, id: x, v: 0 }

  def picks(m: Set[(int, int)]): int -> int = SLOTSET.mapBy(i => m.fold(0, (a, e) => if (e._1 == i) e._2 else a))

  def allRows: int -> bool = IDS.mapBy(i => true)

  action checkpointAll = ckptApply.then(ckptSyncData).then(ckptWriteHdr).then(ckptSyncHdr)

  run restartThenCrash =
    init
      .then(writeFrameWith(ins(0), true))
      .then(writeFrameWith(rm(0), true))
      .then(syncWal)
      .then(checkpointAll)
      .then(walRestart)
      .then(if (UNSYNCED_HDR) writeFrameWith(ins(1), false) else syncWalHeader.then(writeFrameWith(ins(1), false)))
      .then(writeFrameWith(ins(2), true))
      .then(crashWith(picks(Set((1, 1))), picks(Set((1, 1))), if (UNSYNCED_HDR) 0 else 1, allRows, 2))
      .expect(if (UNSYNCED_HDR) not(recoveryAtomic) else recoveryAtomic)

  run staleFrameCompletion =
    init
      .then(writeFrameWith(ins(0), true))
      .then(syncWal)
      .then(writeFrameWith(ins(1), true))
      .then(crashWith(picks(Set((1, 1))), picks(Set((1, 2))), 0, allRows, 0))
      .then(writeFrameWith(ins(1), false))
      .then(writeFrameWith(ins(2), true))
      .then(crashWith(picks(Set((1, 0), (2, 0))), picks(Set((1, 1), (2, 0))), 0, allRows, 0))
      .expect(if (TRUNCATE) recoveryAtomic else not(recoveryAtomic))
}

module fixed {
  import scenarios(UNSYNCED_HDR = false, TRUNCATE = true).*

  run restartThenCrashTest = restartThenCrash

  run staleFrameCompletionTest = staleFrameCompletion
}

module unsyncedRestartHdr {
  import scenarios(UNSYNCED_HDR = true, TRUNCATE = true).*

  run restartThenCrashTest = restartThenCrash

  run staleFrameCompletionTest = staleFrameCompletion
}

module noRecoveryTruncate {
  import scenarios(UNSYNCED_HDR = false, TRUNCATE = false).*

  run restartThenCrashTest = restartThenCrash

  run staleFrameCompletionTest = staleFrameCompletion
}
