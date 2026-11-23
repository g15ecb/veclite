module gpu_route {
  const MAXROWS: int
  const K: int
  const MAXSCORE: int
  const STALE_LIVE: bool
  const SKIP_GAP: bool
  const NO_REFRESH_LIMIT: bool

  type Cand = { id: int, score: int }
  type View = { rows: int, live: int -> bool }

  var score: int -> int
  var rows: int
  var live: int -> bool
  var mainRows: int
  var gpuRows: int
  var gpuLive: int -> bool
  var reader: View
  var result: List[Cand]
  var searched: bool

  pure def lessTotal(a: Cand, b: Cand): bool = a.score < b.score or (a.score == b.score and a.id < b.id)

  pure def minInt(a: int, b: int): int = if (a < b) a else b

  pure def insertBounded(lst: List[Cand], c: Cand, cap: int): List[Cand] = {
    val merged = lst.select(e => lessTotal(e, c)).append(c).concat(lst.select(e => lessTotal(c, e)))
    merged.slice(0, minInt(merged.length(), cap))
  }

  def IDS: Set[int] = 0.to(MAXROWS - 1)

  def topIn(lo: int, hi: int, alive: int -> bool): List[Cand] =
    range(lo, hi).foldl(List(), (acc, i) => if (alive.get(i)) insertBounded(acc, { id: i, score: score.get(i) }, K) else acc)

  def mergeAll(a: List[Cand], b: List[Cand]): List[Cand] = b.foldl(a, (acc, c) => insertBounded(acc, c, K))

  def eligible(v: View): Set[Cand] = 0.to(v.rows - 1).filter(i => v.live.get(i)).map(i => { id: i, score: score.get(i) })

  def reference(v: View): Set[Cand] = eligible(v).filter(c => eligible(v).filter(e => lessTotal(e, c)).size() < K)

  def asSet(l: List[Cand]): Set[Cand] = l.foldl(Set(), (s, c) => s.union(Set(c)))

  action init = {
    nondet sc = IDS.setOfMaps(0.to(MAXSCORE)).oneOf()
    all {
      score' = sc,
      rows' = 0,
      live' = IDS.mapBy(i => false),
      mainRows' = 0,
      gpuRows' = 0,
      gpuLive' = IDS.mapBy(i => false),
      reader' = { rows: 0, live: IDS.mapBy(i => false) },
      result' = List(),
      searched' = false,
    }
  }

  action unchangedGpu = all { gpuRows' = gpuRows, gpuLive' = gpuLive }

  action appendRow = all {
    rows < MAXROWS,
    rows' = rows + 1,
    live' = live.set(rows, true),
    mainRows' = mainRows,
    unchangedGpu,
    score' = score, reader' = reader, result' = result, searched' = false,
  }

  action deleteRow = {
    nondet i = IDS.oneOf()
    all {
      i < rows, live.get(i),
      live' = live.set(i, false),
      rows' = rows, mainRows' = mainRows,
      unchangedGpu,
      score' = score, reader' = reader, result' = result, searched' = false,
    }
  }

  action checkpoint = all {
    minInt(rows, reader.rows) > mainRows,
    mainRows' = minInt(rows, reader.rows),
    rows' = rows, live' = live,
    unchangedGpu,
    score' = score, reader' = reader, result' = result, searched' = false,
  }

  action refreshGpu = all {
    gpuRows' = mainRows,
    gpuLive' = live,
    rows' = rows, live' = live, mainRows' = mainRows,
    score' = score, reader' = reader, result' = result, searched' = false,
  }

  action readerBegin = all {
    reader' = { rows: rows, live: live },
    rows' = rows, live' = live, mainRows' = mainRows,
    unchangedGpu,
    score' = score, result' = result, searched' = false,
  }

  action search = {
    val v = reader
    val split = if (NO_REFRESH_LIMIT) gpuRows else minInt(gpuRows, v.rows)
    val gpuPart = topIn(0, split, if (STALE_LIVE) gpuLive else v.live)
    val cpuFrom = if (SKIP_GAP) mainRows else split
    val cpuPart = topIn(cpuFrom, v.rows, v.live)
    all {
      gpuRows <= mainRows,
      result' = mergeAll(gpuPart, cpuPart),
      searched' = true,
      rows' = rows, live' = live, mainRows' = mainRows,
      unchangedGpu,
      score' = score, reader' = reader,
    }
  }

  action step = any { appendRow, deleteRow, checkpoint, refreshGpu, readerBegin, search }

  val exact = not(searched) or asSet(result) == reference(reader)

  val sorted = not(searched) or result.indices().forall(i => i == 0 or lessTotal(result[i - 1], result[i]))

  val inv = exact and sorted

  val witnessStaleGpu = not(searched and gpuRows < reader.rows and gpuRows > 0 and result.length() == K)

  val witnessDeleteAfterRefresh = not(searched and IDS.exists(i => i < gpuRows and gpuLive.get(i) and not(reader.live.get(i))))

  val witnessReaderOlderThanGpu = not(searched and reader.rows < gpuRows)
}

module fixed {
  import gpu_route(MAXROWS = 6, K = 2, MAXSCORE = 3, STALE_LIVE = false, SKIP_GAP = false, NO_REFRESH_LIMIT = false).*
}

module mutantStaleLive {
  import gpu_route(MAXROWS = 6, K = 2, MAXSCORE = 3, STALE_LIVE = true, SKIP_GAP = false, NO_REFRESH_LIMIT = false).*
}

module mutantSkipGap {
  import gpu_route(MAXROWS = 6, K = 2, MAXSCORE = 3, STALE_LIVE = false, SKIP_GAP = true, NO_REFRESH_LIMIT = false).*
}

module mutantNoRefreshLimit {
  import gpu_route(MAXROWS = 6, K = 2, MAXSCORE = 3, STALE_LIVE = false, SKIP_GAP = false, NO_REFRESH_LIMIT = true).*
}
