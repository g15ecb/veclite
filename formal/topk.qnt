module topk {
  const N: int
  const K: int
  const NCHUNKS: int
  const MAXSCORE: int
  const NTAGS: int
  const FILTERED: bool
  const FILTER_TAG: int
  const TIE_BREAK: bool
  const PRE_FILTER: bool
  const LOCAL_K_DELTA: int

  type Row = { id: int, score: int, tag: int, live: bool }
  type Cand = { id: int, score: int }

  var rows: int -> Row
  var pending: Set[int]
  var locals: int -> List[Cand]
  var order: List[int]
  var result: List[Cand]
  var phase: str

  pure def lessTotal(a: Cand, b: Cand): bool =
    a.score < b.score or (a.score == b.score and a.id < b.id)

  pure def lessImpl(a: Cand, b: Cand): bool =
    if (TIE_BREAK) lessTotal(a, b) else a.score < b.score

  pure def minInt(a: int, b: int): int = if (a < b) a else b

  pure def insertBounded(lst: List[Cand], c: Cand, cap: int): List[Cand] =
    if (cap <= 0) List()
    else {
      val merged = lst.select(e => not(lessImpl(c, e))).append(c).concat(lst.select(e => lessImpl(c, e)))
      merged.slice(0, minInt(merged.length(), cap))
    }

  def eligible(r: Row): bool = r.live and (not(FILTERED) or r.tag == FILTER_TAG)

  def scanEligible(r: Row): bool = if (PRE_FILTER) eligible(r) else r.live

  def chunkLo(c: int): int = c * N / NCHUNKS

  def chunkHi(c: int): int = (c + 1) * N / NCHUNKS

  def scanChunk(c: int): List[Cand] =
    range(chunkLo(c), chunkHi(c)).foldl(List(), (acc, i) =>
      if (scanEligible(rows.get(i))) insertBounded(acc, { id: i, score: rows.get(i).score }, K + LOCAL_K_DELTA)
      else acc)

  def mergeAll(ord: List[int]): List[Cand] =
    ord.foldl(List(), (acc, c) =>
      locals.get(c).foldl(acc, (a, cand) =>
        if (eligible(rows.get(cand.id))) insertBounded(a, cand, K) else a))

  def eligibleCands: Set[Cand] =
    0.to(N - 1).filter(i => eligible(rows.get(i))).map(i => { id: i, score: rows.get(i).score })

  def referenceTopK: Set[Cand] =
    eligibleCands.filter(c => eligibleCands.filter(e => lessTotal(e, c)).size() < K)

  def resultSet: Set[Cand] = result.foldl(Set(), (s, c) => s.union(Set(c)))

  action init = {
    nondet sc = 0.to(N - 1).setOfMaps(0.to(MAXSCORE)).oneOf()
    nondet tg = 0.to(N - 1).setOfMaps(0.to(NTAGS - 1)).oneOf()
    nondet lv = 0.to(N - 1).setOfMaps(Set(true, false)).oneOf()
    all {
      rows' = 0.to(N - 1).mapBy(i => { id: i, score: sc.get(i), tag: tg.get(i), live: lv.get(i) }),
      pending' = 0.to(NCHUNKS - 1),
      locals' = 0.to(NCHUNKS - 1).mapBy(c => List()),
      order' = List(),
      result' = List(),
      phase' = "scan",
    }
  }

  action processChunk = {
    nondet c = pending.oneOf()
    all {
      phase == "scan",
      locals' = locals.put(c, scanChunk(c)),
      order' = order.append(c),
      pending' = pending.exclude(Set(c)),
      rows' = rows,
      result' = result,
      phase' = phase,
    }
  }

  action merge = all {
    phase == "scan",
    pending == Set(),
    result' = mergeAll(order),
    phase' = "done",
    rows' = rows,
    pending' = pending,
    locals' = locals,
    order' = order,
  }

  action finished = all {
    phase == "done",
    rows' = rows,
    pending' = pending,
    locals' = locals,
    order' = order,
    result' = result,
    phase' = phase,
  }

  action step = any { processChunk, merge, finished }

  val resultIsExactTopK = phase != "done" or resultSet == referenceTopK

  val resultSortedUnique = phase != "done" or result.indices().forall(i => i == 0 or lessTotal(result[i - 1], result[i]))

  val resultSizeIsMinKMatches = phase != "done" or result.length() == minInt(K, eligibleCands.size())

  val resultOnlyLiveAndMatching = resultSet.forall(c => eligible(rows.get(c.id)))

  val localsBounded = locals.keys().forall(c => locals.get(c).length() <= K)

  val inv = and {
    resultIsExactTopK,
    resultSortedUnique,
    resultSizeIsMinKMatches,
    resultOnlyLiveAndMatching,
    localsBounded,
  }

  val witnessDone = phase != "done"

  val witnessTruncated = not(phase == "done" and eligibleCands.size() > K)

  val witnessTiesInResult = not(phase == "done" and result.indices().exists(i => i > 0 and result[i - 1].score == result[i].score))
}

module filtered {
  import topk(N = 7, K = 3, NCHUNKS = 3, MAXSCORE = 2, NTAGS = 2, FILTERED = true, FILTER_TAG = 1, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0).*
}

module unfiltered {
  import topk(N = 7, K = 3, NCHUNKS = 3, MAXSCORE = 2, NTAGS = 2, FILTERED = false, FILTER_TAG = 0, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0).*
}

module moreChunksThanRows {
  import topk(N = 3, K = 2, NCHUNKS = 5, MAXSCORE = 1, NTAGS = 2, FILTERED = true, FILTER_TAG = 0, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0).*
}

module kExceedsRows {
  import topk(N = 4, K = 6, NCHUNKS = 2, MAXSCORE = 1, NTAGS = 2, FILTERED = true, FILTER_TAG = 1, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0).*
}

module mutantNoTieBreak {
  import topk(N = 7, K = 3, NCHUNKS = 3, MAXSCORE = 1, NTAGS = 2, FILTERED = true, FILTER_TAG = 1, TIE_BREAK = false, PRE_FILTER = true, LOCAL_K_DELTA = 0).*
}

module mutantPostFilter {
  import topk(N = 7, K = 2, NCHUNKS = 2, MAXSCORE = 2, NTAGS = 2, FILTERED = true, FILTER_TAG = 1, TIE_BREAK = true, PRE_FILTER = false, LOCAL_K_DELTA = 0).*
}

module mutantLocalKMinusOne {
  import topk(N = 7, K = 3, NCHUNKS = 2, MAXSCORE = 2, NTAGS = 2, FILTERED = true, FILTER_TAG = 1, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = -1).*
}
