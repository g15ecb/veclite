module topk {
  const N: int
  const K: int
  const NQ: int
  const NCHUNKS: int
  const MAXSCORE: int
  const NTAGS: int
  const TIE_BREAK: bool
  const PRE_FILTER: bool
  const LOCAL_K_DELTA: int
  const QUERY_MIX: bool

  type Row = { tag: int, live: bool }
  type Query = { filterOn: bool, tag: int }
  type Cand = { id: int, score: int }

  var rows: int -> Row
  var queries: int -> Query
  var score: (int, int) -> int
  var pending: Set[int]
  var locals: (int, int) -> List[Cand]
  var order: List[int]
  var result: int -> List[Cand]
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

  def QS: Set[int] = 0.to(NQ - 1)

  def matches(f: Query, i: int): bool =
    rows.get(i).live and (not(f.filterOn) or rows.get(i).tag == f.tag)

  def eligible(q: int, i: int): bool = matches(queries.get(q), i)

  def scanEligible(q: int, i: int): bool =
    if (not(PRE_FILTER)) rows.get(i).live
    else matches(if (QUERY_MIX) queries.get(0) else queries.get(q), i)

  def chunkLo(c: int): int = c * N / NCHUNKS

  def chunkHi(c: int): int = (c + 1) * N / NCHUNKS

  def scanChunk(c: int): int -> List[Cand] =
    range(chunkLo(c), chunkHi(c)).foldl(QS.mapBy(q => List()), (acc, i) =>
      QS.mapBy(q =>
        if (scanEligible(q, i)) insertBounded(acc.get(q), { id: i, score: score.get((q, i)) }, K + LOCAL_K_DELTA)
        else acc.get(q)))

  def mergeQuery(q: int): List[Cand] =
    order.foldl(List(), (acc, c) =>
      locals.get((c, q)).foldl(acc, (a, cand) =>
        if (eligible(q, cand.id)) insertBounded(a, cand, K) else a))

  def eligibleCands(q: int): Set[Cand] =
    0.to(N - 1).filter(i => eligible(q, i)).map(i => { id: i, score: score.get((q, i)) })

  def referenceTopK(q: int): Set[Cand] =
    eligibleCands(q).filter(c => eligibleCands(q).filter(e => lessTotal(e, c)).size() < K)

  def resultSet(q: int): Set[Cand] = result.get(q).foldl(Set(), (s, c) => s.union(Set(c)))

  action init = {
    nondet sc = tuples(QS, 0.to(N - 1)).setOfMaps(0.to(MAXSCORE)).oneOf()
    nondet tg = 0.to(N - 1).setOfMaps(0.to(NTAGS - 1)).oneOf()
    nondet lv = 0.to(N - 1).setOfMaps(Set(true, false)).oneOf()
    nondet qf = QS.setOfMaps(Set(true, false)).oneOf()
    nondet qt = QS.setOfMaps(0.to(NTAGS - 1)).oneOf()
    all {
      rows' = 0.to(N - 1).mapBy(i => { tag: tg.get(i), live: lv.get(i) }),
      queries' = QS.mapBy(q => { filterOn: qf.get(q), tag: qt.get(q) }),
      score' = sc,
      pending' = 0.to(NCHUNKS - 1),
      locals' = tuples(0.to(NCHUNKS - 1), QS).mapBy(key => List()),
      order' = List(),
      result' = QS.mapBy(q => List()),
      phase' = "scan",
    }
  }

  action unchangedInputs = all {
    rows' = rows,
    queries' = queries,
    score' = score,
  }

  action processChunk = {
    nondet c = pending.oneOf()
    val bufs = scanChunk(c)
    all {
      phase == "scan",
      locals' = locals.keys().mapBy(key => if (key._1 == c) bufs.get(key._2) else locals.get(key)),
      order' = order.append(c),
      pending' = pending.exclude(Set(c)),
      unchangedInputs,
      result' = result,
      phase' = phase,
    }
  }

  action merge = all {
    phase == "scan",
    pending == Set(),
    result' = QS.mapBy(q => mergeQuery(q)),
    phase' = "done",
    unchangedInputs,
    pending' = pending,
    locals' = locals,
    order' = order,
  }

  action finished = all {
    phase == "done",
    unchangedInputs,
    pending' = pending,
    locals' = locals,
    order' = order,
    result' = result,
    phase' = phase,
  }

  action step = any { processChunk, merge, finished }

  val resultIsExactTopK = phase != "done" or QS.forall(q => resultSet(q) == referenceTopK(q))

  val resultSortedUnique = phase != "done" or QS.forall(q =>
    result.get(q).indices().forall(i => i == 0 or lessTotal(result.get(q)[i - 1], result.get(q)[i])))

  val resultSizeIsMinKMatches = phase != "done" or QS.forall(q =>
    result.get(q).length() == minInt(K, eligibleCands(q).size()))

  val resultOnlyLiveAndMatching = QS.forall(q => resultSet(q).forall(c => eligible(q, c.id)))

  val localsBounded = locals.keys().forall(key => locals.get(key).length() <= K)

  val inv = and {
    resultIsExactTopK,
    resultSortedUnique,
    resultSizeIsMinKMatches,
    resultOnlyLiveAndMatching,
    localsBounded,
  }

  val witnessDone = phase != "done"

  val witnessTruncated = not(phase == "done" and QS.exists(q => eligibleCands(q).size() > K))

  val witnessTiesInResult = not(phase == "done" and QS.exists(q =>
    result.get(q).indices().exists(i => i > 0 and result.get(q)[i - 1].score == result.get(q)[i].score)))

  val witnessFiltersDiffer = not(phase == "done" and QS.exists(a => QS.exists(b =>
    queries.get(a).filterOn and queries.get(b).filterOn and queries.get(a).tag != queries.get(b).tag
      and result.get(a).length() > 0 and result.get(b).length() > 0)))
}

module batch {
  import topk(N = 6, K = 2, NQ = 3, NCHUNKS = 3, MAXSCORE = 2, NTAGS = 2, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0, QUERY_MIX = false).*
}

module single {
  import topk(N = 7, K = 3, NQ = 1, NCHUNKS = 3, MAXSCORE = 2, NTAGS = 2, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0, QUERY_MIX = false).*
}

module moreChunksThanRows {
  import topk(N = 3, K = 2, NQ = 2, NCHUNKS = 5, MAXSCORE = 1, NTAGS = 2, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0, QUERY_MIX = false).*
}

module kExceedsRows {
  import topk(N = 4, K = 6, NQ = 2, NCHUNKS = 2, MAXSCORE = 1, NTAGS = 2, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0, QUERY_MIX = false).*
}

module mutantNoTieBreak {
  import topk(N = 6, K = 2, NQ = 2, NCHUNKS = 3, MAXSCORE = 1, NTAGS = 2, TIE_BREAK = false, PRE_FILTER = true, LOCAL_K_DELTA = 0, QUERY_MIX = false).*
}

module mutantPostFilter {
  import topk(N = 6, K = 2, NQ = 2, NCHUNKS = 2, MAXSCORE = 2, NTAGS = 2, TIE_BREAK = true, PRE_FILTER = false, LOCAL_K_DELTA = 0, QUERY_MIX = false).*
}

module mutantLocalKMinusOne {
  import topk(N = 6, K = 2, NQ = 2, NCHUNKS = 2, MAXSCORE = 2, NTAGS = 2, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = -1, QUERY_MIX = false).*
}

module mutantQueryMix {
  import topk(N = 6, K = 2, NQ = 2, NCHUNKS = 2, MAXSCORE = 2, NTAGS = 2, TIE_BREAK = true, PRE_FILTER = true, LOCAL_K_DELTA = 0, QUERY_MIX = true).*
}
