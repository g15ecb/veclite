import os
import tempfile

import numpy as np

import veclite

rng = np.random.default_rng(0)
docs = rng.standard_normal((10_000, 1536), dtype=np.float32)
tenant = rng.integers(0, 50, len(docs))

path = os.path.join(tempfile.mkdtemp(), "docs.db")
with veclite.open(path, dim=1536, storage="f16") as db:
    ids = db.add(docs, tags=tenant)
    db.delete(ids[0])

with veclite.open(path) as db:
    query = docs[42] + 0.01 * rng.standard_normal(1536, dtype=np.float32)
    hits, scores = db.search(query, k=5, tag=tenant[42])
    batch = db.search(docs[:64], k=10)

assert hits[0] == 42 and len(batch) == 64
print("top-5 ids:", hits.tolist(), "cosine distances:", np.round(scores, 4).tolist())
