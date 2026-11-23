import os
import tempfile

import numpy as np

import veclite

rng = np.random.default_rng(0)
docs = {1000 + i: {"tenant": int(rng.integers(0, 50)), "text": f"document {i}"} for i in range(10_000)}
keys = np.fromiter(docs, dtype=np.int64)
vectors = rng.standard_normal((len(docs), 1536), dtype=np.float32)
tenants = [docs[int(k)]["tenant"] for k in keys]

path = os.path.join(tempfile.mkdtemp(), "docs.db")
with veclite.open(path, dim=1536, storage="f16") as db:
    db.add(vectors, keys=keys, tags=tenants)
    db.delete(1000)

with veclite.open(path) as db:
    target = 1042
    query = vectors[target - 1000] + 0.01 * rng.standard_normal(1536, dtype=np.float32)
    hits, distances = db.search(query, k=5, tag=docs[target]["tenant"])
    batch = db.search(vectors[:64], k=10)

assert int(hits[0]) == target and len(batch) == 64 and all(1000 not in ids for ids, _ in batch)
print("best match:", int(hits[0]), docs[int(hits[0])]["text"], "| top-5 keys:", hits.tolist())
