import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import veclite


def reference(data, live, tags, q, k, tag, metric):
    x = data.astype(np.float64)
    qd = q.astype(np.float64)
    if metric == "l2":
        s = ((x - qd) ** 2).sum(axis=1)
    elif metric == "dot":
        s = -(x @ qd)
    else:
        s = 1 - (x @ qd) / (np.linalg.norm(x, axis=1) * np.linalg.norm(qd))
    idx = np.nonzero(live & ((tags == tag) if tag is not None else True))[0]
    order = np.lexsort((idx, s[idx]))
    return idx[order][:k]


@pytest.fixture
def path(tmp_path):
    return str(tmp_path / "t.db")


def ints(rng, n, dim):
    x = rng.integers(-3, 4, (n, dim)).astype(np.float32)
    x[:, 0] += 5
    return x


@pytest.mark.parametrize("metric", ["l2", "dot", "cosine"])
@pytest.mark.parametrize("storage", ["f32", "f16"])
def test_search_matches_reference(path, metric, storage):
    rng = np.random.default_rng(1)
    data = ints(rng, 3000, 24)
    tags = rng.integers(0, 4, 3000)
    with veclite.open(path, dim=24, storage=storage) as db:
        ids = db.add(data[:2000], tags[:2000])
        db.checkpoint()
        ids2 = db.add(data[2000:], tags[2000:])
        assert list(ids) == list(range(2000)) and list(ids2) == list(range(2000, 3000))
        live = np.ones(3000, dtype=bool)
        for d in rng.choice(3000, 300, replace=False):
            db.delete(int(d), commit=False)
            live[d] = False
        db.commit()
        q = ints(rng, 6, 24)
        filters = [None, 0, 1, None, 3, 2]
        hits = db.search(q, k=7, tag=filters, metric=metric)
        for i, (got, _) in enumerate(hits):
            want = reference(data, live, tags, q[i], 7, filters[i], metric)
            if metric == "cosine":
                assert len(np.intersect1d(got, want)) >= 6
            else:
                assert np.array_equal(got, want)
            assert live[got].all() and (filters[i] is None or (tags[got] == filters[i]).all())


def test_batch_equals_single(path):
    rng = np.random.default_rng(2)
    data = rng.standard_normal((5000, 64)).astype(np.float32)
    with veclite.open(path, dim=64) as db:
        db.add(data)
        q = rng.standard_normal((9, 64)).astype(np.float32)
        batch = db.search(q, k=10)
        for i in range(9):
            ids, sc = db.search(q[i], k=10)
            assert np.array_equal(ids, batch[i][0]) and np.array_equal(sc, batch[i][1])
            assert np.all(np.diff(sc) >= 0)


def test_persistence_across_reopen(path):
    rng = np.random.default_rng(3)
    data = rng.standard_normal((1500, 32)).astype(np.float32)
    q = rng.standard_normal(32).astype(np.float32)
    with veclite.open(path, dim=32, storage="f16") as db:
        db.add(data[:1000])
        db.checkpoint()
        db.add(data[1000:])
        db.delete(5)
        before = db.search(q, k=20, metric="l2")
    with veclite.open(path) as db:
        assert db.dim == 32 and db.storage == "f16"
        assert db.info()["rows"] == 1500
        after = db.search(q, k=20, metric="l2")
        assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])
        assert 5 not in after[0]


def test_uncommitted_rows_are_lost_on_reopen(path):
    with veclite.open(path, dim=4) as db:
        db.add(np.ones((3, 4)))
        db.add(np.ones((2, 4)), commit=False)
    with veclite.open(path) as db:
        assert db.info()["rows"] == 3


def test_errors(path):
    with veclite.open(path, dim=4) as db:
        with pytest.raises(veclite.VecliteError) as e:
            db.add([[1, np.nan, 0, 0]])
        assert e.value.code == -12
        with pytest.raises(ValueError):
            db.add(np.ones((2, 5)))
        with pytest.raises(ValueError):
            db.add(np.ones((2, 4)), tags=[1])
        with pytest.raises(veclite.VecliteError) as e:
            db.delete(99)
        assert e.value.code == -23
        db.add(np.ones((1, 4)))
        with pytest.raises(veclite.VecliteError) as e:
            db.search(np.zeros(4), metric="cosine")
        assert e.value.code == -9
        with pytest.raises(ValueError):
            db.search(np.ones(4), k=0)
        with pytest.raises(ValueError):
            db.search(np.ones(4), metric="hamming")
    with pytest.raises(veclite.VecliteError) as e:
        veclite.open(path, dim=8)
    assert e.value.code == -3
    with pytest.raises(veclite.VecliteError) as e:
        veclite.open(path, storage="f16")
    assert e.value.code == -15
    with pytest.raises(ValueError):
        veclite.open(path, storage="int8")
    with pytest.raises(veclite.VecliteError):
        veclite.open(os.path.join(os.path.dirname(path), "fresh.db"))


def test_f16_range_rejected(path):
    with veclite.open(path, dim=2, storage="f16") as db:
        with pytest.raises(veclite.VecliteError) as e:
            db.add([[70000.0, 1.0]])
        assert e.value.code == -16


def test_corrupt_file_rejected(path):
    with veclite.open(path, dim=4) as db:
        db.add(np.ones((2, 4)))
    with open(path, "r+b") as f:
        for off in (24, 4096 + 24):
            f.seek(off)
            f.write(b"\x7f")
    with pytest.raises(veclite.VecliteError) as e:
        veclite.open(path)
    assert e.value.code == -21


def test_closed_database(path):
    db = veclite.open(path, dim=4)
    db.close()
    db.close()
    with pytest.raises(veclite.VecliteError):
        db.search(np.ones(4))
