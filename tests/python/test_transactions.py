import gc
import os
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import veclite


@pytest.fixture
def db(tmp_path):
    with veclite.open(str(tmp_path / "t.db"), dim=4) as d:
        yield d


def test_failed_add_leaves_nothing(db):
    x = np.ones((10, 4), np.float32)
    x[5, 1] = np.nan
    with pytest.raises(veclite.VecliteError) as e:
        db.add(x)
    assert e.value.code == -12 and e.value.row == 5
    assert db.pending == 0 and len(db) == 0
    db.add(np.ones((1, 4)))
    assert len(db) == 1


def test_failed_add_keeps_earlier_pending(db):
    first = db.add(np.ones((3, 4)), commit=False)
    x = np.ones((4, 4), np.float32)
    x[2, 0] = np.inf
    with pytest.raises(veclite.VecliteError):
        db.add(x, commit=False)
    assert db.pending == 3
    db.commit()
    assert len(db) == 3 and list(first) == [0, 1, 2]


def test_failed_delete_mid_batch_rolls_back_only_that_call(db):
    db.add(np.ones((6, 4)))
    db.delete(0, commit=False)
    with pytest.raises(veclite.VecliteError) as e:
        db.delete([1, 2, 99, 3])
    assert e.value.code == -23 and e.value.row == 2
    assert db.pending == 1
    db.commit()
    assert len(db) == 5
    ids, _ = db.search(np.ones(4), k=10, metric="l2")
    assert sorted(ids.tolist()) == [1, 2, 3, 4, 5]


def test_rollback_discards_everything_and_reuses_ids(db):
    db.add(np.ones((2, 4)))
    db.add(np.ones((3, 4)), commit=False)
    db.delete(0, commit=False)
    db.rollback()
    assert db.pending == 0 and len(db) == 2
    assert list(db.add(np.ones((1, 4)))) == [2]
    ids, _ = db.search(np.ones(4), k=10, metric="l2")
    assert sorted(ids.tolist()) == [0, 1, 2]


def test_f16_range_reports_row(tmp_path):
    with veclite.open(str(tmp_path / "h.db"), dim=2, storage="f16") as d:
        x = np.ones((3, 2), np.float32)
        x[1, 1] = 1e6
        with pytest.raises(veclite.VecliteError) as e:
            d.add(x)
        assert e.value.code == -16 and e.value.row == 1 and len(d) == 0


@pytest.mark.parametrize("k", [0, -1, 4097, 2.5, True, "3"])
def test_bad_k(db, k):
    db.add(np.ones((1, 4)))
    with pytest.raises(ValueError, match="k must be"):
        db.search(np.ones(4), k=k)


def test_tag_count_mismatch(db):
    db.add(np.ones((2, 4)))
    with pytest.raises(ValueError, match="one tag per query"):
        db.search(np.ones((3, 4)), tag=[1, 2])
    with pytest.raises(ValueError, match="one tag per vector"):
        db.add(np.ones((2, 4)), tags=[1])


def test_threads_share_one_database(tmp_path):
    rng = np.random.default_rng(1)
    with veclite.open(str(tmp_path / "c.db"), dim=16) as d:
        d.add(rng.standard_normal((500, 16)))
        d.checkpoint()
        q = rng.standard_normal((8, 16)).astype(np.float32)
        errors = []

        def writer():
            try:
                for _ in range(40):
                    d.add(rng.standard_normal((5, 16)))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(40):
                    hits = d.search(q, k=5)
                    assert all(len(h[0]) == 5 for h in hits)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=f) for f in (writer, writer, reader, reader, reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(d) == 500 + 2 * 40 * 5


def test_unclosed_database_is_finalized(tmp_path):
    path = str(tmp_path / "f.db")
    d = veclite.open(path, dim=4)
    d.add(np.ones((2, 4)))
    del d
    gc.collect()
    with veclite.open(path) as again:
        assert len(again) == 2
