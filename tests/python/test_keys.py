import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import veclite

DIM = 64


def embed(texts):
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
            out[i, h % DIM] += 1.0
        out[i, 0] += 0.01
    return out


@pytest.fixture
def db(tmp_path):
    with veclite.open(str(tmp_path / "k.db"), dim=DIM) as d:
        yield d


def test_dict_example(db):
    docs = {
        1001: {"tenant": 7, "text": "how to reset a password"},
        1002: {"tenant": 7, "text": "billing cycles explained"},
        2001: {"tenant": 9, "text": "configuring sso with okta password"},
    }
    keys = list(docs)
    db.add(embed([docs[k]["text"] for k in keys]), keys=keys, tags=[docs[k]["tenant"] for k in keys])
    hits, _ = db.search(embed(["forgot my password reset"])[0], k=2, tag=7)
    assert list(hits) == [1001, 1002]
    assert docs[int(hits[0])]["text"] == "how to reset a password"

    docs[1002]["text"] = "billing cycles and invoices explained"
    db.delete(1002, commit=False)
    db.add(embed([docs[1002]["text"]]), keys=[1002], tags=[7])
    assert 1002 in db and len(db) == 3
    hits, _ = db.search(embed(["invoices"])[0], k=1, tag=7)
    assert list(hits) == [1002]

    with pytest.raises(veclite.VecliteError) as e:
        db.add(embed(["duplicate"]), keys=[1001])
    assert e.value.code == -26 and e.value.row == 0 and len(db) == 3


def test_default_keys_are_row_ids(db):
    assert list(db.add(np.ones((3, DIM)))) == [0, 1, 2]
    hits, _ = db.search(np.ones(DIM), k=3, metric="l2")
    assert sorted(hits.tolist()) == [0, 1, 2]


def test_duplicate_in_batch_rejected_before_writing(db):
    with pytest.raises(veclite.VecliteError) as e:
        db.add(np.ones((4, DIM)), keys=[5, 6, 5, 7])
    assert e.value.code == -26 and e.value.row == 2
    assert len(db) == 0 and db.pending == 0


def test_duplicate_against_live_rolls_back_batch(db):
    db.add(np.ones((1, DIM)), keys=[42])
    with pytest.raises(veclite.VecliteError) as e:
        db.add(np.ones((3, DIM)), keys=[1, 2, 42])
    assert e.value.code == -26 and e.value.row == 2
    assert len(db) == 1 and 1 not in db and 2 not in db


def test_key_reuse_after_delete_and_persistence(tmp_path):
    path = str(tmp_path / "p.db")
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, DIM)).astype(np.float32)
    keys = [-(2**63), 2**63 - 1, 0, -1, 123456789, 7]
    with veclite.open(path, dim=DIM) as d:
        d.add(x[:3], keys=keys[:3])
        d.checkpoint()
        d.add(x[3:], keys=keys[3:])
        d.delete(keys[1])
        d.add(x[1:2] * 2, keys=[keys[1]])
    with veclite.open(path) as d:
        assert len(d) == 6 and all(k in d for k in keys)
        for i, key in enumerate(keys):
            q = x[i] * (2 if i == 1 else 1)
            hits, _ = d.search(q, k=1, metric="l2")
            assert int(hits[0]) == key


def test_delete_unknown_key_rolls_back_batch(db):
    db.add(np.ones((3, DIM)), keys=[10, 11, 12])
    with pytest.raises(veclite.VecliteError) as e:
        db.delete([10, 99, 11])
    assert e.value.code == -23 and e.value.row == 1
    assert all(k in db for k in (10, 11, 12))
    db.delete(10, commit=False)
    with pytest.raises(veclite.VecliteError):
        db.delete(10)
    assert db.pending == 1
    db.rollback()
    assert 10 in db


def test_rollback_restores_keys(db):
    db.add(np.ones((2, DIM)), keys=[1, 2])
    db.delete(1, commit=False)
    db.add(np.ones((1, DIM)), keys=[1], commit=False)
    db.add(np.ones((1, DIM)), keys=[3], commit=False)
    db.rollback()
    assert 1 in db and 2 in db and 3 not in db
    hits, _ = db.search(np.ones(DIM), k=5, metric="l2")
    assert sorted(hits.tolist()) == [1, 2]


def test_key_validation(db):
    with pytest.raises(ValueError, match="one key per vector"):
        db.add(np.ones((2, DIM)), keys=[1])
    assert 5 not in db
