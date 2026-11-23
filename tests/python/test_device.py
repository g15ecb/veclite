import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import veclite

HAS_GPU = veclite.gpu_available()
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="no usable GPU in this build/host")


def ints(rng, n, dim):
    x = rng.integers(-3, 4, (n, dim)).astype(np.float32)
    x[:, 0] += 5
    return x


def same(db, q, **kw):
    cpu = db.search(q, device="cpu", **kw)
    assert db.last_device == "cpu"
    gpu = db.search(q, device="gpu", **kw)
    assert db.last_device == "gpu"
    for (ci, cs), (gi, gs) in zip(cpu, gpu):
        assert np.array_equal(ci, gi) and np.array_equal(cs, gs)


@needs_gpu
@pytest.mark.parametrize("metric", ["l2", "dot"])
@pytest.mark.parametrize("storage", ["f32", "f16"])
def test_gpu_matches_cpu_through_snapshot_cases(tmp_path, metric, storage):
    rng = np.random.default_rng(7)
    data = ints(rng, 9000, 40)
    tags = rng.integers(0, 3, 9000)
    q = ints(rng, 20, 40)
    with veclite.open(str(tmp_path / "g.db"), dim=40, storage=storage) as db:
        db.add(data[:6000], tags[:6000])
        db.checkpoint()
        same(db, q, k=9, metric=metric)
        assert db.info()["gpu_rows"] == 6000
        for d in rng.choice(6000, 500, replace=False):
            db.delete(int(d), commit=False)
        db.commit()
        same(db, q, k=9, metric=metric)
        db.add(data[6000:7000], tags[6000:7000])
        same(db, q, k=9, metric=metric)
        db.checkpoint()
        same(db, q, k=9, metric=metric)
        assert db.info()["gpu_rows"] == 6000, "small checkpoint leaves a CPU-searched gap"
        db.add(data[7000:], tags[7000:])
        db.checkpoint()
        same(db, q, k=9, metric=metric)
        assert db.info()["gpu_rows"] == 9000, "large checkpoint refreshes the GPU copy"
        same(db, q, k=9, metric=metric, tag=[None, 1, 2, 0] * 5)


@needs_gpu
def test_auto_routing(tmp_path):
    rng = np.random.default_rng(8)
    with veclite.open(str(tmp_path / "a.db"), dim=16, gpu_min_batch=32) as db:
        db.add(rng.standard_normal((3000, 16)))
        db.search(rng.standard_normal((8, 16)))
        assert db.last_device == "cpu"
        db.checkpoint()
        db.search(rng.standard_normal((31, 16)))
        assert db.last_device == "cpu"
        db.search(rng.standard_normal((32, 16)))
        assert db.last_device == "gpu"
        db.search(rng.standard_normal((32, 16)), tag=1)
        assert db.last_device == "cpu"
        db.search(rng.standard_normal((32, 16)), device="cpu")
        assert db.last_device == "cpu"


@needs_gpu
def test_gpu_on_empty_main_file_falls_back(tmp_path):
    with veclite.open(str(tmp_path / "e.db"), dim=4, device="gpu") as db:
        db.add(np.ones((3, 4)))
        ids, _ = db.search(np.ones(4), k=3)
        assert len(ids) == 3


@pytest.mark.skipif(HAS_GPU, reason="host has a usable GPU")
def test_no_gpu(tmp_path):
    with veclite.open(str(tmp_path / "n.db"), dim=4) as db:
        db.add(np.ones((3, 4)))
        db.checkpoint()
        db.search(np.ones((200, 4)))
        assert db.last_device == "cpu"
        with pytest.raises(veclite.VecliteError) as e:
            db.search(np.ones(4), device="gpu")
        assert e.value.code == -31


def test_device_validation(tmp_path):
    with pytest.raises(ValueError):
        veclite.open(str(tmp_path / "v.db"), dim=4, device="tpu")
    with pytest.raises(ValueError):
        veclite.open(str(tmp_path / "v.db"), dim=4, gpu_min_batch=0)
    with veclite.open(str(tmp_path / "v.db"), dim=4) as db:
        db.add(np.ones((1, 4)))
        with pytest.raises(ValueError):
            db.search(np.ones(4), device="npu")
