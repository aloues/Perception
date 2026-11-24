"""ThreadLoader de EXP-061: mismo orden y contenido que los lotes pedidos (sin procesos hijos)."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp061_finetune_local import ListBatches, ThreadLoader  # noqa: E402


class FakeDS:
    def __getitem__(self, i):
        return torch.full((2, 2, 3), i, dtype=torch.uint8)


def test_threadloader_order_and_content():
    bs = ListBatches()
    bs.batches = [[3, 1], [7], [0, 2, 5], [9, 4]]
    out = list(ThreadLoader(FakeDS(), bs, threads=3, prefetch=2))
    assert len(out) == 4
    for got, want in zip(out, bs.batches):
        assert got.shape[0] == len(want)
        assert got[:, 0, 0, 0].tolist() == want


def test_threadloader_reusable_after_reassigning_batches():
    bs = ListBatches()
    loader = ThreadLoader(FakeDS(), bs, threads=2)
    bs.batches = [[1], [2]]
    assert [b[0, 0, 0, 0].item() for b in loader] == [1, 2]
    bs.batches = [[5, 6]]
    assert [b[:, 0, 0, 0].tolist() for b in loader] == [[5, 6]]
