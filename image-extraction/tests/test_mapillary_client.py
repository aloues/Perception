"""Cliente de Mapillary sin red: una sesión falsa devuelve teselas llenas o 500 en bbox grandes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mapillary_client as mc  # noqa: E402


class R:
    def __init__(self, status, data=None):
        self.status_code, self._d = status, data or []

    def json(self):
        return {"data": self._d}


class Session:
    """Bbox de lado > 0.5 → HTTP 500 (tesela demasiado densa); si no, 1 imagen por bbox."""

    def __init__(self):
        self.n = 0

    def get(self, url, params, timeout):
        self.n += 1
        x0, y0, x1, y1 = map(float, params["bbox"].split(","))
        if x1 - x0 > 0.5:
            return R(500)
        return R(200, [{"id": f"{x0:.3f}_{y0:.3f}", "geometry": {"coordinates": [x0, y0]}, "captured_at": 1.45e12}])


def test_persistent_500_splits_tile_instead_of_failing():
    c = mc.MapillaryClient("TOKEN", session=Session(), limit=3, sleep=lambda s: None)
    data = c.images_in_bbox((0.0, 0.0, 1.0, 1.0))
    assert len(data) == 4 and not c.failed            # 1 → 4 sub-teselas que sí responden


def test_unresolvable_tile_is_recorded_not_raised():
    c = mc.MapillaryClient("TOKEN", session=Session(), limit=3, sleep=lambda s: None)
    assert c.images_in_bbox((0.0, 0.0, 1.0, 1.0), max_depth=0) == [] and c.failed == [(0.0, 0.0, 1.0, 1.0)]


def test_to_frame_parses_year():
    df = mc.to_frame([{"id": 1, "geometry": {"coordinates": [-77.0, -12.0]}, "captured_at": 1.45e12}])
    assert df.loc[0, "year"] == 2015 and df.loc[0, "lon"] == -77.0
