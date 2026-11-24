"""Tests del collector SIN red ni API de pago: una sesión HTTP falsa devuelve metadata y JPEG sintéticos."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import LineString, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streetview_collector as sc  # noqa: E402

CFG = sc.load_config()


def jpeg(seed: int, w: int = 640, h: int = 400) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, status: int, content: bytes):
        self.status_code, self.content = status, content


class FakeSession:
    """Metadata OK salvo en lat > 0 (ZERO_RESULTS); imágenes ruidosas; `fail_headings` → 500 siempre."""

    def __init__(self, fail_headings=(), corrupt_headings=()):
        self.calls: list[tuple[str, dict]] = []
        self.fail, self.corrupt = set(fail_headings), set(corrupt_headings)

    def get(self, url, params, timeout):
        self.calls.append((url, dict(params)))
        lat, lon = map(float, params.get("location", "0,0").split(","))
        if url == sc.METADATA_URL:
            if lat > 0:
                return FakeResponse(200, json.dumps({"status": "ZERO_RESULTS"}).encode())
            return FakeResponse(200, json.dumps({"status": "OK", "pano_id": f"P{lat:.4f}", "date": "2024-03",
                                                 "location": {"lat": lat + 1e-5, "lng": lon}}).encode())
        h = int(params["heading"])
        if h in self.fail:
            return FakeResponse(500, b"")
        if h in self.corrupt:
            return FakeResponse(200, b"\xff\xd8not-a-jpeg" * 3000)
        return FakeResponse(200, jpeg(int(abs(lat) * 1e4) + h))


def client(session) -> sc.StreetViewClient:
    net = {**CFG["network"], "max_requests_per_s": 1e6, "max_retries": 1}
    return sc.StreetViewClient(CFG["camera"], net, "SECRET-KEY", session=session, sleep=lambda s: None)


def spots(n: int = 2, lat0: float = -12.1) -> pd.DataFrame:
    return pd.DataFrame({"spot_id": [f"S{i}" for i in range(n)], "district": "lince", "spot_source": "test",
                         "source_ref": "", "lat_req": [lat0 - 0.001 * i for i in range(n)],
                         "lon_req": -77.03, "segment_key": "", "in_district": True})


# ─── cámara ───────────────────────────────────────────────────────────────────

def test_headings_match_historical_range():
    assert sc.headings(CFG["camera"]) == list(range(0, 360, 30))


def test_headings_never_duplicate_0_and_360():
    cam = {**CFG["camera"], "heading_start": 330}
    hs = sc.headings(cam)
    assert len(set(hs)) == 12 and all(0 <= h < 360 for h in hs) and 0 in hs and 360 not in hs
    with pytest.raises(ValueError):
        sc.headings({**CFG["camera"], "heading_step": 60})   # 12 × 60° da la vuelta dos veces


def test_camera_params_are_the_historical_ones():
    cam = CFG["camera"]
    assert (cam["size"], cam["fov"], cam["pitch"]) == ("640x400", 90, 0)


def test_fingerprint_changes_only_with_camera_criteria():
    fp = sc.camera_fingerprint(CFG["camera"])
    assert fp == sc.camera_fingerprint(dict(CFG["camera"]))
    assert fp != sc.camera_fingerprint({**CFG["camera"], "fov": 120})
    assert fp == sc.camera_fingerprint({**CFG["camera"], "return_error_code": False})  # no altera la imagen


def test_image_request_reproduces_historical_query():
    s = FakeSession()
    client(s).image(-12.1, -77.03, 30)
    url, p = s.calls[-1]
    assert url == sc.STATIC_URL
    assert p["size"] == "640x400" and p["fov"] == 90 and p["pitch"] == 0 and p["heading"] == 30
    assert p["location"] == "-12.1,-77.03"


def test_key_is_redacted():
    assert "SECRET" not in sc.redact("https://x/streetview?size=1&key=SECRET-KEY&heading=0")


# ─── recolección / reanudación ────────────────────────────────────────────────

def test_collect_downloads_12_views_with_manifest(tmp_path):
    s = FakeSession()
    stats = sc.collect(CFG, spots(2), tmp_path, client(s))
    assert stats["views_downloaded"] == 24 and not stats["aborted"]
    V = pd.read_csv(tmp_path / "_manifest" / "views.csv")
    assert set(V["heading_deg"]) == set(range(0, 360, 30))
    assert V["view_id"].is_unique
    assert (V["download_status"] == "ok").all() and V["sha256"].str.len().eq(64).all()
    assert (tmp_path / "S0" / "heading_330.jpg").exists()
    S = pd.read_csv(tmp_path / "_manifest" / "spots.csv")
    assert set(S["status"]) == {"complete"} and S["pano_id"].notna().all()


def test_resume_does_not_redownload_valid_views(tmp_path):
    sc.collect(CFG, spots(2), tmp_path, client(FakeSession()))
    s2 = FakeSession()
    stats = sc.collect(CFG, spots(2), tmp_path, client(s2))
    assert stats["views_downloaded"] == 0 and stats["views_skipped_valid"] == 24
    assert s2.calls == []


def test_resume_redownloads_tampered_file(tmp_path):
    sc.collect(CFG, spots(1), tmp_path, client(FakeSession()))
    p = tmp_path / "S0" / "heading_60.jpg"
    p.write_bytes(jpeg(999))                      # hash ya no coincide con el manifiesto
    s2 = FakeSession()
    stats = sc.collect(CFG, spots(1), tmp_path, client(s2))
    assert stats["views_downloaded"] == 1
    assert [c[1]["heading"] for c in s2.calls if c[0] == sc.STATIC_URL] == [60]


def test_corrupt_and_failed_views_are_recorded_not_saved(tmp_path):
    s = FakeSession(fail_headings={90}, corrupt_headings={180})
    stats = sc.collect(CFG, spots(1), tmp_path, client(s))
    assert stats["views_downloaded"] == 10 and stats["views_failed"] == 2
    assert not (tmp_path / "S0" / "heading_90.jpg").exists()
    assert not (tmp_path / "S0" / "heading_180.jpg").exists()
    S = pd.read_csv(tmp_path / "_manifest" / "spots.csv")
    assert S["status"].iloc[0] == "partial"
    inc = pd.read_csv(tmp_path / "_manifest" / "incidents.csv")
    assert {"view"} <= set(inc["kind"])


def test_no_coverage_spot_is_not_billed(tmp_path):
    s = FakeSession()
    sp = spots(1, lat0=+1.0)                       # FakeSession: lat > 0 → ZERO_RESULTS
    stats = sc.collect(CFG, sp, tmp_path, client(s))
    assert stats["spots_no_coverage"] == 1
    assert not [c for c in s.calls if c[0] == sc.STATIC_URL]


def test_spots_outside_district_are_skipped_by_default(tmp_path):
    sp = spots(2)
    sp.loc[1, "in_district"] = False
    s = FakeSession()
    sc.collect(CFG, sp, tmp_path, client(s))
    assert {c[1]["location"] for c in s.calls} == {"-12.1,-77.03"}


def test_abort_after_consecutive_errors(tmp_path):
    cfg = {**CFG, "network": {**CFG["network"], "max_consecutive_errors": 3}}
    s = FakeSession(fail_headings=set(range(0, 360, 30)))
    stats = sc.collect(cfg, spots(3), tmp_path, client(s))
    assert stats["aborted"] and stats["views_downloaded"] == 0


# ─── planificación ────────────────────────────────────────────────────────────

def test_sample_along_segments_spacing():
    pts = sc.sample_along_segments([("a", LineString([(0, 0), (100, 0)])),
                                    ("b", LineString([(0, 0), (30, 0)])),
                                    ("c", LineString([(0, 0), (5, 0)]))], spacing_m=50, min_segment_m=10)
    assert [(k, round(x)) for k, _, x, _ in pts] == [("a", 25), ("a", 75), ("b", 15)]


def test_grid_points_reproduce_extract_district_arithmetic():
    pts = sc.grid_points((-77.03, -12.16, -77.0, -12.13), 80)
    old = pd.read_csv(sc.REPO_ROOT / "image-extraction" / "grid_barranco.csv")
    assert len(pts) == len(old)
    assert np.allclose(np.array(pts), old[["lat", "lng"]].to_numpy(), atol=1e-7)


def test_crime_points_unavailable_for_lince():
    with pytest.raises(ValueError, match="no hay registros"):
        sc.spots_from_crime_points(CFG, "lince")


def test_crime_points_replicate_historical_ids_san_isidro():
    S = sc.spots_from_crime_points(CFG, "san_isidro", 2016)
    assert len(S) == 1258 and S["spot_id"].str.endswith(".0").all()


def test_plan_marks_points_outside_polygon(monkeypatch):
    poly = Polygon([(-77.04, -12.11), (-77.02, -12.11), (-77.02, -12.09), (-77.04, -12.09)])
    fake = pd.DataFrame({"spot_id": ["in", "out"], "district": "lince", "spot_source": "crime_points",
                         "source_ref": "", "lat_req": [-12.10, -12.20], "lon_req": [-77.03, -77.03],
                         "segment_key": "", "in_district": np.nan})
    monkeypatch.setattr(sc, "spots_from_crime_points", lambda cfg, d, y: fake)
    S = sc.plan_spots(CFG, "lince", "crime_points", polygon=poly)
    assert S.set_index("spot_id")["in_district"].to_dict() == {"in": True, "out": False}


def test_cost_estimate_counts_12_requests_per_spot():
    c = sc.estimate_cost(1000, CFG)
    assert c["image_requests"] == 12000 and c["metadata_requests"] == 1000
    assert c["usd_estimate_without_free_quota"] == pytest.approx(12000 / 1000 * CFG["cost"]["usd_per_1000_static"])


# ─── QA ───────────────────────────────────────────────────────────────────────

def test_verify_reports_complete_dataset(tmp_path):
    sp = spots(2)
    sc.collect(CFG, sp, tmp_path, client(FakeSession()))
    rep = sc.verify(CFG, sp, tmp_path)
    assert rep["spots_12_of_12"] == 2 and rep["files_missing"] == 0 and rep["files_corrupt"] == 0
    assert rep["headings_outside_expected"] == 0 and rep["distinct_size_mode"] == [[640, 400, "RGB"]]
    assert rep["exact_duplicate_hashes_across_spots"] == 0


def test_near_duplicate_counter_ignores_same_spot():
    h = np.array([0, 1, 0b111111, 0], dtype=np.uint64)
    g = np.array(["a", "a", "b", "b"])
    # pares de grupos distintos con Hamming ≤ 2: (0,3)=0, (1,3)=1 → 2
    assert sc.count_near_duplicates(h, g, 2) == 2


# ─── presupuesto, deduplicación, huella y .env (prompt maestro 2026-10-29) ────

def budget(tmp_path, max_run=10_000, cap=10_000) -> sc.Budget:
    return sc.Budget(tmp_path / "_budget" / "ledger.csv", cap, max_run, 7.0, tramo="TX", district="lince",
                     spot_source="test")


def test_budget_stops_before_cap_and_writes_ledger(tmp_path):
    s = FakeSession()
    b = budget(tmp_path, max_run=20)                    # peor caso por imagen = 1 + max_retries(4) = 5
    stats = sc.collect(CFG, spots(3), tmp_path / "ds", client(s), budget=b)
    n_img = len([c for c in s.calls if c[0] == sc.STATIC_URL])
    assert stats["budget_stopped"] and n_img == 16 and b.used == 16
    L = pd.read_csv(tmp_path / "_budget" / "ledger.csv")
    assert len(L) == 1 and L["images_requested"].iloc[0] == 16 and L["images_cumulative"].iloc[0] == 16


def test_budget_counts_prior_ledger_rows(tmp_path):
    sc.collect(CFG, spots(1), tmp_path / "ds", client(FakeSession()), budget=budget(tmp_path, cap=30))
    b2 = budget(tmp_path, cap=30)
    assert b2.prior == 12 and b2.remaining() == 18
    s = FakeSession()
    stats = sc.collect(CFG, spots(3), tmp_path / "ds", client(s), budget=b2)
    assert stats["budget_stopped"] and stats["billable_images_cumulative"] <= 30
    L = pd.read_csv(tmp_path / "_budget" / "ledger.csv")
    assert len(L) == 2 and L["images_cumulative"].iloc[-1] == L["images_requested"].sum()


def test_duplicate_coordinates_are_requested_once(tmp_path):
    sp = spots(3)
    sp.loc[2, ["lat_req", "lon_req"]] = sp.loc[0, ["lat_req", "lon_req"]].to_numpy()
    s = FakeSession()
    stats = sc.collect(CFG, sp, tmp_path, client(s))
    assert stats["spots_dup"] == 1 and stats["views_downloaded"] == 24
    S = pd.read_csv(tmp_path / "_manifest" / "spots.csv", dtype=str).set_index("spot_id")
    assert S.loc["S2", "status"] == "dup" and S.loc["S2", "dup_of"] == "S0"
    rep = sc.verify(CFG, sp, tmp_path)
    assert rep["spots_dup_of_complete"] == 1 and rep["coverage_rate_vs_planned_in_district"] == 1.0


def test_resume_refuses_other_camera_fingerprint(tmp_path):
    sc.collect(CFG, spots(1), tmp_path, client(FakeSession()))
    cfg = {**CFG, "camera": {**CFG["camera"], "fov": 120}}
    with pytest.raises(RuntimeError, match="huella"):
        sc.collect(cfg, spots(1), tmp_path, client(FakeSession()))


def test_env_loader_parses_without_overwriting(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# c\nA=1\nexport B='x|y=z'\nC=\"q\"\nA2\n", encoding="utf-8")
    env = {"C": "keep"}
    assert sc.load_env_file(p, env) == ["A", "B"]
    assert env == {"A": "1", "B": "x|y=z", "C": "keep"}


def test_cap_from_config_is_270_usd():
    assert sc.cap_total_images(CFG) == int(270 / 7.0 * 1000)


class SamePanoSession(FakeSession):
    """Dos coordenadas distintas que la metadata resuelve al MISMO panorama."""

    def get(self, url, params, timeout):
        r = super().get(url, params, timeout)
        if url == sc.METADATA_URL:
            d = json.loads(r.content.decode())
            if d.get("status") == "OK":
                d["pano_id"] = "SAME"
            return FakeResponse(200, json.dumps(d).encode())
        return r


def test_shared_pano_is_requested_once(tmp_path):
    s = SamePanoSession()
    stats = sc.collect(CFG, spots(3), tmp_path, client(s))
    assert stats["views_downloaded"] == 12 and stats["spots_dup"] == 2
    S = pd.read_csv(tmp_path / "_manifest" / "spots.csv", dtype=str).set_index("spot_id")
    assert (S.loc[["S1", "S2"], "dup_of"] == "S0").all() and (S.loc[["S1", "S2"], "dup_reason"] == "pano").all()
    s2 = SamePanoSession()                                  # al reanudar, el pano ya descargado sigue siendo canónico
    sc.collect(CFG, spots(4), tmp_path, client(s2))
    assert not [c for c in s2.calls if c[0] == sc.STATIC_URL]


def test_parallel_runs_do_not_overwrite_each_other_in_ledger(tmp_path):
    a, b = budget(tmp_path), budget(tmp_path)               # dos corridas abiertas a la vez (T1 y T2)
    a.row["run_id"], b.row["run_id"] = "A", "B"
    a.charge(10); a.commit()
    b.charge(5); b.commit()
    a.charge(10); a.commit()                                # A se creó antes de que existiera la fila de B
    L = pd.read_csv(tmp_path / "_budget" / "ledger.csv").set_index("run_id")
    assert L.loc["A", "images_requested"] == 20 and L.loc["B", "images_requested"] == 5
    assert a.prior == 5 and a.remaining() == 10_000 - 25
