"""R6: golden.stereo_source = commanded_epipolar (direction from the epipolar
constraint, length from the commanded baseline), with the slider 5 deg off the
camera x axis."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))
import e0_stereo_check  # noqa: E402
import synth  # noqa: E402
from app.modules.artifact_pose import methods as M  # noqa: E402
from app.modules.artifact_pose.motion import pose_errors  # noqa: E402
from app.modules.artifact_pose.stereo_direction import axis_angle_deg  # noqa: E402
from app.modules.ground_truth.reference_board import board_mask, detect_reference_board  # noqa: E402

MIS = np.deg2rad(5.0)
DIRECTION = np.array([np.cos(MIS), 0.0, np.sin(MIS)])           # slider rotated 5 deg about camera y
VIEWS = [((0.010, 0.0, 0.0), (0, 0, 0)), ((-0.006, 0.0, 0.012), (0, 1.0, 0)), ((0.015, 0.0, -0.010), (0.5, -1.0, 0))]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    sc = synth.make_scene(12)
    rng = np.random.default_rng(3)
    render = lambda T: sc.render(T, noise_sigma=1.0, rng=rng, supersample=3)  # noqa: E731
    mask = lambda img: board_mask(img.shape, detect_reference_board(img, sc.K, sc.D)["hull_polygon_px"], 25)  # noqa: E731
    L, R = render(sc.T_gw()), render(sc.pose_from_motion(0.1 * DIRECTION))
    cfg = M.PoseMethodConfig(method="M4", diamond_sigma_px=0.15, orb_sigma_px=0.7, backend="python")
    tmp = tmp_path_factory.mktemp("g")
    goldens = {src: M.build_golden_model(L, R, sc.K, sc.D, cfg=cfg, workdir=tmp / src, stereo_source=src,
                                         masks=(mask(L), mask(R)))
               for src in ("diamond", "commanded", "commanded_epipolar")}
    views = []
    for c, rot in VIEWS:
        img = render(sc.pose_from_motion(c, rot))
        dR = synth.rotvec_deg_to_R(rot)
        views.append((img, mask(img), synth.make_T(dR, -dR @ np.asarray(c))))
    return sc, cfg, goldens, views


def test_epipolar_direction_is_accurate(world):
    sc, cfg, goldens, views = world
    meta = goldens["commanded_epipolar"].meta["hybrid_stereo"]
    T = np.asarray(meta["T_right_left"])
    t_true = -0.1 * DIRECTION                                    # X_right = X_left - C_right
    assert axis_angle_deg(T[:3, 3], t_true) < 0.3
    assert np.linalg.norm(T[:3, 3]) == pytest.approx(0.1)
    assert float(T[:3, 3] @ t_true) > 0                           # cheirality picked the right sign
    assert meta["angle_to_camera_x_deg"] == pytest.approx(5.0, abs=0.3)
    assert meta["epipolar_inliers"] >= 50 and meta["stereo_source"] == "commanded_epipolar"
    assert meta["diamond_baseline_mm"] is not None and meta["angle_to_diamond_deg"] is not None


def test_m4_with_epipolar_golden_is_not_worse(world):
    sc, cfg, goldens, views = world
    med = {}
    for src, g in goldens.items():
        errs = []
        for img, m, gt in views:
            est = M.estimate_pose([img], g, cfg, masks=[m])
            assert est.ok, (src, est.status)
            errs.append(pose_errors(est.dT, gt)["e_t_mm"])
        med[src] = float(np.median(errs))
    assert med["commanded_epipolar"] <= med["commanded"] + 0.1, med
    assert med["commanded_epipolar"] <= med["diamond"] + 0.1, med


def test_e0_stereo_check_on_synthetic_session(tmp_path):
    session = synth.make_pose_session(tmp_path, session_id="E03S", frames_per_pose=1, stereo_direction=DIRECTION,
                                      offsets=[{"x_mm": 0, "z_mm": 0, "pan_deg": 0, "tilt_deg": 0}])
    out = tmp_path / "stereo_check.json"
    assert e0_stereo_check.main(["--sessions", str(session), "--out", str(out), "--log-root", str(tmp_path / "logs")]) == 0
    res = json.loads(out.read_text())
    row = res["sessions"][0]
    assert row["angle_epipolar_x_deg"] == pytest.approx(5.0, abs=0.5) and row["epipolar_inliers"] > 50
    assert {"diamond_baseline_mm", "angle_diamond_epipolar_deg", "angle_diamond_x_deg"} <= set(row)
    assert res["decision"]["recommended_stereo_source"] in ("diamond", "commanded_epipolar")
    assert "2.0 mm" in res["decision"]["rule"] and "1.0 deg" in res["decision"]["rule"]
