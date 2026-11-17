"""M1-M5 on synthetic renders against the TRUE motion (not the board)."""
from dataclasses import replace

import numpy as np
import pytest

import synth
from app.experiments.profiles import load_profile
from app.modules.artifact_pose import methods as M
from app.modules.artifact_pose.common import extract_orb
from app.modules.artifact_pose.geometry import DEFAULT_QUALITY, refine_pose
from app.modules.artifact_pose.motion import pose_errors
from app.modules.artifact_pose.solver import Block, refine_pose_blocks
from app.modules.ground_truth.reference_board import board_mask, detect_reference_board

VIEWS = [((0.010, 0.0, 0.0), (0.0, 0.0, 0.0)), ((-0.005, 0.0, 0.010), (0.0, 1.0, 0.0))]


def true_dT(c, rot):
    dR = synth.rotvec_deg_to_R(rot)
    return synth.make_T(dR, -dR @ np.asarray(c))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    sc = synth.make_scene(5, width=960)
    rng = np.random.default_rng(9)

    def render(T):
        return sc.render(T, noise_sigma=1.0, rng=rng, supersample=3)

    def mask(img):
        return board_mask(img.shape, detect_reference_board(img, sc.K, sc.D)["hull_polygon_px"], 25)

    L, R = render(sc.T_gw()), render(sc.pose_from_motion([0.1, 0, 0]))
    views = []
    for c, rot in VIEWS:
        imgs = [render(sc.pose_from_motion(c, rot)) for _ in range(3)]
        views.append((imgs, [mask(i) for i in imgs], true_dT(c, rot)))
    base = M.PoseMethodConfig(method="M3", backend="python")
    tmp = tmp_path_factory.mktemp("golden")
    masks = (mask(L), mask(R))
    goldens = {
        "commanded": M.build_golden_model(L, R, sc.K, sc.D, cfg=replace(base, ratio=0.6), workdir=tmp / "c",
                                          stereo_source="commanded", masks=masks, build_markerless=True),
        "commanded_full": M.build_golden_model(L, R, sc.K, sc.D, cfg=base, workdir=tmp / "f",
                                               stereo_source="commanded", masks=masks, build_hybrid=False,
                                               build_markerless=True),
        "diamond": M.build_golden_model(L, R, sc.K, sc.D, cfg=base, workdir=tmp / "d", masks=masks),
    }
    return sc, views, goldens


CASES = [
    ("M1", {}, "diamond", 25.0),
    ("M3", dict(diamond_sigma_px=0.01, orb_sigma_px=1.0), "commanded", 8.0),
    ("M4", dict(diamond_sigma_px=0.15, orb_sigma_px=0.7, covariance=True), "commanded", 5.0),
    ("M2", dict(orb_sigma_px=1.0, ratio=0.6, resize_to=(848, 480)), "commanded", 10.0),
    ("M2", dict(orb_sigma_px=1.0, ratio=0.6), "commanded_full", 3.0),
]


@pytest.mark.parametrize("method,kw,golden_key,max_et", CASES)
def test_single_frame_methods_accuracy(world, method, kw, golden_key, max_et):
    sc, views, goldens = world
    cfg = M.PoseMethodConfig(method=method, backend="python", **kw)
    golden = goldens[golden_key]
    if method == "M2" and cfg.resize_to is None:
        golden.markerless  # full-resolution M2 golden
    for imgs, masks, gt in views:
        est = M.estimate_pose([imgs[0]], golden, cfg, masks=[masks[0]])
        assert est.ok, est.status
        err = pose_errors(est.dT, gt)
        assert err["e_t_mm"] < max_et, (method, err)
        assert err["e_R_deg"] < max_et / 10
        assert est.method == method and est.timings_ms["total"] > 0
        if cfg.covariance:
            assert est.sigma_center_mm is not None and min(est.sigma_center_mm) > 0
            assert np.asarray(est.cov_xi).shape == (6, 6)


@pytest.mark.parametrize("fusion", ["stacked", "average"])
def test_m5_fuses_three_frames(world, fusion):
    sc, views, goldens = world
    cfg = M.PoseMethodConfig(method="M5", frames=3, fusion=fusion, diamond_sigma_px=0.15, orb_sigma_px=0.7,
                             covariance=True, backend="python")
    for imgs, masks, gt in views:
        est = M.estimate_pose(imgs, goldens["commanded"], cfg, masks=masks)
        assert est.ok and est.frames_used == 3
        assert pose_errors(est.dT, gt)["e_t_mm"] < 5.0
        assert est.sigma_center_mm is not None
    with pytest.raises(ValueError, match="expects 3 frames"):
        M.estimate_pose(views[0][0][:2], goldens["commanded"], cfg)


def test_m5_stacked_is_tighter_than_single_frame(world):
    sc, views, goldens = world
    imgs, masks, gt = views[0]
    single = M.PoseMethodConfig(method="M4", diamond_sigma_px=0.15, orb_sigma_px=0.7, covariance=True,
                                backend="python")
    m5 = replace(single, method="M5", frames=3)
    s1 = M.estimate_pose([imgs[0]], goldens["commanded"], single, masks=[masks[0]])
    s3 = M.estimate_pose(imgs, goldens["commanded"], m5, masks=masks)
    assert np.linalg.norm(s3.sigma_center_mm) < np.linalg.norm(s1.sigma_center_mm)


def test_m2_never_uses_the_diamond(world):
    sc, views, goldens = world
    meta = goldens["commanded"].meta["markerless"]
    assert meta["baseline_m"] == 0.1 and meta["slider_x_sign_in_camera"] in (1.0, -1.0)
    assert meta["baseline_check"]["note"].startswith("log only")
    # Blank out the Diamond: M2 must still work, M1 must fail.
    imgs, masks, gt = views[0]
    img = imgs[0].copy()
    x, y = np.rint(sc.project_plane(sc.pose_from_motion(*VIEWS[0]), [sc.diamond_center_m])[0]).astype(int)
    img[y - 60:y + 60, x - 60:x + 60] = 128
    cfg2 = M.PoseMethodConfig(method="M2", orb_sigma_px=1.0, ratio=0.6, backend="python")
    assert M.estimate_pose([img], goldens["commanded_full"], cfg2, masks=[masks[0]]).ok
    assert not M.estimate_pose([img], goldens["commanded"], M.PoseMethodConfig(method="M1")).ok


def test_board_mask_removes_features(world):
    sc, views, goldens = world
    imgs, masks, _ = views[0]
    _, _, xy = extract_orb(imgs[0], backend="python", mask=masks[0])
    m = masks[0]
    assert len(xy) > 100
    assert all(m[int(round(y)), int(round(x))] > 0 for x, y in xy)


def test_solver_blocks_reproduce_refine_pose(world):
    sc, views, goldens = world
    g = goldens["commanded"]
    imgs, masks, _ = views[1]
    cfg = M.PoseMethodConfig(method="M3", backend="python")
    obs = M._hybrid_observations(imgs[0], g, cfg, masks[0], "python")
    d = obs["diamond"]
    q = replace(DEFAULT_QUALITY, diamond_sigma_px=0.2, orb_sigma_px=0.9)
    a = refine_pose(d["rvec"], d["tvec"], d["obj_pts"], d["corners"], obs["o3"], obs["o2"], sc.K, sc.D, q)
    b = refine_pose_blocks(d["rvec"], d["tvec"], [Block(d["obj_pts"], d["corners"], 0.2, False, "diamond"),
                                                  Block(obs["o3"], obs["o2"], 0.9, True, "orb")], sc.K, sc.D, q)
    assert a["accepted"] == b["accepted"]
    assert np.allclose(a["rvec"], b["rvec"], atol=1e-9) and np.allclose(a["tvec"], b["tvec"], atol=1e-10)
    assert a["final_robust_cost"] == pytest.approx(b["final_robust_cost"], rel=1e-9)
    assert a["num_orb_inliers"] == b["num_orb_inliers"]


def test_m4_requires_noise_model(tmp_path):
    prof = load_profile("M4", overrides={"pose": {"noise_model": str(tmp_path / "missing.yaml")}})
    with pytest.raises(M.NoiseModelError, match="never fall back"):
        M.PoseMethodConfig.from_profile(prof)
    (tmp_path / "nm.yaml").write_text("schema: cita.noise_model.v1\ndiamond_sigma_px: 0.07\norb_sigma_px: 0.9\n")
    cfg = M.PoseMethodConfig.from_profile(load_profile("M5", overrides={"pose": {"noise_model": str(tmp_path / "nm.yaml")}}))
    assert (cfg.diamond_sigma_px, cfg.orb_sigma_px, cfg.frames, cfg.covariance) == (0.07, 0.9, 3, True)
    (tmp_path / "bad.yaml").write_text("schema: other\ndiamond_sigma_px: 0.07\norb_sigma_px: 0.9\n")
    with pytest.raises(M.NoiseModelError, match="schema"):
        M.PoseMethodConfig.from_profile(load_profile("M4", overrides={"pose": {"noise_model": str(tmp_path / "bad.yaml")}}))
