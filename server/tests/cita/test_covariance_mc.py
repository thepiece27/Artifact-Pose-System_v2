"""Monte Carlo check of the first-order covariance (I1).

With Gaussian pixel noise matching the sigmas, the normalised camera-centre
errors |c_est - c_true| / sigma_c must fall within 2 sigma about 95 % of the
time (spec: 95 % +- 5 %).
"""
import numpy as np
import pytest

import synth
from app.modules.artifact_pose.covariance import pose_covariance
from app.modules.artifact_pose.geometry import DEFAULT_QUALITY, project
from app.modules.artifact_pose.motion import as_T, camera_center_in_golden, relative_motion
from app.modules.artifact_pose.solver import Block, refine_pose_blocks

K = synth.scaled_intrinsics(960)
D = np.zeros(5)
H = 0.02
DIAMOND = np.array([[-H, H, 0], [H, H, 0], [H, -H, 0], [-H, -H, 0]])


def _trial(rng, sigma_d, sigma_o, huber_delta, n_orb=120):
    T_gw = synth.make_T(np.diag([1.0, -1.0, -1.0]), [0, 0, 0.65])
    c_true = rng.uniform(-0.02, 0.02, 3)
    dR = synth.rotvec_deg_to_R(rng.uniform(-2, 2, 3))
    T_cw = synth.make_T(dR, -dR @ c_true) @ T_gw
    orb = np.c_[rng.uniform(-0.3, 0.3, n_orb), rng.uniform(-0.2, 0.2, n_orb), rng.uniform(-0.02, 0.02, n_orb)]
    r, t = cv_rt(T_cw)
    d2 = project(DIAMOND, r, t, K, D) + rng.normal(0, sigma_d, (4, 2))
    o2 = project(orb, r, t, K, D) + rng.normal(0, sigma_o, (n_orb, 2))
    blocks = [Block(DIAMOND, d2, sigma_d, False, "diamond"), Block(orb, o2, sigma_o, True, "orb")]
    x0 = np.r_[r + rng.normal(0, 0.003, 3), t + rng.normal(0, 0.003, 3)]
    q = DEFAULT_QUALITY
    fit = refine_pose_blocks(x0[:3], x0[3:], blocks, K, D, q, huber_delta=huber_delta)
    T_est = as_T(fit["rvec"], fit["tvec"])
    cov = pose_covariance(T_est, blocks, K, D, huber_delta=huber_delta, T_gw=T_gw,
                          max_reprojection_px=1e9)
    err_mm = (camera_center_in_golden(relative_motion(T_gw, T_est)) - c_true) * 1000
    return err_mm, np.asarray(cov["sigma_center_mm"]), cov


def cv_rt(T):
    import cv2
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]


@pytest.mark.parametrize("huber_delta", [3.0, 2.0])
def test_covariance_two_sigma_coverage(huber_delta):
    rng = np.random.default_rng(2027)
    errs, sigmas = [], []
    for _ in range(150):
        e, s, cov = _trial(rng, sigma_d=0.3, sigma_o=1.0, huber_delta=huber_delta)
        errs.append(e)
        sigmas.append(s)
    z = np.abs(np.asarray(errs)) / np.asarray(sigmas)
    frac = float(np.mean(z <= 2.0))
    assert 0.90 <= frac <= 1.0, frac
    # Also not grossly conservative: normalised RMS close to 1.
    assert 0.75 < float(np.sqrt(np.mean(z ** 2))) < 1.3


def test_posterior_scale_tracks_wrong_sigmas():
    rng = np.random.default_rng(7)
    # Model says sigma_o = 1 px but the real noise is 2 px: s^2 ~ 4 (Huber off).
    _, _, cov = _trial(rng, sigma_d=1.0, sigma_o=1.0, huber_delta=1e6, n_orb=400)
    assert 0.6 < cov["s2"] < 1.6
    T_gw = synth.make_T(np.diag([1.0, -1.0, -1.0]), [0, 0, 0.65])
    orb = np.c_[rng.uniform(-0.3, 0.3, 400), rng.uniform(-0.2, 0.2, 400), np.zeros(400)]
    r, t = cv_rt(T_gw)
    o2 = project(orb, r, t, K, D) + rng.normal(0, 2.0, (400, 2))
    cov2 = pose_covariance(T_gw, [Block(orb, o2, 1.0, True, "orb")], K, D, huber_delta=1e6, T_gw=T_gw)
    assert 2.8 < cov2["s2"] < 5.5
    assert np.all(np.asarray(cov2["sigma_center_mm_post"]) > np.asarray(cov2["sigma_center_mm"]))
