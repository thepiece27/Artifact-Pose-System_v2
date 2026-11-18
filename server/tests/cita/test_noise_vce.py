"""R2: variance-component estimation vs static jitter.

Simulation: each session has a FIXED Diamond corner error (print/localisation
bias, ~0.3 px, seeded) plus 0.03 px temporal noise; ORB observations have
1.0 px noise. VCE must recover sigma_d ~ 0.3 and sigma_o ~ 1.0; the jitter
estimator only sees the 0.03 px temporal part.
"""
import numpy as np
import pytest

import synth
from app.experiments.profiles import load_profile
from app.modules.artifact_pose import methods as M
from app.modules.artifact_pose.geometry import project
from app.modules.artifact_pose.noise_vce import FrameObservation, diamond_jitter_sigma, estimate_vce

K = synth.scaled_intrinsics(1920)
D = np.zeros(5)
H = 0.02
DIAMOND = np.array([[-H, H, 0], [H, H, 0], [H, -H, 0], [-H, -H, 0]])
T_GW = synth.make_T(np.diag([1.0, -1.0, -1.0]), [0.24, -0.10, 0.65])   # Diamond off-centre, as on the rig


def rt(T):
    import cv2
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]


def simulate(seed=2027, n_sessions=4, frames_per_session=6, n_orb=120, bias_px=0.3, temporal_px=0.03,
             orb_px=1.0):
    rng = np.random.default_rng(seed)
    frames, bursts = [], []
    for s in range(n_sessions):
        bias = rng.normal(0.0, bias_px, (4, 2))                       # fixed for the session
        orb3 = np.c_[rng.uniform(-0.45, 0.10, n_orb), rng.uniform(-0.2, 0.2, n_orb), rng.uniform(-0.01, 0.01, n_orb)]
        for k in range(frames_per_session):
            c = rng.uniform(-0.02, 0.02, 3)
            dR = synth.rotvec_deg_to_R(rng.uniform(-2, 2, 3))
            T = synth.make_T(dR, -dR @ c) @ T_GW
            r, t = rt(T)
            d2 = project(DIAMOND, r, t, K, D) + bias + rng.normal(0, temporal_px, (4, 2))
            o2 = project(orb3, r, t, K, D) + rng.normal(0, orb_px, (n_orb, 2))
            frames.append(FrameObservation(DIAMOND, d2, orb3, o2, r + rng.normal(0, 2e-3, 3),
                                           t + rng.normal(0, 2e-3, 3), session_id=f"S{s}", frame_id=f"{s}_{k}"))
        # Static burst of the same session (for the jitter estimator).
        r, t = rt(T_GW)
        bursts.append([project(DIAMOND, r, t, K, D) + bias + rng.normal(0, temporal_px, (4, 2)) for _ in range(10)])
    return frames, bursts


def test_vce_recovers_both_sigmas_and_jitter_does_not():
    frames, bursts = simulate()
    res = estimate_vce(frames, K, D)
    assert res["converged"] and res["n_iterations"] <= 20
    assert 0.2 <= res["sigma_d_px"] <= 0.45, res["sigma_d_px"]
    assert 0.8 <= res["sigma_o_px"] <= 1.25, res["sigma_o_px"]
    assert res["redundancy_d"] > 20 and res["warning"] is None
    # Redundancies sum to m - 6 per frame.
    assert res["redundancy_d"] + res["redundancy_o"] == pytest.approx(
        res["n_residuals_d"] + res["n_residuals_o"] - 6 * res["n_frames"], rel=1e-6)
    jitter = np.median([diamond_jitter_sigma(b) for b in bursts])
    assert jitter < 0.06
    assert res["iterations"][0]["sigma_d_px"] == 0.5 and res["iterations"][0]["sigma_o_px"] == 1.5


def test_vce_warns_when_diamond_redundancy_is_small():
    frames, _ = simulate(n_sessions=1, frames_per_session=2)
    res = estimate_vce(frames, K, D)
    assert res["warning"] and "redundancy" in res["warning"]


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_load_noise_model_versions_and_source_check(tmp_path):
    v1 = _write(tmp_path / "v1.yaml", "schema: cita.noise_model.v1\ndiamond_sigma_px: 0.1\norb_sigma_px: 0.9\n")
    assert M.load_noise_model(v1, expected_stereo_source="diamond")["orb_sigma_px"] == 0.9
    v2 = _write(tmp_path / "v2.yaml", "schema: cita.noise_model.v2\nestimator: vce\ngolden_stereo_source: diamond\n"
                                      "diamond_sigma_px: 0.3\norb_sigma_px: 1.1\n")
    assert M.load_noise_model(v2, expected_stereo_source="diamond")["estimator"] == "vce"
    with pytest.raises(M.NoiseModelError, match="golden_stereo_source"):
        M.load_noise_model(v2, expected_stereo_source="commanded_epipolar")
    prof = load_profile("M4", overrides={"pose": {"noise_model": str(v2)},
                                         "golden": {"stereo_source": "commanded"}})
    with pytest.raises(M.NoiseModelError, match="golden_stereo_source"):
        M.PoseMethodConfig.from_profile(prof)
    cfg = M.PoseMethodConfig.from_profile(load_profile("M4", overrides={"pose": {"noise_model": str(v2)}}))
    assert (cfg.diamond_sigma_px, cfg.orb_sigma_px) == (0.3, 1.1)
    bad = _write(tmp_path / "bad.yaml", "schema: cita.noise_model.v3\ndiamond_sigma_px: 0.3\norb_sigma_px: 1.1\n")
    with pytest.raises(M.NoiseModelError, match="schema"):
        M.load_noise_model(bad)
