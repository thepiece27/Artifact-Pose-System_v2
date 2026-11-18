"""Phase 4 acceptance on synthetic pairs (+ E2 evaluation chain)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "server" / "tools" / "experiments"))

import calibrate_thresholds  # noqa: E402
import define_roi  # noqa: E402
import eval_change  # noqa: E402
import eval_reloc  # noqa: E402
import make_tables  # noqa: E402
import replay_change  # noqa: E402
import run_reloc_trials  # noqa: E402
import synth  # noqa: E402
from app.experiments.trial_log import read_jsonl  # noqa: E402
from app.modules.change_detection import photometric, registration, roi as roi_mod, ssim_map  # noqa: E402


@pytest.fixture(scope="module")
def pair_sets(tmp_path_factory):
    root = tmp_path_factory.mktemp("e3")
    val = synth.make_change_pairs(root, seed=21, split="val", n_unchanged=4, n_changed=2)
    test = synth.make_change_pairs(root, seed=22, split="test", n_unchanged=3, n_changed=3)
    return root, val, test


def test_registration_blocks(pair_sets):
    root, val, test = pair_sets
    man = json.loads((test / "pairs.json").read_text())
    poly = np.array(man["roi"]["polygon_ref_px"])
    ref, cur = cv2.imread(str(test / "golden.png")), cv2.imread(str(test / "cur_004.png"))
    leg = registration.legacy_whole_frame(cur, ref, poly)
    roi = registration.roi_homography(cur, ref, poly)
    assert leg.H_ref_from_cur is not None and roi.H_ref_from_cur is not None
    assert roi.ecc_ok and roi.roi_afd_px is not None and roi.roi_afd_px > 0
    assert roi.valid_mask[int(poly[:, 1].mean()), int(poly[:, 0].mean())] == 255
    assert roi.valid_mask[5, 5] == 0          # outside the painting
    # legacy_whole_frame equals the moved legacy function (same aligned image).
    from app.modules.change_detection.legacy import sift_align_with_mask
    cv2.setRNGSeed(0)
    a, _, n = sift_align_with_mask(cur, ref)
    cv2.setRNGSeed(0)
    leg = registration.legacy_whole_frame(cur, ref, poly)
    assert n == leg.n_inliers and np.array_equal(a, leg.aligned)


def test_photometric_and_ssim():
    rng = np.random.default_rng(0)
    ref = rng.integers(20, 230, (200, 300, 3), dtype=np.uint8)
    cur = np.clip(ref.astype(float) * 0.8 + 15, 0, 255).astype(np.uint8)
    out, _, info = photometric.normalize(cur, ref, None, "gain_offset")
    assert np.abs(out.astype(int) - ref.astype(int)).mean() < 1.5
    assert all(abs(a - 1.25) < 0.02 for a, b in info["gain_offset"])
    s = ssim_map.standard_ssim(ref, ref, None, work_long_side_px=256)
    assert s["ssim"] == pytest.approx(1.0) and float(s["d"].max()) == 0.0
    leg = ssim_map.legacy_ssim(ref, cur)
    assert 0 < leg["ssim"] < 1 and leg["diff_uint8"].dtype == np.uint8


def test_roi_tools(tmp_path):
    scene = synth.make_scene(3)
    img = scene.render(scene.T_gw(), supersample=3)
    path = tmp_path / "golden.png"
    cv2.imwrite(str(path), img)
    poly_true = scene.painting_polygon_image(scene.T_gw())
    poly_txt = ";".join(f"{x:.1f},{y:.1f}" for x, y in poly_true)
    assert define_roi.main(["--image", str(path), "--polygon", poly_txt, "--size-mm", "297,210",
                            "--out", str(tmp_path / "roi.json"), "--log-root", str(tmp_path / "logs")]) == 0
    roi = roi_mod.load_roi(tmp_path / "roi.json")
    assert roi_mod.px_per_mm(roi) == pytest.approx(scene.px_per_mm_golden(), rel=0.02)
    assert (tmp_path / "roi.png").exists()


def test_e3_chain_acceptance(pair_sets, tmp_path):
    root, val, test = pair_sets
    thr = tmp_path / "thresholds.yaml"
    # Test data is refused for tuning.
    with pytest.raises(SystemExit, match="non-validation"):
        calibrate_thresholds.main(["--pairs", str(test), "--split", "val", "--out", str(thr),
                                   "--log-root", str(tmp_path / "logs")])
    assert calibrate_thresholds.main(["--pairs", str(val), "--split", "val", "--out", str(thr),
                                      "--log-root", str(tmp_path / "logs")]) == 0
    t = yaml.safe_load(thr.read_text())
    assert t["schema"] == "cita.thresholds.v1" and 0 < t["tau"] < 1 and t["achieved_far_val"] <= 0.05
    assert replay_change.main(["--pairs", str(test), "--pipelines", "P0,P1,P2,P3,P4", "--thresholds", str(thr),
                               "--mock-yolo", "--log-root", str(tmp_path / "logs"), "--run-id", "e3",
                               "--out-dir", str(tmp_path / "out")]) == 0
    rows = read_jsonl(tmp_path / "logs" / "E3" / "e3.jsonl")
    assert len(rows) == 5 * 6 and {r["status"] for r in rows} == {"ok"}
    p3 = [r for r in rows if r["arm"] == "P3" and r["input"]["pair_id"].endswith("005")][0]
    pkg = json.loads(Path(p3["output"]["ai_input_package_path"]).read_text())
    assert pkg["schema"] == "cita.ai_input.v1" and pkg["regions"] and pkg["threshold"]["mode"] == "fixed"
    assert eval_change.main(["--logs", str(tmp_path / "logs" / "E3" / "e3.jsonl"), "--pairs", str(test),
                             "--out", str(tmp_path / "ev"), "--n-boot", "200", "--log-root", str(tmp_path / "logs")]) == 0
    ev = json.loads((tmp_path / "ev.json").read_text())["summary"]
    # Acceptance: fewer false alarms than P0 on unchanged pairs; recall >= 0.9 for damage >= 5 mm.
    assert ev["P2"]["false_alarm_rate_unchanged"] < ev["P0"]["false_alarm_rate_unchanged"]
    assert ev["P3"]["false_alarm_rate_unchanged"] < ev["P0"]["false_alarm_rate_unchanged"]
    for code in ("P2", "P3"):
        by = ev[code]["recall_by_size"]
        assert all(v is None or v >= 0.9 for k, v in by.items() if k in ("medium", "large"))
        assert ev[code]["recall"] >= 0.9 and ev[code]["roc_auc"] >= 0.9
    assert ev["P3"]["class_macro_f1"] is not None
    assert ev["P2"]["sensitivity_vs_offset"] and ev["P2"]["sensitivity_vs_offset_deg"]
    assert make_tables.main(["--eval", str(tmp_path / "ev.json"), "--out", str(tmp_path / "tab"),
                             "--log-root", str(tmp_path / "logs")]) == 0
    assert "| P3 |" in (tmp_path / "tab.md").read_text(encoding="utf-8")


def test_e2_eval_chain(tmp_path):
    assert run_reloc_trials.main(["--virtual", "--paintings", "P001", "--n-perturb", "4", "--arms",
                                  "C-A,C-B,C-C,C-D", "--log-root", str(tmp_path), "--run-id", "e2",
                                  "--jacobian-dir", str(tmp_path / "jac")]) == 0
    assert eval_reloc.main(["--logs", str(tmp_path / "E2" / "e2.jsonl"), "--out", str(tmp_path / "ev2"),
                            "--n-boot", "100", "--log-root", str(tmp_path), "--run-id", "ev2"]) == 0
    assert read_jsonl(tmp_path / "E2" / "ev2.jsonl")[0]["arm"] == "eval_reloc"
    ev = json.loads((tmp_path / "ev2.json").read_text())
    assert set(ev["arms"]) == {"C-A", "C-B", "C-C", "C-D"} and ev["n_paired_blocks"] == 4
    for a in ev["arms"]:
        assert ev["summary"][a]["n_trials"] == 4 and ev["summary"][a]["success_gt_rate"] is not None
    assert len(ev["paired_tests"]["mcnemar_success_gt"]) == 6
    assert make_tables.main(["--eval", str(tmp_path / "ev2.json"), "--out", str(tmp_path / "t2"),
                             "--log-root", str(tmp_path)]) == 0
