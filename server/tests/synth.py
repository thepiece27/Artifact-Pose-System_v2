"""Seeded synthetic planar scenes for CITA tests, fixtures and smoke runs.

Everything here is metric and deterministic for a given seed:

* world/plane frame: plane Z = 0, X right, Y up (ArUco marker convention);
* camera frame: OpenCV (x right, y down, z forward);
* ``T_cw`` maps world coordinates into camera coordinates;
* relative motion ``dT = T_cw(current) @ inv(T_cw(golden))`` (Appendix A.1).

The scene contains a textured board ("formex"), an A4 painting with its own
texture, a ChArUco Diamond (IDs 10/20/30/40, 40/25 mm) and an independent
reference ChArUco board (DICT_5X5_100) used as ground truth.  Images are
rendered through the plane-induced homography (Appendix A.5), without lens
distortion (D = 0).
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2 import aruco

# Intrinsics of camera_params_lens_1.5.yaml (3840x2160); scaled for speed.
REAL_K = np.array([[2922.2458367375621, 0.0, 1871.3666351471443],
                   [0.0, 2935.1434699736446, 1047.8804594857584],
                   [0.0, 0.0, 1.0]])
REAL_SIZE = (3840, 2160)

DIAMOND_IDS = (10, 20, 30, 40)
DIAMOND_SQUARE_M, DIAMOND_MARKER_M = 0.040, 0.025

BOARD_DEFAULT = {"squares_x": 5, "squares_y": 7, "square_mm": 30.0, "marker_mm": 22.0,
                 "dictionary": "DICT_5X5_100", "ids_offset": 0}


def scaled_intrinsics(width: int) -> np.ndarray:
    """Scale the real calibration to ``width`` pixels (pixel-centre convention)."""
    s = width / REAL_SIZE[0]
    K = REAL_K.copy()
    K[0, 0] *= s
    K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return K


def rotvec_deg_to_R(rotvec_deg) -> np.ndarray:
    return cv2.Rodrigues(np.deg2rad(np.asarray(rotvec_deg, dtype=np.float64)).reshape(3, 1))[0]


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def structured_texture(h: int, w: int, rng: np.random.Generator, *, colourful: bool) -> np.ndarray:
    """Multi-scale noise + sinusoids + random strokes: many ORB/SIFT corners."""
    img = np.zeros((h, w, 3), np.float32)
    for scale in (8, 24, 64):
        small = rng.random((max(2, h // scale), max(2, w // scale), 3)).astype(np.float32)
        img += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * (scale / 64.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(4):
        fx, fy, ph = rng.uniform(0.005, 0.05), rng.uniform(0.005, 0.05), rng.uniform(0, 6.28)
        img += 0.15 * np.sin(fx * xx + fy * yy + ph)[..., None]
    img -= img.min()
    img /= max(float(img.max()), 1e-6)
    img = (img * 200 + 25).astype(np.uint8)
    if not colourful:
        img = np.repeat(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., None], 3, axis=2)
    n_strokes = (h * w) // 6000
    for _ in range(n_strokes):
        colour = tuple(int(c) for c in rng.integers(0, 256, 3))
        if not colourful:
            g = int(rng.integers(0, 256))
            colour = (g, g, g)
        kind = rng.integers(0, 3)
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        if kind == 0:
            cv2.circle(img, p, int(rng.integers(3, 18)), colour, -1, cv2.LINE_AA)
        elif kind == 1:
            q = (p[0] + int(rng.integers(-40, 40)), p[1] + int(rng.integers(-40, 40)))
            cv2.line(img, p, q, colour, int(rng.integers(1, 4)), cv2.LINE_AA)
        else:
            q = (p[0] + int(rng.integers(4, 30)), p[1] + int(rng.integers(4, 30)))
            cv2.rectangle(img, p, q, colour, -1)
    return cv2.GaussianBlur(img, (3, 3), 0.8)


def aruco_dictionary(name: str):
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def diamond_image(px_per_m: float) -> np.ndarray:
    """A Diamond is a 3x3 ChArUco board whose 4 markers carry the Diamond IDs."""
    board = aruco.CharucoBoard((3, 3), DIAMOND_SQUARE_M, DIAMOND_MARKER_M,
                               aruco_dictionary("DICT_4X4_50"),
                               np.array(DIAMOND_IDS, dtype=np.int32))
    side = int(round(3 * DIAMOND_SQUARE_M * px_per_m))
    return board.generateImage((side, side), marginSize=0, borderBits=1)


def board_image(params: dict, px_per_m: float) -> np.ndarray:
    n_markers = (params["squares_x"] * params["squares_y"]) // 2
    ids = np.arange(params["ids_offset"], params["ids_offset"] + n_markers, dtype=np.int32)
    board = aruco.CharucoBoard((params["squares_x"], params["squares_y"]),
                               params["square_mm"] / 1000, params["marker_mm"] / 1000,
                               aruco_dictionary(params["dictionary"]), ids)
    w = int(round(params["squares_x"] * params["square_mm"] / 1000 * px_per_m))
    h = int(round(params["squares_y"] * params["square_mm"] / 1000 * px_per_m))
    return board.generateImage((w, h), marginSize=0, borderBits=1)


@dataclass
class Damage:
    kind: str          # scratch | stain | loss
    center_m: tuple    # plane coordinates (X, Y), metres
    size_mm: float     # largest extent, millimetres
    angle_deg: float = 0.0
    cls: str = ""

    def to_json(self) -> dict:
        return {"kind": self.kind, "center_m": list(self.center_m), "size_mm": self.size_mm,
                "angle_deg": self.angle_deg, "class": self.cls or self.kind}


@dataclass
class SynthScene:
    K: np.ndarray
    D: np.ndarray
    image_size: tuple
    texture: np.ndarray
    px_per_m: float
    plane_x0_m: float          # plane X of texture column 0
    plane_y0_m: float          # plane Y of texture row 0 (top edge, Y up)
    golden_distance_m: float
    painting_rect_m: tuple     # (X0, Y0, X1, Y1) with Y0 < Y1
    diamond_center_m: tuple
    board_center_m: tuple
    board_params: dict
    seed: int
    damages: list = field(default_factory=list)

    # -- frames -------------------------------------------------------------
    def T_gw(self) -> np.ndarray:
        """Golden camera: centred on the plane origin, looking along world -Z."""
        return make_T(np.diag([1.0, -1.0, -1.0]), [0.0, 0.0, self.golden_distance_m])

    def pose_from_motion(self, c_golden_m, rotvec_deg=(0.0, 0.0, 0.0)) -> np.ndarray:
        """T_cw of a camera whose centre is ``c_golden_m`` in the golden camera
        frame and whose relative rotation dR = Exp(rotvec)."""
        dR = rotvec_deg_to_R(rotvec_deg)
        dT = make_T(dR, -dR @ np.asarray(c_golden_m, dtype=np.float64))
        return dT @ self.T_gw()

    # -- texture <-> plane ---------------------------------------------------
    def texture_to_plane(self) -> np.ndarray:
        s = 1.0 / self.px_per_m
        return np.array([[s, 0, self.plane_x0_m], [0, -s, self.plane_y0_m], [0, 0, 1.0]])

    def plane_to_texture_px(self, xy_m) -> np.ndarray:
        xy = np.asarray(xy_m, dtype=np.float64).reshape(-1, 2)
        return np.c_[(xy[:, 0] - self.plane_x0_m) * self.px_per_m,
                     (self.plane_y0_m - xy[:, 1]) * self.px_per_m]

    def homography(self, T_cw) -> np.ndarray:
        """Image <- texture homography (Appendix A.5 with the plane Z = 0)."""
        R, t = T_cw[:3, :3], T_cw[:3, 3]
        return self.K @ np.c_[R[:, 0], R[:, 1], t] @ self.texture_to_plane()

    def project_plane(self, T_cw, xy_m) -> np.ndarray:
        xy = np.asarray(xy_m, dtype=np.float64).reshape(-1, 2)
        P = np.c_[xy, np.zeros(len(xy))]
        pts, _ = cv2.projectPoints(P, cv2.Rodrigues(T_cw[:3, :3])[0], T_cw[:3, 3], self.K, self.D)
        return pts.reshape(-1, 2)

    # -- rendering -------------------------------------------------------------
    def render(self, T_cw, *, noise_sigma: float = 0.0, gain: float = 1.0, offset: float = 0.0,
               rng: np.random.Generator | None = None, texture: np.ndarray | None = None,
               supersample: int = 1) -> np.ndarray:
        """``supersample`` > 1 renders at s x resolution and area-averages
        (models pixel integration, removes aliasing). Fixtures use 1."""
        tex = self.texture if texture is None else texture
        s = int(supersample)
        H = self.homography(T_cw)
        if s > 1:
            S = np.array([[s, 0, (s - 1) / 2.0], [0, s, (s - 1) / 2.0], [0, 0, 1.0]])
            big = cv2.warpPerspective(tex, S @ H, (self.image_size[0] * s, self.image_size[1] * s),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(40, 40, 40))
            img = cv2.resize(big, self.image_size, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.warpPerspective(tex, H, self.image_size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))
        if gain != 1.0 or offset != 0.0 or noise_sigma > 0:
            f = img.astype(np.float32) * gain + offset
            if noise_sigma > 0:
                rng = rng or np.random.default_rng(self.seed + 1)
                f += rng.normal(0.0, noise_sigma, f.shape).astype(np.float32)
            img = np.clip(np.rint(f), 0, 255).astype(np.uint8)
        return img

    # -- damage ------------------------------------------------------------------
    def damaged_texture(self, damages: list[Damage]) -> np.ndarray:
        tex = self.texture.copy()
        for dmg in damages:
            c = self.plane_to_texture_px([dmg.center_m])[0]
            L = dmg.size_mm / 1000 * self.px_per_m
            if dmg.kind == "scratch":
                a = np.deg2rad(dmg.angle_deg)
                d = np.array([np.cos(a), -np.sin(a)]) * L / 2
                p, q = np.rint(c - d).astype(int), np.rint(c + d).astype(int)
                thick = max(2, int(round(0.0008 * self.px_per_m)))
                cv2.line(tex, tuple(int(v) for v in p), tuple(int(v) for v in q), (235, 235, 235), thick, cv2.LINE_AA)
            elif dmg.kind == "stain":
                overlay = tex.copy()
                cv2.ellipse(overlay, tuple(int(v) for v in np.rint(c)), (int(L / 2), int(L / 3)),
                            dmg.angle_deg, 0, 360, (30, 50, 70), -1, cv2.LINE_AA)
                tex = cv2.addWeighted(overlay, 0.75, tex, 0.25, 0)
            elif dmg.kind == "loss":
                half = int(L / 2)
                cx, cy = (int(v) for v in np.rint(c))
                cv2.rectangle(tex, (cx - half, cy - half // 2), (cx + half, cy + half // 2), (245, 245, 240), -1)
            else:
                raise ValueError(f"Unknown damage kind {dmg.kind}")
        return tex

    def damage_bbox_image(self, T_cw, dmg: Damage, pad_mm: float = 1.0) -> list[float]:
        """Axis-aligned image bbox (xyxy) of a damage footprint seen from T_cw."""
        r = (dmg.size_mm / 2 + pad_mm) / 1000
        cx, cy = dmg.center_m
        corners = [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
        if dmg.kind == "scratch":
            a = np.deg2rad(dmg.angle_deg)
            d = np.array([np.cos(a), np.sin(a)]) * dmg.size_mm / 2000
            pad = pad_mm / 1000
            corners = [(cx - d[0] - pad, cy - d[1] - pad), (cx + d[0] + pad, cy + d[1] + pad),
                       (cx - d[0] + pad, cy - d[1] + pad), (cx + d[0] - pad, cy + d[1] - pad)]
        px = self.project_plane(T_cw, corners)
        return [float(px[:, 0].min()), float(px[:, 1].min()), float(px[:, 0].max()), float(px[:, 1].max())]

    def painting_polygon_image(self, T_cw) -> np.ndarray:
        x0, y0, x1, y1 = self.painting_rect_m
        return self.project_plane(T_cw, [(x0, y1), (x1, y1), (x1, y0), (x0, y0)])

    def board_polygon_image(self, T_cw, margin_mm: float = 5.0) -> np.ndarray:
        p = self.board_params
        w = p["squares_x"] * p["square_mm"] / 1000 / 2 + margin_mm / 1000
        h = p["squares_y"] * p["square_mm"] / 1000 / 2 + margin_mm / 1000
        cx, cy = self.board_center_m
        return self.project_plane(T_cw, [(cx - w, cy + h), (cx + w, cy + h), (cx + w, cy - h), (cx - w, cy - h)])

    def px_per_mm_golden(self) -> float:
        return float(self.K[0, 0] / self.golden_distance_m / 1000)


def _paste(tex: np.ndarray, patch_gray: np.ndarray, center_px, margin_px: int) -> None:
    h, w = patch_gray.shape[:2]
    cx, cy = (int(round(v)) for v in center_px)
    x0, y0 = cx - w // 2, cy - h // 2
    cv2.rectangle(tex, (x0 - margin_px, y0 - margin_px), (x0 + w + margin_px, y0 + h + margin_px),
                  (255, 255, 255), -1)
    tex[y0:y0 + h, x0:x0 + w] = cv2.cvtColor(patch_gray, cv2.COLOR_GRAY2BGR)


def make_scene(seed: int = 0, *, width: int = 960, px_per_mm: float = 3.0,
               distance_m: float = 0.65, with_board: bool = True, with_diamond: bool = True,
               board_params: dict | None = None) -> SynthScene:
    rng = np.random.default_rng(seed)
    height = int(round(width * REAL_SIZE[1] / REAL_SIZE[0]))
    K = scaled_intrinsics(width)
    ppm = px_per_mm * 1000.0
    plane_w, plane_h = 1.10, 0.75
    tw, th = int(plane_w * ppm), int(plane_h * ppm)
    tex = structured_texture(th, tw, rng, colourful=False)
    x0, y0 = -plane_w / 2, plane_h / 2
    painting = (-0.0985, -0.105, 0.1985, 0.105)
    scene = SynthScene(K=K, D=np.zeros(5), image_size=(width, height), texture=tex,
                       px_per_m=ppm, plane_x0_m=x0, plane_y0_m=y0, golden_distance_m=distance_m,
                       painting_rect_m=painting, diamond_center_m=(-0.24, 0.10),
                       board_center_m=(0.31, 0.0), board_params=dict(board_params or BOARD_DEFAULT),
                       seed=seed)
    p0 = scene.plane_to_texture_px([(painting[0], painting[3])])[0].astype(int)
    p1 = scene.plane_to_texture_px([(painting[2], painting[1])])[0].astype(int)
    art = structured_texture(p1[1] - p0[1], p1[0] - p0[0], rng, colourful=True)
    tex[p0[1]:p1[1], p0[0]:p1[0]] = art
    if with_diamond:
        _paste(tex, diamond_image(ppm), scene.plane_to_texture_px([scene.diamond_center_m])[0],
               int(0.012 * ppm))
    if with_board:
        _paste(tex, board_image(scene.board_params, ppm),
               scene.plane_to_texture_px([scene.board_center_m])[0], int(0.008 * ppm))
    scene.texture = tex
    return scene


# ---------------------------------------------------------------------------
# Session and change-pair writers (manifest schema cita.session.v1)
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_calibration(path: Path, scene: SynthScene) -> Path:
    fs = cv2.FileStorage(str(path), cv2.FileStorage_WRITE)
    fs.write("image_width", int(scene.image_size[0]))
    fs.write("image_height", int(scene.image_size[1]))
    fs.write("camera_matrix", scene.K)
    fs.write("distortion_coefficients", scene.D.reshape(1, -1))
    fs.write("lens_position", 1.5)
    fs.release()
    return path


def make_pose_session(root: Path, *, seed: int = 0, session_id: str = "E1_SYNTH",
                      offsets: list[dict] | None = None, frames_per_pose: int = 3,
                      noise_sigma: float = 1.0, width: int = 960, painting_id: str = "SYN001",
                      supersample: int = 3) -> Path:
    """E1-style session: golden + stereo right + a grid of perturbed views.

    Ground truth of every frame is written to ``truth.json`` next to the
    manifest (never read by the methods; tests compare against it).
    """
    scene = make_scene(seed, width=width)
    rng = np.random.default_rng(seed + 100)
    out = Path(root) / session_id
    out.mkdir(parents=True, exist_ok=True)
    calib = write_calibration(out / "calibration.yaml", scene)
    offsets = offsets if offsets is not None else [
        {"x_mm": dx, "z_mm": dz, "pan_deg": pan, "tilt_deg": 0.0}
        for dx, dz, pan in [(0, 0, 0), (-10, 0, 0), (10, 0, 0), (0, 20, 0), (20, -20, 2), (-5, 10, -2)]]
    frames, truth = [], {}

    def add(frame_id, T_cw, meta):
        img = scene.render(T_cw, noise_sigma=noise_sigma, rng=rng, supersample=supersample)
        f = out / f"{frame_id}.png"
        cv2.imwrite(str(f), img)
        frames.append({"frame_id": frame_id, "file": f.name, "sha256": _sha256(f), **meta})
        dT = T_cw @ np.linalg.inv(scene.T_gw())
        truth[frame_id] = {"c_m": (-dT[:3, :3].T @ dT[:3, 3]).tolist(),
                           "rotvec_deg": np.rad2deg(cv2.Rodrigues(dT[:3, :3])[0].ravel()).tolist()}

    base = {"target": None, "odometry": None, "offset_mm": None, "offset_deg": None, "stage": None,
            "settle_s": 0.0, "t_capture": None, "camera_metadata": {}}
    add("f00000_g_0", scene.T_gw(), {**base, "pose_index": -1, "rep": 0, "role": "golden"})
    add("f00001_g_right", scene.pose_from_motion([0.100, 0.0, 0.0]),
        {**base, "pose_index": -1, "rep": 0, "role": "stereo_right"})
    idx = 2
    for pose_index, off in enumerate(offsets):
        # Synthetic rig: slider X = camera x, slider Z = camera z (depth),
        # pan = rotation about camera y, tilt = about camera x.
        c = np.array([off["x_mm"], 0.0, off["z_mm"]]) / 1000.0
        rot = [off.get("tilt_deg", 0.0), off.get("pan_deg", 0.0), 0.0]
        for rep in range(frames_per_pose):
            fid = f"f{idx:05d}_p{pose_index:03d}_{rep}"
            add(fid, scene.pose_from_motion(c, rot),
                {**base, "pose_index": pose_index, "rep": rep, "target": dict(off), "role": "grid"})
            idx += 1
    manifest = {
        "schema": "cita.session.v1", "session_id": session_id, "experiment": "E1", "split": "na",
        "painting_id": painting_id, "lighting": "L0", "distance_cm": scene.golden_distance_m * 100,
        "device_id": "synthetic", "tool_version": "synth", "motion_profile": {},
        "camera": {"width": scene.image_size[0], "height": scene.image_size[1], "lens_position": 1.5,
                   "calibration_file": calib.name, "calibration_sha256": _sha256(calib)},
        "golden": {"frame_ids": ["f00000_g_0"], "stereo_right_frame_id": "f00001_g_right"},
        "synthetic": {"seed": seed, "board": scene.board_params},
        "frames": frames,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    return out


DEFAULT_DAMAGES = [
    Damage("scratch", (0.02, 0.03), 18.0, 25.0, "scratch"),
    Damage("stain", (0.12, -0.05), 9.0, 0.0, "stain"),
    Damage("loss", (-0.05, -0.06), 6.0, 0.0, "loss"),
]


def make_change_pairs(root: Path, *, seed: int = 0, split: str = "test", n_unchanged: int = 3,
                      n_changed: int = 3, session_id: str | None = None, width: int = 960,
                      damages: list[Damage] | None = None, supersample: int = 3) -> Path:
    """E3-style pairs (golden vs current) with GT boxes in golden image coords."""
    scene = make_scene(seed, width=width)
    rng = np.random.default_rng(seed + 7)
    session_id = session_id or f"E3_SYNTH_{split}"
    out = Path(root) / session_id
    out.mkdir(parents=True, exist_ok=True)
    calib = write_calibration(out / "calibration.yaml", scene)
    T_g = scene.T_gw()
    golden = scene.render(T_g, noise_sigma=1.0, rng=rng, supersample=supersample)
    cv2.imwrite(str(out / "golden.png"), golden)
    damages = damages if damages is not None else DEFAULT_DAMAGES
    pairs = []
    roi = scene.painting_polygon_image(T_g)
    for i in range(n_unchanged + n_changed):
        changed = i >= n_unchanged
        off = rng.uniform(-1, 1, 3) * np.array([0.004, 0.003, 0.004])
        rot = rng.uniform(-1, 1, 3) * np.array([0.4, 0.4, 0.2])
        T_c = scene.pose_from_motion(off, rot)
        tex = scene.damaged_texture(damages) if changed else None
        img = scene.render(T_c, noise_sigma=1.5, gain=float(rng.uniform(0.92, 1.08)),
                           offset=float(rng.uniform(-6, 6)), rng=rng, texture=tex, supersample=supersample)
        name = f"cur_{i:03d}.png"
        cv2.imwrite(str(out / name), img)
        pair_id = f"{session_id}_{i:03d}"
        gt = {"schema": "cita.change_gt.v1", "pair_id": pair_id, "split": split,
              "golden_image": "golden.png", "current_image": name, "stage": 1 if changed else 0,
              "changes": [{"class": d.cls or d.kind, "bbox_ref_xyxy": scene.damage_bbox_image(T_g, d),
                           "size_mm": d.size_mm, "log_ref": f"synthetic#{k}"}
                          for k, d in enumerate(damages)] if changed else [],
              "offset_mm": float(np.linalg.norm(off) * 1000),
              "offset_deg": float(np.linalg.norm(rot))}
        (out / f"gt_{i:03d}.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
        pairs.append({"pair_id": pair_id, "golden_image": "golden.png", "current_image": name,
                      "gt": f"gt_{i:03d}.json", "offset_mm": gt["offset_mm"], "offset_deg": gt["offset_deg"]})
    manifest = {"schema": "cita.pairs.v1", "session_id": session_id, "experiment": "E3", "split": split,
                "camera": {"calibration_file": calib.name, "calibration_sha256": _sha256(calib)},
                "roi": {"polygon_ref_px": roi.tolist(),
                        "size_mm": [297.0, 210.0], "source": "synthetic_projection"},
                "px_per_mm": scene.px_per_mm_golden(), "pairs": pairs}
    (out / "pairs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
