"""Shared detection, feature extraction and versioned reference storage."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
from datetime import datetime, timezone
import cv2
import numpy as np
from cv2 import aruco
from .geometry import (DEFAULT_QUALITY, camera_params, descriptors, points,
                       pose_is_ambiguous, square_pose_candidates, vector3)

BASE_DIR = Path(__file__).resolve().parents[3]
PARAMS_FILE = str(BASE_DIR / "data/camera_params.yaml")
DATA_DIR = str(BASE_DIR / "data")
VIS_DIR = str(BASE_DIR / "data/visualization")
GOLDEN_POSE_FILE = str(BASE_DIR / "data/golden_pose.yaml")
SQUARE_LENGTH, MARKER_LENGTH = 0.040, 0.025
DICT_ID = aruco.DICT_4X4_50
DIAMOND_IDS = (10, 20, 30, 40)
h = SQUARE_LENGTH / 2
DIAMOND_OBJ_PTS = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])
TRANS_TOLERANCE = DEFAULT_QUALITY.translation_tolerance_m
ROT_TOLERANCE = DEFAULT_QUALITY.rotation_tolerance_deg
sys.path.insert(0, os.environ.get("POSE_CPP_PATH", str(BASE_DIR / "native/pose_solver_cpp")))
try:
    pose_solver_cpp = importlib.import_module("pose_solver_cpp")
    if getattr(pose_solver_cpp, "API_VERSION", None) != 2:
        raise ImportError("Legacy extension; rebuild native API v2")
    HAS_CPP, CPP_IMPORT_ERROR = True, None
except ImportError as exc:
    pose_solver_cpp, HAS_CPP, CPP_IMPORT_ERROR = None, False, str(exc)


def resolve_backend(backend):
    if backend not in ("auto", "python", "g2o"):
        raise ValueError("Unknown backend")
    if backend == "g2o" and not HAS_CPP:
        raise ValueError(f"Native v2 backend unavailable: {CPP_IMPORT_ERROR}")
    return pose_solver_cpp if HAS_CPP and backend != "python" else None


def read_calibration(filepath=PARAMS_FILE):
    fs = cv2.FileStorage(str(filepath), cv2.FileStorage_READ)
    if not fs.isOpened():
        raise ValueError(f"Cannot open calibration: {filepath}")
    try:
        K, D = camera_params(fs.getNode("camera_matrix").mat(),
                             fs.getNode("distortion_coefficients").mat())
        size = tuple(int(fs.getNode(name).real()) for name in ("image_width", "image_height"))
        if min(size) <= 0:
            raise ValueError("Calibration must declare image_width and image_height")
        return {"K": K, "D": D, "image_size": size}
    finally:
        fs.release()


def load_camera_params(filepath=None):
    calibration = read_calibration(filepath or PARAMS_FILE)
    return calibration["K"], calibration["D"]


def load_camera_lens_position(filepath=None):
    """Read optional lens metadata without making calibration loading lenient."""
    path = Path(filepath or PARAMS_FILE)
    fs = cv2.FileStorage(str(path), cv2.FileStorage_READ)
    if not fs.isOpened():
        return None
    try:
        node = fs.getNode("lens_position")
        return None if node.empty() else float(node.real())
    finally:
        fs.release()


def validate_image(image, image_size=None):
    if (image is None or image.dtype != np.uint8 or image.size == 0 or
            image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3)):
        raise ValueError("Image must be nonempty uint8 grayscale or BGR")
    if image_size is not None and (image.shape[1], image.shape[0]) != tuple(image_size):
        raise ValueError(f"Image size {(image.shape[1], image.shape[0])} differs from calibration {image_size}; recalibrate or explicitly rescale intrinsics")
    return np.ascontiguousarray(image)


def detect_diamond(image, K, D, quality=DEFAULT_QUALITY, expected_ids=DIAMOND_IDS):
    image = validate_image(image)
    K, D = camera_params(K, D)
    dictionary = aruco.getPredefinedDictionary(DICT_ID)
    board = aruco.CharucoBoard((3, 3), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    params.adaptiveThreshWinSizeMax = 53
    cp = aruco.CharucoParameters()
    cp.cameraMatrix, cp.distCoeffs = K, D
    detector = aruco.CharucoDetector(board, cp, params)
    corners, ids, _, _ = detector.detectDiamonds(image)
    if ids is None:
        return None
    indices = [i for i, found in enumerate(np.asarray(ids).reshape(-1, 4))
               if tuple(found) == tuple(expected_ids)]
    if len(indices) != 1:
        return None
    xy = np.asarray(corners[indices[0]], dtype=np.float64).reshape(4, 2)
    candidates = square_pose_candidates(DIAMOND_OBJ_PTS, xy, K, D, quality)
    if not candidates or candidates[0]["rms_px"] > quality.max_diamond_rms_px:
        return None
    return {**candidates[0], "corners": xy, "obj_pts": DIAMOND_OBJ_PTS.copy(),
            "ids": list(expected_ids), "candidates": candidates,
            "ambiguous": pose_is_ambiguous(candidates, quality)}


def extract_orb(image, max_features=5000, min_node_size=64, max_depth=7, backend="auto", features_per_leaf=4):
    image = validate_image(image)
    native = resolve_backend(backend)
    if min(max_features, min_node_size, max_depth, features_per_leaf) <= 0 or max_depth > 16:
        raise ValueError("Invalid ORB/QuadTree limits")
    if native is not None:
        out = native.extract_with_quadtree(image, max_features, min_node_size, max_depth, features_per_leaf)
        kp = [cv2.KeyPoint(x=k["x"], y=k["y"], size=k["size"], angle=k["angle"],
                          response=k["response"], octave=k["octave"]) for k in out["keypoints"]]
        desc = descriptors(out["descriptors"])
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        orb = cv2.ORB_create(nfeatures=max_features, WTA_K=2)
        detected = list(orb.detect(gray, None))
        def split(items, x, y, w, height, depth):
            if not items:
                return []
            if w // 2 < min_node_size or height // 2 < min_node_size or depth >= max_depth:
                return sorted(items, key=lambda k: -k.response)[:features_per_leaf]
            hw, hh = w // 2, height // 2
            groups = [[], [], [], []]
            for k in items:
                groups[(2 if k.pt[1] >= y + hh else 0) + (1 if k.pt[0] >= x + hw else 0)].append(k)
            output = []
            for i, group in enumerate(groups):
                if len(group) == 1:
                    output.extend(group)
                else:
                    output.extend(split(group, x + (hw if i % 2 else 0), y + (hh if i >= 2 else 0),
                                        w - hw if i % 2 else hw, height - hh if i >= 2 else hh, depth + 1))
            return output
        selected = split(detected, 0, 0, gray.shape[1], gray.shape[0], 0)
        kp, desc = orb.compute(gray, selected)
        kp = list(kp or [])
        desc = np.empty((0, 32), np.uint8) if desc is None else descriptors(desc)
    xy = np.array([k.pt for k in kp], dtype=np.float64).reshape(-1, 2)
    if len(xy) != len(desc):
        raise ValueError("ORB keypoint/descriptor alignment violated")
    return kp, desc, xy


def calibration_fingerprint(K, D, image_size):
    K, D = camera_params(K, D)
    return hashlib.sha256(np.r_[K.ravel(), D.ravel(), image_size].astype("<f8").tobytes()).hexdigest()


def save_golden_pose(filepath, diamond_result, points_3d, points_2d, desc,
                     image_size, baseline=None, *, K, D, strategy="hybrid", stereo=None,
                     min_orb_inliers=DEFAULT_QUALITY.min_orb_inliers):
    """Content-addressed descriptors first; atomic YAML commit last.

    A crash may leave an unused descriptor file, never an undetected mixed pair.
    """
    xyz, xy, desc = points(points_3d, 3), points(points_2d, 2), descriptors(desc)
    if len(xyz) != len(xy) or len(xyz) != len(desc):
        raise ValueError("Reference arrays must have identical row counts")
    if type(min_orb_inliers) is not int or min_orb_inliers < 6:
        raise ValueError("Reference minimum must be an integer of at least six")
    if strategy not in ("diamond", "hybrid") or (strategy == "hybrid" and len(xyz) < min_orb_inliers):
        raise ValueError("Invalid strategy or insufficient hybrid landmarks")
    if diamond_result.get("ambiguous", True):
        raise ValueError("Cannot save an ambiguous reference pose")
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(desc.tobytes()).hexdigest()
    desc_path = filepath.with_name(f"{filepath.stem}_descriptors.{digest[:16]}.npy")
    meta = {"schema_version": 2, "frame": "diamond_world_to_camera", "units": "metres",
            "image_size": list(image_size), "diamond_ids": list(DIAMOND_IDS),
            "square_length_m": SQUARE_LENGTH, "marker_length_m": MARKER_LENGTH,
            "dictionary": int(DICT_ID), "strategy": strategy,
            "min_orb_inliers": min_orb_inliers,
            "calibration_sha256": calibration_fingerprint(K, D, image_size),
            "descriptor_file": desc_path.name, "descriptor_sha256": digest,
            "stereo": stereo, "timestamp": datetime.now(timezone.utc).isoformat()}
    with tempfile.NamedTemporaryFile(dir=filepath.parent, suffix=".npy", delete=False) as f:
        desc_tmp = Path(f.name)
        np.save(f, desc, allow_pickle=False)
    try:
        os.replace(desc_tmp, desc_path)
    finally:
        desc_tmp.unlink(missing_ok=True)
    fd, name = tempfile.mkstemp(dir=filepath.parent, suffix=".yaml")
    os.close(fd)
    temp = Path(name)
    try:
        fs = cv2.FileStorage(str(temp), cv2.FileStorage_WRITE)
        if not fs.isOpened():
            raise OSError(f"Cannot write {temp}")
        try:
            fs.write("metadata_json", json.dumps(meta, ensure_ascii=True, allow_nan=False))
            fs.write("rvec_diamond", vector3(diamond_result["rvec"]).reshape(3, 1))
            fs.write("tvec_diamond", vector3(diamond_result["tvec"]).reshape(3, 1))
            fs.write("corners_diamond", points(diamond_result["corners"], 2))
            fs.write("points_3d", xyz)
            fs.write("points_2d", xy)
        finally:
            fs.release()
        os.replace(temp, filepath)
    finally:
        temp.unlink(missing_ok=True)


def load_golden_pose(filepath=None, *, K=None, D=None, image_size=None):
    filepath = Path(filepath or GOLDEN_POSE_FILE)
    fs = cv2.FileStorage(str(filepath), cv2.FileStorage_READ)
    if not fs.isOpened():
        raise ValueError(f"Cannot open reference: {filepath}")
    try:
        node = fs.getNode("metadata_json")
        if node.empty():
            raise ValueError("Legacy golden schema unsupported; regenerate golden with v2")
        meta = json.loads(node.string())
        if not isinstance(meta, dict):
            raise ValueError("Golden metadata must be an object")
        if (meta.get("schema_version") != 2 or meta.get("frame") != "diamond_world_to_camera" or
                meta.get("units") != "metres" or meta.get("diamond_ids") != list(DIAMOND_IDS) or
                meta.get("square_length_m") != SQUARE_LENGTH or meta.get("marker_length_m") != MARKER_LENGTH or
                meta.get("dictionary") != int(DICT_ID) or meta.get("strategy") not in ("diamond", "hybrid")):
            raise ValueError("Incompatible golden schema, board or coordinate frame")
        r, t = vector3(fs.getNode("rvec_diamond").mat()), vector3(fs.getNode("tvec_diamond").mat())
        xyz, xy = fs.getNode("points_3d").mat(), fs.getNode("points_2d").mat()
        xyz = np.empty((0, 3)) if xyz is None else points(xyz, 3)
        xy = np.empty((0, 2)) if xy is None else points(xy, 2)
    finally:
        fs.release()
    desc_name = meta["descriptor_file"]
    if Path(desc_name).name != desc_name or "/" in desc_name or "\\" in desc_name:
        raise ValueError("Descriptor path must be a sibling filename")
    desc = descriptors(np.load(filepath.parent / desc_name, allow_pickle=False))
    if hashlib.sha256(desc.tobytes()).hexdigest() != meta["descriptor_sha256"]:
        raise ValueError("Descriptor checksum mismatch")
    if len(xyz) != len(xy) or len(xyz) != len(desc):
        raise ValueError("Corrupted golden: 3D/2D/descriptor counts differ")
    minimum = meta.get("min_orb_inliers", 6)
    if type(minimum) is not int or minimum < 6:
        raise ValueError("Corrupted golden: invalid minimum landmark count")
    if meta["strategy"] == "hybrid" and len(xyz) < minimum:
        raise ValueError("Corrupted golden: insufficient landmarks")
    if K is not None or D is not None or image_size is not None:
        if K is None or D is None or image_size is None:
            raise ValueError("Provide K, D and image_size together")
        if calibration_fingerprint(K, D, image_size) != meta["calibration_sha256"]:
            raise ValueError("Golden calibration/resolution mismatch; regenerate reference")
    return {"rvec": r, "tvec": t, "points_3d": xyz, "points_2d": xy,
            "descriptors": desc, "metadata": meta}
