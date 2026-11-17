"""Reference ChArUco board: independent pose ground truth.

The board is glued on the painting plane (or the formex) and never used by any
pose method: ``board_mask`` removes its image region from feature extraction
of every method, so the GT stays independent.

Board IDs must not collide with the Diamond (IDs 10/20/30/40, DICT_4X4_50).
Using a different family (default DICT_5X5_100) avoids collisions entirely.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from cv2 import aruco

from app.modules.artifact_pose.motion import as_T, relative_motion

DIAMOND_IDS = (10, 20, 30, 40)


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 5
    squares_y: int = 7
    square_mm: float = 30.0
    marker_mm: float = 22.0
    dictionary: str = "DICT_5X5_100"
    ids_offset: int = 0
    min_corners: int = 12
    mask_dilate_px: int = 25

    @classmethod
    def from_profile(cls, section) -> "BoardSpec":
        return cls(**{k: getattr(section, k) for k in cls.__dataclass_fields__})

    @property
    def n_markers(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    @property
    def ids(self) -> np.ndarray:
        return np.arange(self.ids_offset, self.ids_offset + self.n_markers, dtype=np.int32)

    def validate(self) -> None:
        if not hasattr(aruco, self.dictionary):
            raise ValueError(f"Unknown ArUco dictionary {self.dictionary}")
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, self.dictionary))
        size = dictionary.bytesList.shape[0]
        if self.ids[-1] >= size:
            raise ValueError(f"Board needs IDs up to {self.ids[-1]} but {self.dictionary} has {size}")
        if self.dictionary.startswith("DICT_4X4") and set(self.ids.tolist()) & set(DIAMOND_IDS):
            raise ValueError("Reference board IDs overlap the Diamond IDs 10/20/30/40 in the 4x4 family")

    def board(self):
        self.validate()
        dictionary = aruco.getPredefinedDictionary(getattr(aruco, self.dictionary))
        return aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_mm / 1000.0,
                                  self.marker_mm / 1000.0, dictionary, self.ids)

    def outline_m(self) -> np.ndarray:
        """Board corners in board coordinates (metres, z = 0)."""
        w, h = self.squares_x * self.square_mm / 1000.0, self.squares_y * self.square_mm / 1000.0
        return np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], dtype=np.float64)


def _detect_board(image: np.ndarray, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        ch_corners, ch_ids, m_corners, m_ids = detector.detectBoard(gray)
    else:  # pragma: no cover - OpenCV < 4.7
        dictionary = board.getDictionary()
        m_corners, m_ids, _ = aruco.detectMarkers(gray, dictionary)
        ch_corners = ch_ids = None
        if m_ids is not None and len(m_ids):
            _, ch_corners, ch_ids = aruco.interpolateCornersCharuco(m_corners, m_ids, gray, board)
    return ch_corners, ch_ids, m_corners, m_ids


def detect_reference_board(image: np.ndarray, K, D, spec: BoardSpec | None = None) -> dict:
    """Pose of the reference board (camera <- board).

    Returns ``{ok, rvec, tvec, T_cb, n_corners, reproj_rms_px, hull_polygon_px, reason}``.
    ``solvePnP`` (IPPE when planar) is followed by Levenberg-Marquardt refinement.
    """
    spec = spec or BoardSpec()
    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64).reshape(-1)
    board = spec.board()
    ch_corners, ch_ids, m_corners, m_ids = _detect_board(image, board)
    out = {"ok": False, "rvec": None, "tvec": None, "T_cb": None, "n_corners": 0,
           "n_markers": 0 if m_ids is None else int(len(m_ids)),
           "reproj_rms_px": None, "hull_polygon_px": None, "reason": None}
    if ch_ids is None or ch_corners is None or len(ch_ids) == 0:
        out["reason"] = "no_charuco_corners"
        return out
    out["n_corners"] = int(len(ch_ids))
    img_pts = np.asarray(ch_corners, dtype=np.float64).reshape(-1, 2)
    hull = cv2.convexHull(img_pts.astype(np.float32)).reshape(-1, 2)
    out["hull_polygon_px"] = hull.tolist()
    if len(ch_ids) < spec.min_corners:
        out["reason"] = f"too_few_corners<{spec.min_corners}"
        return out
    obj_all = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    obj_pts = obj_all[np.asarray(ch_ids).reshape(-1)]
    flag = cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flag)
    if not ok:
        out["reason"] = "solvepnp_failed"
        return out
    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, D, rvec, tvec)
    proj = cv2.projectPoints(obj_pts, rvec, tvec, K, D)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1))))
    depth = (cv2.Rodrigues(rvec)[0] @ obj_pts.T + tvec.reshape(3, 1))[2]
    if not np.isfinite(rms) or np.any(depth <= 0):
        out["reason"] = "invalid_pose"
        return out
    outline = cv2.projectPoints(spec.outline_m(), rvec, tvec, K, D)[0].reshape(-1, 2)
    out.update(ok=True, rvec=rvec.reshape(3).tolist(), tvec=tvec.reshape(3).tolist(),
               T_cb=as_T(rvec, tvec).tolist(), reproj_rms_px=rms,
               hull_polygon_px=outline.tolist(), outline_polygon_px=outline.tolist(),
               corner_hull_px=hull.tolist(), reason=None)
    return out


def board_mask(shape, hull_polygon_px, dilate_px: int = 25) -> np.ndarray:
    """uint8 mask: 255 = usable for features, 0 = (dilated) board region."""
    h, w = shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    if hull_polygon_px is None or len(hull_polygon_px) < 3:
        return mask
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [np.rint(np.asarray(hull_polygon_px, dtype=np.float64)).astype(np.int32)], 255)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        region = cv2.dilate(region, k)
    mask[region > 0] = 0
    return mask


def relative_motion_gt(board_pose_golden: dict | np.ndarray, board_pose_current: dict | np.ndarray) -> np.ndarray:
    """A.1 with the board pose: dT_gt = T_cb(current) @ inv(T_cb(golden))."""
    def T(x):
        if isinstance(x, dict):
            return np.asarray(x["T_cb"], dtype=np.float64)
        return np.asarray(x, dtype=np.float64)
    return relative_motion(T(board_pose_golden), T(board_pose_current))


def board_mask_for_image(image: np.ndarray, K, D, spec: BoardSpec | None = None) -> tuple[np.ndarray, dict]:
    spec = spec or BoardSpec()
    det = detect_reference_board(image, K, D, spec)
    return board_mask(image.shape, det.get("hull_polygon_px"), spec.mask_dilate_px), det
