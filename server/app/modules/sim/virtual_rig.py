"""Virtual rig: slider X/Z + pan/tilt servos with hidden kinematic errors.

True state = pose of the camera relative to the golden camera. Parameters
unknown to every controller:

* slider axes rotated w.r.t. the painting/camera axes (``slider_misalign_deg``),
* steps/mm scale error per slider axis,
* pan/tilt rotation centre offset from the optical centre (``pivot_offset_mm``),
* servo dead-band, backlash (hysteresis), angle quantisation (legacy integer
  servo angles unless the move carries ``motion_profile.float_angles``),
* pose measurement noise (``noise_model``) in ``pose_noise`` mode.

Frames: golden camera frame G = OpenCV camera axes at the golden pose.
Slider X nominally = +x_G; slider Z nominally = +z_G (``z_axis='depth'``) or
-y_G (``'vertical'``). Pan rotates about the vertical axis (-y_G), tilt about
x_G, both through the pivot. ``dT`` (golden -> current camera) follows A.1.

Measurement modes: ``pose_noise`` (true pose + Gaussian noise, fast) and
``render`` (warp of a planar golden image with the plane-induced homography
A.5, then the real pose pipeline via a caller-supplied estimator).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from app.modules.artifact_pose.geometry import DEFAULT_QUALITY, camera_deviation
from app.modules.artifact_pose.motion import camera_center_in_golden, rotvec


def _rot(axis, deg) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    return cv2.Rodrigues((axis / np.linalg.norm(axis) * np.deg2rad(deg)).reshape(3, 1))[0]


@dataclass
class RigParams:
    slider_misalign_deg: float = 10.0       # rotation of both slider axes about the vertical axis
    slider_misalign_tilt_deg: float = 0.0   # about the camera x axis
    step_scale_error: tuple = (0.02, -0.015)  # fractional steps/mm error (x, z)
    z_axis: str = "depth"
    pivot_offset_mm: tuple = (0.0, 10.0, -30.0)   # servo rotation centre in golden camera frame
    pan_sign: int = 1
    tilt_sign: int = 1
    deadband_deg: float = 0.2
    backlash_deg: float = 0.4
    servo_scale_error: float = 0.0
    noise_c_mm: float = 0.3
    noise_rot_deg: float = 0.03
    steps_per_mm: float = 800.0
    golden_distance_m: float = 0.65

    @classmethod
    def ideal(cls) -> "RigParams":
        return cls(slider_misalign_deg=0.0, slider_misalign_tilt_deg=0.0, step_scale_error=(0.0, 0.0),
                   pivot_offset_mm=(0.0, 0.0, 0.0), deadband_deg=0.0, backlash_deg=0.0,
                   noise_c_mm=0.0, noise_rot_deg=0.0)


class _Servo:
    """Dead-band + backlash (play) + optional integer quantisation."""

    def __init__(self, deadband: float, backlash: float, scale_error: float) -> None:
        self.cmd = 0.0         # commanded angle (relative to home), as the device stores it
        self.actual = 0.0      # physical output angle
        self.deadband, self.backlash, self.scale = deadband, backlash, scale_error

    def command(self, target: float, float_angles: bool) -> None:
        target = float(target) if float_angles else float(int(round(target)))
        if abs(target - self.cmd) < 1e-12:
            return
        self.cmd = target
        desired = target * (1.0 + self.scale)
        delta = desired - self.actual
        if abs(delta) <= self.deadband:
            return
        # Backlash: the output lags by half the play in the direction of travel.
        self.actual = desired - np.sign(delta) * self.backlash / 2.0


class VirtualRig:
    def __init__(self, params: RigParams | None = None, *, seed: int = 0, T_gw: np.ndarray | None = None) -> None:
        self.p = params or RigParams()
        self.rng = np.random.default_rng(seed)
        self.T_gw = T_gw if T_gw is not None else np.r_[np.c_[np.diag([1.0, -1.0, -1.0]), [0, 0, self.p.golden_distance_m]],
                                                        [[0, 0, 0, 1]]]
        R_mis = _rot([0, 1, 0], self.p.slider_misalign_deg) @ _rot([1, 0, 0], self.p.slider_misalign_tilt_deg)
        z_nom = np.array([0, 0, 1.0]) if self.p.z_axis == "depth" else np.array([0, -1.0, 0])
        self.a_x, self.a_z = R_mis @ np.array([1.0, 0, 0]), R_mis @ z_nom
        self.pivot = np.asarray(self.p.pivot_offset_mm, dtype=np.float64) / 1000.0
        self.x_steps = 0
        self.z_steps = 0
        self.pan = _Servo(self.p.deadband_deg, self.p.backlash_deg, self.p.servo_scale_error)
        self.tilt = _Servo(self.p.deadband_deg, self.p.backlash_deg, self.p.servo_scale_error)
        self.n_moves = 0

    # -- kinematics --------------------------------------------------------------
    def camera_in_golden(self) -> tuple[np.ndarray, np.ndarray]:
        """(R_g<-c, C_g): camera orientation and centre in the golden frame."""
        x_m = self.x_steps / self.p.steps_per_mm / 1000.0 * (1 + self.p.step_scale_error[0])
        z_m = self.z_steps / self.p.steps_per_mm / 1000.0 * (1 + self.p.step_scale_error[1])
        slider = self.a_x * x_m + self.a_z * z_m
        R = _rot([0, -1, 0], self.p.pan_sign * self.pan.actual) @ _rot([1, 0, 0], self.p.tilt_sign * self.tilt.actual)
        C = slider + self.pivot - R @ self.pivot
        return R, C

    def true_dT(self) -> np.ndarray:
        R, C = self.camera_in_golden()
        dT = np.eye(4)
        dT[:3, :3] = R.T
        dT[:3, 3] = -R.T @ C
        return dT

    def true_errors(self) -> dict:
        dT = self.true_dT()
        c = camera_center_in_golden(dT) * 1000
        th = np.rad2deg(rotvec(dT[:3, :3]))
        return {"c_mm": c.tolist(), "c_norm_mm": float(np.linalg.norm(c)), "theta_deg": th.tolist(),
                "rot_deg": float(np.linalg.norm(th))}

    # -- actuation ---------------------------------------------------------------
    def move(self, payload: dict) -> dict:
        """Apply an MQTT ``move`` payload exactly like the device would."""
        prof = payload.get("motion_profile") or {}
        float_angles = bool(prof.get("float_angles", False))
        unidirectional = bool(prof.get("unidirectional", False))
        backlash_deg = float(prof.get("backlash_deg", 0.0))
        approach = 1 if int(prof.get("approach_sign", 1)) >= 0 else -1
        phase = str(payload.get("alignment_phase", "")).lower()
        if phase != "rotation":
            self.x_steps += int(payload.get("x_steps", 0)) * (1 if int(payload.get("x_dir", 1)) >= 0 else -1)
            self.z_steps += int(payload.get("z_steps", 0)) * (1 if int(payload.get("z_dir", 1)) >= 0 else -1)
        if phase != "translation":
            for servo, key in ((self.pan, "yaw_delta"), (self.tilt, "pitch_delta")):
                delta = float(payload.get(key, 0.0))
                if abs(delta) < 0.05 and phase == "rotation":
                    continue  # device skips sub-0.05 deg rotations
                target = servo.cmd + delta
                if unidirectional and backlash_deg > 0 and (target - servo.cmd) * approach < 0:
                    servo.command(target - approach * backlash_deg, float_angles)
                servo.command(target, float_angles)
        self.n_moves += 1
        return {"status": "ok", "commanded": {"yaw_deg": self.pan.cmd, "pitch_deg": self.tilt.cmd,
                                              "x_steps_odometry": self.x_steps, "z_steps_odometry": self.z_steps}}

    def perturb(self, *, x_mm: float = 0.0, z_mm: float = 0.0, pan_deg: float = 0.0, tilt_deg: float = 0.0,
                float_angles: bool = True) -> dict:
        return self.move({"x_steps": abs(int(round(x_mm * self.p.steps_per_mm))), "x_dir": 1 if x_mm >= 0 else -1,
                          "z_steps": abs(int(round(z_mm * self.p.steps_per_mm))), "z_dir": 1 if z_mm >= 0 else -1,
                          "yaw_delta": pan_deg, "pitch_delta": tilt_deg,
                          "motion_profile": {"float_angles": float_angles}})

    # -- measurement ---------------------------------------------------------------
    def measure(self) -> dict:
        """Noisy pose estimate (PoseEstimate-like JSON, incl. legacy deviation)."""
        dT = self.true_dT()
        c = camera_center_in_golden(dT) + self.rng.normal(0, self.p.noise_c_mm / 1000.0, 3)
        dth = np.deg2rad(self.rng.normal(0, self.p.noise_rot_deg, 3))
        R_noisy = cv2.Rodrigues(dth.reshape(3, 1))[0] @ dT[:3, :3]
        dT_n = np.eye(4)
        dT_n[:3, :3] = R_noisy
        dT_n[:3, 3] = -R_noisy @ c
        T_cw = dT_n @ self.T_gw
        rv_g = cv2.Rodrigues(self.T_gw[:3, :3])[0].ravel()
        rv_c = cv2.Rodrigues(T_cw[:3, :3])[0].ravel()
        dev = camera_deviation(rv_c, T_cw[:3, 3], rv_g, self.T_gw[:3, 3], DEFAULT_QUALITY)
        corr = np.asarray(dev["translation_correction_world_m"])
        dev.update(delta_x=float(corr[0]), delta_y=float(corr[1]), delta_z=float(corr[2]))
        sig_c = [max(self.p.noise_c_mm, 1e-3)] * 3
        sig_r = [max(self.p.noise_rot_deg, 1e-4)] * 3
        return {"ok": True, "status": "accepted", "method": "virtual", "camera_center_in_golden_m": c.tolist(),
                "rotvec_deg": np.rad2deg(rotvec(R_noisy)).tolist(), "sigma_center_mm": sig_c,
                "sigma_rot_deg": sig_r, "deviation": dev, "delta_R": R_noisy.tolist(),
                "delta_t": dT_n[:3, 3].tolist()}

    def render(self, golden_image: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Warp a planar golden image to the current true pose (A.5)."""
        dT = self.true_dT()
        n = self.T_gw[:3, :3] @ np.array([0.0, 0.0, 1.0])
        d = float(n @ self.T_gw[:3, 3])
        H = K @ (dT[:3, :3] + np.outer(dT[:3, 3], n) / d) @ np.linalg.inv(K)
        h, w = golden_image.shape[:2]
        return cv2.warpPerspective(golden_image, H, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40))


def nominal_fixed_signs(params: RigParams | None = None) -> dict:
    """SIGN_* that make the fixed mapping correct on the IDEAL version of the
    rig (same axis conventions, no misalignment/pivot/backlash). Used to give
    C-A/C-B a fair start on the virtual rig."""
    base = replace(params or RigParams(), slider_misalign_deg=0.0, slider_misalign_tilt_deg=0.0,
                   step_scale_error=(0.0, 0.0), pivot_offset_mm=(0.0, 0.0, 0.0), deadband_deg=0.0,
                   backlash_deg=0.0, noise_c_mm=0.0, noise_rot_deg=0.0, servo_scale_error=0.0)
    signs = {}
    for key, field_name, kw in (("x", "delta_x", {"x_mm": 1.0}), ("z", "delta_z", {"z_mm": 1.0}),
                                ("pan", "delta_pan", {"pan_deg": 1.0}), ("tilt", "delta_tilt", {"tilt_deg": 1.0})):
        rig = VirtualRig(base)
        rig.perturb(**kw)
        r = rig.measure()["deviation"][field_name]
        signs[key] = -1 if r > 0 else 1
    return signs


@dataclass
class VirtualPort:
    """RigPort over a VirtualRig (pose_noise mode, or render + estimator)."""
    rig: VirtualRig
    estimator: object = None            # callable(list[images]) -> PoseEstimate (render mode)
    golden_image: np.ndarray | None = None
    K: np.ndarray | None = None
    frames: int = 1
    log: list = field(default_factory=list)

    def move(self, payload: dict) -> dict:
        ack = self.rig.move(payload)
        self.log.append({"payload": payload, "ack": ack, "true": self.rig.true_errors()})
        return ack

    def observe(self):
        if self.estimator is None:
            return self.rig.measure()
        imgs = [self.rig.render(self.golden_image, self.K) for _ in range(self.frames)]
        return self.estimator(imgs)
