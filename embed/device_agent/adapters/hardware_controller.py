"""Dieu khien Pan-Tilt (PCA9685) va Slider (Stepper qua GPIO)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock
from typing import Any, Optional

try:  # package import (embed/tools, tests) or flat import (runtime/main_app.py)
    from .motion_profile import MotionProfile, unidirectional_waypoints
except ImportError:  # pragma: no cover
    from adapters.motion_profile import MotionProfile, unidirectional_waypoints


@dataclass
class ServoConfig:
    """Cau hinh servo cho PCA9685."""

    pan_channel: int = 0
    tilt_channel: int = 1
    pan_min: int = 0
    pan_max: int = 180
    tilt_min: int = 0
    tilt_max: int = 120


@dataclass
class SliderConfig:
    """Cau hinh stepper slider."""

    x_pul_pin: int = 17
    x_dir_pin: int = 27
    z_pul_pin: int = 22
    z_dir_pin: int = 23
    microstep: int = 32
    pulse_delay: float = 0.0005


class HardwareController:
    """Controller tong hop cho servo pan-tilt va slider X/Z."""

    # Vi tri home mac dinh cua servo (do). Dung lam diem chuan truoc khi chup anh.
    HOME_YAW: int = 110
    HOME_PITCH: int = 35

    def __init__(
        self,
        servo_config: Optional[ServoConfig] = None,
        slider_config: Optional[SliderConfig] = None,
        *,
        kit: Any = None,
        gpio: Any = None,
        sleep=time.sleep,
    ) -> None:
        """``kit``/``gpio`` inject drivers (tests, dry-run); default = real hardware."""
        self.servo_config = servo_config or ServoConfig()
        self.slider_config = slider_config or SliderConfig()

        # Float typed (I2); in legacy mode the stored values stay integral.
        self.current_yaw: float = float(self.HOME_YAW)
        self.current_pitch: float = float(self.HOME_PITCH)
        self.current_x_steps: int = 0
        self.current_z_steps: int = 0
        self._stop_event = Event()
        self._state_lock = Lock()
        self._sleep = sleep
        # History of physical servo commands (for logs / tests), bounded.
        self.servo_command_log: list[tuple[float, float]] = []

        self._kit = kit
        self._gpio = gpio

        if self._kit is None:
            self._init_servo_driver()
        if self._gpio is None:
            self._init_stepper_driver()
        else:
            self._gpio.setmode(self._gpio.BCM)
            self._gpio.setup(
                [
                    self.slider_config.x_pul_pin,
                    self.slider_config.x_dir_pin,
                    self.slider_config.z_pul_pin,
                    self.slider_config.z_dir_pin,
                ],
                self._gpio.OUT,
            )

        # Gui lenh vat ly ngay khi khoi dong de dong bo trang thai RAM voi phan cung.
        # Servo co the o bat ky vi tri nao sau khi Pi restart.
        print(f"[HW] Khoi dong: dua servo ve home (Yaw={self.HOME_YAW}° Pitch={self.HOME_PITCH}°)")
        self.go_home()

    def _init_servo_driver(self) -> None:
        """Khoi tao PCA9685 qua ServoKit."""
        try:
            from adafruit_servokit import ServoKit

            self._kit = ServoKit(channels=16)
        except Exception as exc:
            print(f"[HW] Loi ket noi PCA9685/I2C: {exc}")
            raise

    def _init_stepper_driver(self) -> None:
        """Khoi tao GPIO cho stepper slider."""
        try:
            import RPi.GPIO as GPIO

            self._gpio = GPIO
            self._gpio.setmode(GPIO.BCM)
            self._gpio.setup(
                [
                    self.slider_config.x_pul_pin,
                    self.slider_config.x_dir_pin,
                    self.slider_config.z_pul_pin,
                    self.slider_config.z_dir_pin,
                ],
                GPIO.OUT,
            )
        except Exception as exc:
            print(f"[HW] Loi khoi tao GPIO stepper: {exc}")
            raise

    def set_pan_tilt(self, yaw_deg: float, pitch_deg: float, *, float_angles: bool = False) -> None:
        """Dat truc tiep goc Yaw/Pitch nhan duoc tu lenh.

        ``float_angles=False`` (mac dinh, hanh vi cu): lam tron ve so nguyen.
        ``float_angles=True`` (I2): giu goc so thuc, PCA9685 nhan float.
        """
        if self._kit is None:
            raise RuntimeError("PCA9685 chua duoc khoi tao")

        try:
            if float_angles:
                requested_pan, requested_tilt = float(yaw_deg), float(pitch_deg)
            else:
                requested_pan, requested_tilt = int(round(yaw_deg)), int(round(pitch_deg))
            pan_target = max(self.servo_config.pan_min, min(self.servo_config.pan_max, requested_pan))
            tilt_target = max(self.servo_config.tilt_min, min(self.servo_config.tilt_max, requested_tilt))

            clamp_note = ""
            if pan_target != requested_pan:
                clamp_note += f" [Pan clamp: yeu cau {requested_pan:.2f}°]"
            if tilt_target != requested_tilt:
                clamp_note += f" [Tilt clamp: yeu cau {requested_tilt:.2f}°]"

            print(
                f"[SERVO] Pan : {self.current_yaw:6.2f}° -> {pan_target:6.2f}°  |"
                f"  Tilt: {self.current_pitch:6.2f}° -> {tilt_target:6.2f}°"
                + clamp_note
            )

            self._kit.servo[self.servo_config.pan_channel].angle = pan_target
            self._kit.servo[self.servo_config.tilt_channel].angle = tilt_target
            with self._state_lock:
                self.current_yaw = float(pan_target)
                self.current_pitch = float(tilt_target)
                self.servo_command_log.append((float(pan_target), float(tilt_target)))
                del self.servo_command_log[:-256]
        except Exception as exc:
            raise RuntimeError(f"Loi dieu khien servo: {exc}") from exc

    def approach_pan_tilt(self, yaw_deg: float, pitch_deg: float, profile: Optional[MotionProfile] = None) -> list:
        """I2: set servos, approaching each target from the same side.

        With the legacy profile this is exactly ``set_pan_tilt`` (one command).
        Returns the list of (yaw, pitch) waypoints commanded.
        """
        profile = (profile or MotionProfile()).sanitized()
        yaw_wp = unidirectional_waypoints(
            self.current_yaw, float(yaw_deg), enabled=profile.unidirectional,
            approach_sign=profile.approach_sign, backlash=profile.backlash_deg,
            lower=self.servo_config.pan_min, upper=self.servo_config.pan_max)
        pitch_wp = unidirectional_waypoints(
            self.current_pitch, float(pitch_deg), enabled=profile.unidirectional,
            approach_sign=profile.approach_sign, backlash=profile.backlash_deg,
            lower=self.servo_config.tilt_min, upper=self.servo_config.tilt_max)
        waypoints = []
        if len(yaw_wp) > 1 or len(pitch_wp) > 1:
            waypoints.append((yaw_wp[0], pitch_wp[0]))
        waypoints.append((float(yaw_deg), float(pitch_deg)))
        for i, (y, p) in enumerate(waypoints):
            self.set_pan_tilt(y, p, float_angles=profile.float_angles)
            if i + 1 < len(waypoints):
                # Short pause so the overshoot is actually reached.
                self._sleep(max(0.15, min(0.5, profile.settle_s)))
        return waypoints

    def release_servos(self) -> None:
        """Ngat PWM (angle=None): servo khong giu momen, het rung/run."""
        if self._kit is None:
            return
        self._kit.servo[self.servo_config.pan_channel].angle = None
        self._kit.servo[self.servo_config.tilt_channel].angle = None
        print("[SERVO] Da ngat PWM (servo_release_after_move)")

    def move_slider_x(self, steps: int, direction: int) -> None:
        """Tinh tien slider truc X voi so buoc cho truoc."""
        self._run_stepper_axis(
            steps=steps,
            direction=direction,
            pul_pin=self.slider_config.x_pul_pin,
            dir_pin=self.slider_config.x_dir_pin,
            pulse_delay=None,
        )
        with self._state_lock:
            self.current_x_steps += steps if direction > 0 else -steps

    def move_slider_z(self, steps: int, direction: int) -> None:
        """Tinh tien slider truc Z (ho tro bo sung theo context du an)."""
        self._run_stepper_axis(
            steps=steps,
            direction=direction,
            pul_pin=self.slider_config.z_pul_pin,
            dir_pin=self.slider_config.z_dir_pin,
            pulse_delay=None,
        )
        with self._state_lock:
            self.current_z_steps += steps if direction > 0 else -steps

    def move_slider_x_with_delay(self, steps: int, direction: int, pulse_delay: float) -> None:
        """Tinh tien slider truc X voi pulse_delay override cho profile toc do."""
        self._run_stepper_axis(
            steps=steps,
            direction=direction,
            pul_pin=self.slider_config.x_pul_pin,
            dir_pin=self.slider_config.x_dir_pin,
            pulse_delay=pulse_delay,
        )
        with self._state_lock:
            self.current_x_steps += steps if direction > 0 else -steps

    def move_slider_z_with_delay(self, steps: int, direction: int, pulse_delay: float) -> None:
        """Tinh tien slider truc Z voi pulse_delay override cho profile toc do."""
        self._run_stepper_axis(
            steps=steps,
            direction=direction,
            pul_pin=self.slider_config.z_pul_pin,
            dir_pin=self.slider_config.z_dir_pin,
            pulse_delay=pulse_delay,
        )
        with self._state_lock:
            self.current_z_steps += steps if direction > 0 else -steps

    def emergency_stop(self) -> None:
        """Stop the active pulse loops as soon as possible and de-energize pins."""
        self._stop_event.set()
        if self._gpio is not None:
            for pin in (
                self.slider_config.x_pul_pin,
                self.slider_config.z_pul_pin,
            ):
                try:
                    self._gpio.output(pin, self._gpio.LOW)
                except Exception:
                    pass

    def clear_emergency_stop(self) -> None:
        self._stop_event.clear()

    def go_home(self) -> None:
        """Dua servo ve vi tri home (HOME_YAW, HOME_PITCH). Khong anh huong slider."""
        self.set_pan_tilt(float(self.HOME_YAW), float(self.HOME_PITCH))

    def reset_position(self) -> None:
        """Dua servo ve home va slider ve vi tri luc khoi dong."""
        print(f"[HW] reset_position: servo -> home ({self.HOME_YAW}°, {self.HOME_PITCH}°), slider -> vi tri khoi dong")
        self.clear_emergency_stop()
        self.go_home()

        if self.current_x_steps != 0:
            direction_x = -1 if self.current_x_steps > 0 else 1
            self.move_slider_x(abs(self.current_x_steps), direction_x)

        if self.current_z_steps != 0:
            direction_z = -1 if self.current_z_steps > 0 else 1
            self.move_slider_z(abs(self.current_z_steps), direction_z)

    def cleanup(self) -> None:
        """Giai phong tai nguyen phan cung."""
        self.emergency_stop()
        if self._gpio is not None:
            self._gpio.cleanup()

    def _run_stepper_axis(
        self,
        steps: int,
        direction: int,
        pul_pin: int,
        dir_pin: int,
        pulse_delay: Optional[float],
    ) -> None:
        """Phat xung stepper cho 1 truc."""
        if steps <= 0:
            return

        effective_pulse_delay = (
            pulse_delay if pulse_delay is not None else self.slider_config.pulse_delay
        )
        if effective_pulse_delay <= 0:
            effective_pulse_delay = self.slider_config.pulse_delay

        if self._gpio is None:
            raise RuntimeError("GPIO chua duoc khoi tao")

        gpio_direction = self._gpio.HIGH if direction > 0 else self._gpio.LOW
        self._gpio.output(dir_pin, gpio_direction)

        for _ in range(steps):
            if self._stop_event.is_set():
                raise RuntimeError("Emergency stop active")
            self._gpio.output(pul_pin, self._gpio.HIGH)
            self._sleep(effective_pulse_delay)
            self._gpio.output(pul_pin, self._gpio.LOW)
            self._sleep(effective_pulse_delay)
