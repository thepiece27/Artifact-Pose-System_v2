from __future__ import annotations

import enum
import secrets
from datetime import datetime, timezone
from typing import List

from sqlalchemy import (
    Column, String, Text, DateTime, ForeignKey, 
    Boolean, Enum, Float, Integer
)
from sqlalchemy.orm import relationship

from app.core.database import Base

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

def _gen_id() -> str:
    """Generate a unique 6-character hex ID."""
    return secrets.token_hex(3)

class ImageType(str, enum.Enum):
    baseline = "baseline"
    inspection = "inspection"

class AlertLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"

class ComparisonStatus(str, enum.Enum):
    good = "good"
    damaged = "damaged"
    warning = "warning"

class InspectionType(str, enum.Enum):
    scheduled = "scheduled"
    sudden = "sudden"

class Artifact(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(String(6), primary_key=True, index=True, default=_gen_id)
    name = Column(String(255), nullable=False, index=True)
    location = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="good")
    
    # Recurring inspection interval in days. 0 means one-time/manual.
    inspection_interval_days = Column(Integer, nullable=False, default=0)
    
    baseline_image_id = Column(
        String(6), 
        ForeignKey("images.image_id", use_alter=True, name="fk_artifact_baseline_image"),
        nullable=True
    )
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False)

    @property
    def id(self) -> str:
        return self.artifact_id

    @property
    def has_image(self) -> bool:
        return self.baseline_image_id is not None

    images = relationship("Image", foreign_keys="[Image.artifact_id]", back_populates="artifact", cascade="all, delete-orphan")
    comparisons = relationship("ImageComparison", back_populates="artifact", cascade="all, delete-orphan")
    schedules = relationship("Schedule", back_populates="artifact", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="artifact", cascade="all, delete-orphan")
    baseline_image = relationship("Image", foreign_keys=[baseline_image_id], post_update=True)

class Image(Base):
    __tablename__ = "images"

    image_id = Column(String(6), primary_key=True, index=True, default=_gen_id)
    artifact_id = Column(String(6), ForeignKey("artifacts.artifact_id", ondelete="CASCADE"), nullable=False)
    device_id = Column(String(6), ForeignKey("iot_devices.device_id"), nullable=True)
    operator_id = Column(String(6), ForeignKey("users.user_id"), nullable=True)
    image_type = Column(Enum(ImageType, name="image_type"), nullable=False)
    image_path = Column(String(500), nullable=False)
    captured_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)
    is_valid = Column(Boolean, default=True, nullable=False)

    artifact = relationship("Artifact", foreign_keys=[artifact_id], back_populates="images")
    device = relationship("IotDevice")
    operator = relationship("User")

class ImageComparison(Base):
    __tablename__ = "image_comparisons"

    comparison_id = Column(String(6), primary_key=True, index=True, default=_gen_id)
    artifact_id = Column(String(6), ForeignKey("artifacts.artifact_id", ondelete="CASCADE"), nullable=False)
    previous_image_id = Column(String(6), ForeignKey("images.image_id"), nullable=True)
    current_image_id = Column(String(6), ForeignKey("images.image_id"), nullable=False)
    schedule_id = Column(String(6), ForeignKey("schedules.id"), nullable=True)
    
    damage_score = Column(Float, nullable=False, default=0.0)
    ssim_score = Column(String(16), nullable=True)
    heatmap_path = Column(String(500), nullable=True)
    status = Column(Enum(ComparisonStatus, name="comparison_status"), default=ComparisonStatus.good, nullable=False)
    inspection_type = Column(Enum(InspectionType, name="inspection_type_enum"), default=InspectionType.sudden, nullable=False)
    description = Column(Text, nullable=True)
    detections_json = Column(Text, nullable=True)
    created_by = Column(String(100), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)

    artifact = relationship("Artifact", back_populates="comparisons")
    previous_image = relationship("Image", foreign_keys=[previous_image_id])
    current_image = relationship("Image", foreign_keys=[current_image_id])
    schedule = relationship("Schedule", back_populates="inspection_result")
    alerts = relationship("Alert", back_populates="comparison", cascade="all, delete-orphan")

class Alert(Base):
    __tablename__ = "alerts"

    alert_id = Column(String(6), primary_key=True, index=True, default=_gen_id)
    artifact_id = Column(String(6), ForeignKey("artifacts.artifact_id", ondelete="CASCADE"), nullable=False)
    comparison_id = Column(String(6), ForeignKey("image_comparisons.comparison_id", ondelete="CASCADE"), nullable=False)
    alert_level = Column(Enum(AlertLevel, name="alert_level"), nullable=False)
    is_handled = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)

    artifact = relationship("Artifact", back_populates="alerts")
    comparison = relationship("ImageComparison", back_populates="alerts")

class Schedule(Base):
    __tablename__ = "schedules"

    id = Column(String(6), primary_key=True, index=True, default=_gen_id)
    artifact_id = Column(String(6), ForeignKey("artifacts.artifact_id", ondelete="CASCADE"), nullable=False)
    scheduled_date = Column(DateTime(timezone=True), nullable=False)
    scheduled_time = Column(String(8), nullable=False, default="09:00")
    operator_username = Column(String(100), nullable=False, default="")
    notes = Column(Text, nullable=True)
    completed = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), default=_utc_now, nullable=False)

    artifact = relationship("Artifact", back_populates="schedules")
    inspection_result = relationship("ImageComparison", back_populates="schedule", uselist=False)
