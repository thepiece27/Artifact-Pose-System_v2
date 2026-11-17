"""Plane-aware change detection up to SSIM + AI input packages (I4, P0-P4).

The YOLO model, its inference code, confidence threshold, two-phase crops,
NMS and status rules are NOT changed here: only what is fed to YOLO (which
crop, which coordinates) and the mapping of its boxes back to the reference
image frame.
"""
