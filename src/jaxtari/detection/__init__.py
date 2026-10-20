"""Detector-based object-centric observations (pure JAX: jittable, vmappable, no torch).

    from jaxtari.detection import YOLOObjectCentricWrapper, load_detector
"""
from .labels import CLASS_NAMES, load_registry
from .wrapper import MISSING, YOLOObjectCentricState, YOLOObjectCentricWrapper
from .yolo import Detections, YOLODetector, load_detector, resolve_checkpoint

__all__ = ["YOLOObjectCentricWrapper", "YOLOObjectCentricState", "YOLODetector", "Detections", "load_detector", "resolve_checkpoint",
           "load_registry", "CLASS_NAMES", "MISSING"]
