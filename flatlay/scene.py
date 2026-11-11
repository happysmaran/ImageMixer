"""Measurements taken from the target photo so inserted objects can be matched to it."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SceneStats:
    dark_floor: np.ndarray      # RGB level of the scene's own blacks; an inserted black must not be blacker
    surface_rgb: np.ndarray     # dominant surface colour, the one reflected by glossy edges
    shadow_guess: np.ndarray    # per-channel light transmission used until the user samples a real shadow


def analyze(bgr: np.ndarray) -> SceneStats:
    s = min(1.0, 600 / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb = small[:, :, ::-1].astype(np.float32).reshape(-1, 3)
    lum = rgb.mean(1)
    # The darkest few percent are the only true blacks in a flat-lay (other devices, deep shadow), so they
    # define the black point an inserted object has to reach to look like it was in the room.
    sel = lum <= np.percentile(lum, 8)
    dark = np.median(rgb[sel], axis=0)
    surface = np.median(rgb, axis=0)
    # Shadows on a saturated surface keep hue but lose more of the weaker channels; scaling by the
    # surface's own chroma reproduces that without a sample.
    chroma = surface / max(float(surface.max()), 1.0)
    shadow = np.clip(0.30 + 0.28 * chroma, 0.2, 0.6)
    return SceneStats(dark_floor=dark, surface_rgb=surface, shadow_guess=shadow)
