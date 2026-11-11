"""Parameters that describe one composite, kept in full-resolution scene pixels.

Keeping every length in full-resolution units lets the UI preview at any size and the export reproduce
the same picture, because the renderer scales lengths by the preview factor instead of the caller
recomputing them.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scene import SceneStats


@dataclass
class Params:
    # Placement: centre of the object's front face and its long-side length, in scene pixels.
    cx: float = 0.0
    cy: float = 0.0
    height_px: float = 1000.0
    rotation_deg: float = 0.0          # clockwise
    # Physical size, used for the thickness-to-pixels conversion that drives side faces and shadows.
    length_mm: float = 100.0
    thickness_mm: float = 8.0
    # Perspective: the side face an object shows grows with its distance from the optical axis.
    side_faces: bool = True
    # How the object's volume is drawn at its sides: "flat" (print, sticker), "slab" (device, book, frame: the
    # silhouette extruded straight back) or "round" (mug, bottle, ball: sides curve away and darken).
    shape: str = "slab"
    # Gap between the object's back and the surface it is photographed against. Zero for things lying flat or
    # mounted flush; a hanging frame or a lamp on a bracket casts a detached, softer, displaced shadow.
    standoff_mm: float = 0.0
    flip_h: bool = False
    cam_height: float = 1.06           # camera height above the surface, in scene widths
    principal_x: float = -1.0          # optical axis in scene px; negative means "image centre"
    principal_y: float = -1.0
    side_material: str = "plastic_over_steel"
    side_brightness: float = 1.0
    # Light: direction the shadow falls, degrees, 0 = right, 90 = down (image coordinates).
    light_az: float = -54.0
    shadow_len: float = 0.55           # in object thicknesses
    shadow_strength: float = 1.0
    shadow_soft: float = 1.0
    shadow_rgb: list = field(default_factory=lambda: [0.55, 0.48, 0.32])  # fraction of light that survives
    contact: float = 1.0
    # Tone matching.
    black_rgb: list = field(default_factory=lambda: [16.0, 13.0, 11.0])   # target black point of the scene
    gain: float = 0.94
    rim: float = 1.0
    sheen: float = 1.0
    softness: float = 0.8              # px at full resolution, to match the scene's optics
    grain: float = 1.3
    clear_edge_frac: float = 0.0       # translucent trim along the bottom edge, as a fraction of length
    # Surface finish scales how strongly sheen and rim light respond; "glass" switches to a see-through
    # placeholder because real refraction cannot be recovered from a single photo.
    finish: str = "glossy"
    glass_opacity: float = 0.22
    use_behind: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        names = {f.name for f in fields(cls)}
        base = cls()
        for k, v in (d or {}).items():
            if k in names:
                setattr(base, k, v)
        return base

    def to_dict(self) -> dict:
        return asdict(self)

    def principal(self, scene_w: int, scene_h: int) -> tuple:
        px = scene_w / 2 if self.principal_x < 0 else self.principal_x
        py = scene_h / 2 if self.principal_y < 0 else self.principal_y
        return px, py

    @property
    def thickness_px(self) -> float:
        return self.thickness_mm * self.height_px / max(self.length_mm, 1e-6)

    @property
    def standoff_px(self) -> float:
        return self.standoff_mm * self.height_px / max(self.length_mm, 1e-6)


def default_params(scene_w: int, scene_h: int, stats: SceneStats, cut_aspect: float, light=None) -> Params:
    p = Params()
    p.cx, p.cy = scene_w / 2, scene_h / 2
    p.height_px = 0.55 * scene_h
    p.cam_height = 1.06
    p.black_rgb = [float(v) for v in stats.dark_floor]
    p.shadow_rgb = [float(v) for v in stats.shadow_guess]
    # A measured light beats the guess, but only when the photo had enough shadows to measure it from.
    if light is not None and light.confidence >= 0.35:
        p.light_az = float(light.az)
        if light.shadow_rgb is not None:
            p.shadow_rgb = [float(v) for v in light.shadow_rgb]
    return p
