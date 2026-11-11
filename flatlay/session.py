"""State for one editing session: the scene, the cut-out object and the operations the UI drives.

Previews render on a downscaled copy of the scene because slider feedback has to feel immediate; export
renders the same parameters at full resolution, so what was approved on screen is what is saved.
"""
from __future__ import annotations

import threading
import uuid
from typing import Callable

import cv2
import numpy as np

from . import surface
from .cutout import Cutout, cut_out
from .lighting import LightEstimate, estimate_object_light, estimate_scene_light, light_mismatch
from .model import Params, default_params
from .render import object_black, render
from .scene import SceneStats, analyze


class Session:
    MAX_UNDO = 2     # full-resolution scenes are large; two steps back is what memory allows on a laptop

    def __init__(self, scene_bgr: np.ndarray, object_bgr: np.ndarray):
        self.id = uuid.uuid4().hex[:12]
        self.lock = threading.RLock()
        self.scene = scene_bgr
        self.object_photo = object_bgr
        self.stats: SceneStats = analyze(scene_bgr)
        self.scene_light: LightEstimate = estimate_scene_light(scene_bgr)
        self.cutout: Cutout = cut_out(object_bgr)
        self.obj_light: LightEstimate = estimate_object_light(self.cutout)
        self.obj_black = object_black(self.cutout)
        self.behind = np.zeros(scene_bgr.shape[:2], np.float32)
        self.cache: dict = {}
        self._small: dict = {}
        self._behind_small: dict = {}
        self._undo: list = []

    # ---- scene ---------------------------------------------------------------------------------
    @property
    def size(self) -> tuple:
        return self.scene.shape[1], self.scene.shape[0]

    def _scene_at(self, scale: float) -> np.ndarray:
        if scale >= 0.999:
            return self.scene
        key = round(scale, 4)
        if key not in self._small:
            if len(self._small) > 3:
                self._small.clear()
            self._small[key] = cv2.resize(self.scene, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return self._small[key]

    def _behind_at(self, scale: float) -> np.ndarray | None:
        if not self.behind.any():
            return None
        if scale >= 0.999:
            return self.behind
        key = round(scale, 4)
        if key not in self._behind_small:
            if len(self._behind_small) > 3:
                self._behind_small.clear()
            h, w = self._scene_at(scale).shape[:2]
            self._behind_small[key] = cv2.resize(self.behind, (w, h), interpolation=cv2.INTER_AREA)
        return self._behind_small[key]

    def defaults(self) -> Params:
        w, h = self.size
        c = self.cutout
        return default_params(w, h, self.stats, c.rgb.shape[1] / c.rgb.shape[0], self.scene_light)

    def aspect(self) -> float:
        return self.cutout.rgb.shape[1] / self.cutout.rgb.shape[0]

    def cutout_info(self) -> dict:
        """How the outline was made and whether it looks plausible, so the UI can say so instead of leaving
        the user to discover a cut-out that is really the whole photo."""
        c = self.cutout
        src_w, src_h = c.source_size
        coverage = float((c.alpha > 0.5).sum() * (c.rect[2] - c.rect[0]) * (c.rect[3] - c.rect[1])
                         / max(1, c.alpha.size) / max(1, src_w * src_h))
        note = None
        if coverage > 0.6:
            note = ("The object fills most of its photo, so it was probably not separated from its background. "
                    "Use Fix the cutout to draw a box around it"
                    + ("." if c.method == "model" else ", or install onnxruntime for a trained cutout model."))
        return {"method": c.method, "coverage": coverage, "note": note}

    # ---- rendering -----------------------------------------------------------------------------
    def render_preview(self, params: Params, max_w: int = 1400) -> np.ndarray:
        with self.lock:
            scale = min(1.0, max_w / self.size[0])
            return render(self._scene_at(scale), self.cutout, params, scale, self.cache, self.obj_black,
                          self._behind_at(scale))

    def render_full(self, params: Params) -> np.ndarray:
        with self.lock:
            return render(self.scene, self.cutout, params, 1.0, self.cache, self.obj_black, self._behind_at(1.0))

    def scene_preview(self, max_w: int = 1400) -> np.ndarray:
        with self.lock:
            return self._scene_at(min(1.0, max_w / self.size[0]))

    # ---- lighting ------------------------------------------------------------------------------
    def analysis(self) -> dict:
        with self.lock:
            return {
                "scene_light": self.scene_light.to_dict(),
                "object_light": self.obj_light.to_dict(),
                "mismatch": light_mismatch(self.scene_light, self.obj_light),
            }

    # ---- cutout --------------------------------------------------------------------------------
    def recut(self, rect) -> None:
        with self.lock:
            self.cutout = cut_out(self.object_photo, rect)
            self.obj_light = estimate_object_light(self.cutout)
            self.obj_black = object_black(self.cutout)
            self.cache.clear()

    def object_overlay(self, max_side: int = 900) -> tuple:
        """Object photo with the current cutout tinted, for the cutout-fixing dialog. Returns (image, scale)."""
        with self.lock:
            h, w = self.object_photo.shape[:2]
            s = min(1.0, max_side / max(h, w))
            img = cv2.resize(self.object_photo, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            x0, y0, x1, y1 = self.cutout.rect
            a = cv2.resize(self.cutout.alpha, (max(1, int((x1 - x0) * s)), max(1, int((y1 - y0) * s))))
            ov = img.copy()
            sx, sy = int(x0 * s), int(y0 * s)
            hh, ww = min(a.shape[0], img.shape[0] - sy), min(a.shape[1], img.shape[1] - sx)
            reg = ov[sy:sy + hh, sx:sx + ww].astype(np.float32)
            tint = np.array([255, 190, 60], np.float32)       # BGR: a cyan-ish highlight over the kept area
            m = (a[:hh, :ww] * 0.45)[..., None]
            ov[sy:sy + hh, sx:sx + ww] = (reg * (1 - m) + tint * m).astype(np.uint8)
            return ov, s

    # ---- calibration ---------------------------------------------------------------------------
    def shadow_from_samples(self, lit: tuple, shadow: tuple, radius: int = 12) -> list:
        """Per-channel light transmission from one lit and one shadowed point on the same surface."""
        with self.lock:
            def mean_at(pt):
                x, y = int(pt[0]), int(pt[1])
                h, w = self.scene.shape[:2]
                patch = self.scene[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
                return patch.reshape(-1, 3).mean(0)[::-1]
            lit_rgb, sh_rgb = mean_at(lit), mean_at(shadow)
            ratio = np.clip(sh_rgb / np.maximum(lit_rgb, 1.0), 0.03, 1.0)
            return [float(v) for v in ratio]

    # ---- things that stay in front of the object -------------------------------------------------
    def paint_behind(self, points: list, radius: float, erase: bool = False, snap: bool = True) -> float:
        """Mark scene pixels that must hide the object. Returns the fraction of the scene now marked.

        A hand-painted stroke is rarely accurate to the pixel, so with `snap` the stroke only chooses *which*
        thing in front is meant and GrabCut finds its edge, which is what makes tucking an object behind a leaf
        or a cable usable without careful tracing.
        """
        if not points:
            raise ValueError("Paint over the part of the scene that should stay in front.")
        with self.lock:
            H, W = self.behind.shape
            pts = np.array(points, np.float32)
            r = float(max(2.0, radius))
            pad = int(max(3 * r, 60))
            x0, y0 = max(0, int(pts[:, 0].min()) - pad), max(0, int(pts[:, 1].min()) - pad)
            x1, y1 = min(W, int(pts[:, 0].max()) + pad), min(H, int(pts[:, 1].max()) + pad)
            if x1 <= x0 or y1 <= y0:
                raise ValueError("That stroke is outside the photo.")
            stroke = np.zeros((y1 - y0, x1 - x0), np.uint8)
            local = (pts - np.array([x0, y0], np.float32)).round().astype(np.int32)
            if len(local) == 1:
                cv2.circle(stroke, tuple(local[0]), int(r), 255, -1)
            else:
                cv2.polylines(stroke, [local.reshape(-1, 1, 2)], False, 255, int(2 * r), cv2.LINE_AA)
            region = (stroke > 127)
            if erase:
                self.behind[y0:y1, x0:x1][region] = 0
            else:
                add = region
                if snap:
                    snapped = self._snap_region(x0, y0, x1, y1, stroke, r)
                    if snapped is not None:
                        add = snapped
                soft = cv2.GaussianBlur(add.astype(np.float32), (0, 0), 1.2)
                self.behind[y0:y1, x0:x1] = np.maximum(self.behind[y0:y1, x0:x1], soft)
            self._behind_small.clear()
            return float(self.behind.mean())

    def _snap_region(self, x0, y0, x1, y1, stroke, r):
        win = self.scene[y0:y1, x0:x1]
        s = min(1.0, 640 / max(win.shape[:2]))
        small = cv2.resize(win, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        st = cv2.resize(stroke, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        rs = max(1, int(r * s))
        k = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))
        gc = np.full(st.shape, cv2.GC_BGD, np.uint8)
        gc[cv2.dilate(st, k(3 * rs)) > 0] = cv2.GC_PR_BGD
        gc[cv2.dilate(st, k(rs)) > 0] = cv2.GC_PR_FGD
        core = cv2.erode(st, k(max(1, rs // 2)))
        gc[(core if core.any() else st) > 0] = cv2.GC_FGD
        try:
            bgd = np.zeros((1, 65)); fgd = np.zeros((1, 65))
            cv2.grabCut(small, gc, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return None
        m = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        n, lab, _, _ = cv2.connectedComponentsWithStats(m)
        keep = np.unique(lab[st > 0])
        m = np.isin(lab, keep[keep > 0]).astype(np.uint8)
        # A result that swallowed far more than the brush, or almost nothing of it, means the colours gave
        # GrabCut nothing to hold on to; the brush shape is the safer answer then.
        stroke_area = max(1, int((st > 0).sum()))
        if m.sum() < 0.3 * stroke_area or m.sum() > 12 * stroke_area:
            return None
        m = cv2.resize(m, (stroke.shape[1], stroke.shape[0]), interpolation=cv2.INTER_NEAREST)
        return m > 0

    def clear_behind(self) -> None:
        with self.lock:
            self.behind[:] = 0
            self._behind_small.clear()

    def behind_overlay(self, max_w: int = 1400) -> np.ndarray:
        """RGBA picture of the marked area, for showing the mask over the preview."""
        with self.lock:
            scale = min(1.0, max_w / self.size[0])
            h, w = self._scene_at(scale).shape[:2]
            m = cv2.resize(self.behind, (w, h), interpolation=cv2.INTER_AREA)
            out = np.zeros((h, w, 4), np.uint8)
            out[..., 0], out[..., 1], out[..., 2] = 230, 190, 60       # BGR
            out[..., 3] = np.clip(m * 150, 0, 255).astype(np.uint8)
            return out

    # ---- canvas extension ----------------------------------------------------------------------
    def extend(self, side: str, px: int, blend_px: int = 320, falloff: float = 0.06,
               progress: Callable[[float], None] | None = None) -> dict:
        new = surface.extend(self.scene, side, px, blend_px, falloff, progress=progress)
        with self.lock:
            self._undo.append((self.scene, self.behind))
            self._undo = self._undo[-self.MAX_UNDO:]
            self.scene = new
            pad = {"left": ((0, 0), (px, 0)), "right": ((0, 0), (0, px)),
                   "top": ((px, 0), (0, 0)), "bottom": ((0, px), (0, 0))}[side]
            self.behind = np.pad(self.behind, pad)
            self._small.clear()
            self._behind_small.clear()
            shift = {"left": (px, 0), "top": (0, px)}.get(side, (0, 0))
            return {"shift": shift, "size": self.size}

    def undo_extend(self) -> dict | None:
        with self.lock:
            if not self._undo:
                return None
            cur_w, cur_h = self.size
            self.scene, self.behind = self._undo.pop()
            self._small.clear()
            self._behind_small.clear()
            return {"size": self.size, "delta": (cur_w - self.size[0], cur_h - self.size[1])}


def load_bgr(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("That file could not be read as an image.")
    return img
