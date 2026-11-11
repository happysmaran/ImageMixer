"""Reading the light out of photos.

An inserted object only looks present if its shadow falls the way the scene's other shadows do, and if its own
shading does not argue with them. Both are read from the pixels so the user starts from a measured value rather
than a guess. Every estimate carries a confidence, because a plain wall with nothing on it has no shadows to
read, and a wrong direction presented as fact is worse than a visible "unknown".

Directions use image coordinates in degrees: 0 points right, 90 points down. The value always means the
direction a shadow falls, which is also the direction away from the light.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .cutout import Cutout
from .surface import _surface_mask

NBINS = 24


@dataclass
class LightEstimate:
    az: float                 # direction the shadow falls, degrees
    confidence: float         # 0..1; below ~0.3 the value should not be trusted
    shadow_rgb: list | None   # fraction of light surviving inside the shadow, per RGB channel
    sources: int              # how many objects in the photo contributed

    def to_dict(self) -> dict:
        return {"az": self.az, "confidence": self.confidence, "shadow_rgb": self.shadow_rgb,
                "sources": self.sources}


def _angle(v: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def angle_between(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _normals_and_bins(comp: np.ndarray):
    """Outward normal of the component at every pixel outside it, from the gradient of the distance field."""
    dist = cv2.distanceTransform((comp == 0).astype(np.uint8), cv2.DIST_L2, 5)
    dist = cv2.GaussianBlur(dist, (0, 0), 2.0)
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=3)
    n = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = gx / n, gy / n
    ang = np.arctan2(ny, nx)
    bins = ((ang + np.pi) / (2 * np.pi) * NBINS).astype(int) % NBINS
    return dist, nx, ny, bins


def estimate_scene_light(bgr: np.ndarray) -> LightEstimate:
    """Direction and strength of the shadows the objects in this photo already cast."""
    s = min(1.0, 900 / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = _surface_mask(small, None)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    rgbf = cv2.GaussianBlur(small, (0, 0), 1.2)[:, :, ::-1].astype(np.float32)
    L = lab[:, :, 0]

    obj = ((1 - mask) > 0.5).astype(np.uint8)
    obj = cv2.morphologyEx(obj, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(obj)
    total = small.shape[0] * small.shape[1]
    order = [i for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1 if stats[i, cv2.CC_STAT_AREA] > 0.004 * total][:5]

    votes, shadow_samples, ref_samples = [], [], []
    for i in order:
        comp = (lab_cc == i).astype(np.uint8)
        if comp.all():          # nothing but object: no surface around it to read shadows from
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        r_ring = max(6, int(0.14 * min(w, h)))
        dist, nx, ny, bins = _normals_and_bins(comp)
        ring = (dist > 2) & (dist <= r_ring) & (mask > 0.5)
        far = (dist > 2.5 * r_ring) & (dist <= 4.5 * r_ring) & (mask > 0.5)
        if ring.sum() < 200 or far.sum() < 200:
            continue
        ref = float(np.median(L[far]))
        deficit = np.where(ring, np.maximum(0.0, ref - L), 0.0)
        # Texture makes single pixels bright or dark; only a deficit above the surface's own noise counts.
        floor = 1.0 * float(np.std(L[far]))
        weight = np.maximum(0.0, deficit - floor)
        if weight.sum() < 1e-3:
            continue
        score = np.zeros(NBINS)
        for k in range(NBINS):
            sel = ring & (bins == k)
            cnt = int(sel.sum())
            if cnt > 20:
                score[k] = float(weight[sel].sum()) / cnt
        centres = (np.arange(NBINS) + 0.5) / NBINS * 2 * np.pi - np.pi
        vec = np.array([(score * np.cos(centres)).sum(), (score * np.sin(centres)).sum()])
        mag, tot = float(np.hypot(*vec)), float(score.sum())
        if tot < 0.5:
            continue
        votes.append((vec, mag / tot, tot))
        idx = np.flatnonzero(ring & (deficit > np.percentile(deficit[ring], 92)))
        if len(idx) > 30:
            shadow_samples.append(np.median(rgbf.reshape(-1, 3)[idx], axis=0))
            ref_samples.append(np.median(rgbf[far], axis=0))

    if not votes:
        return LightEstimate(az=-45.0, confidence=0.0, shadow_rgb=None, sources=0)

    total_vec = np.zeros(2)
    for vec, coherence, tot in votes:
        total_vec += vec / (np.hypot(*vec) + 1e-9) * coherence * min(tot, 10.0)
    norm = sum(c * min(t, 10.0) for _, c, t in votes)
    confidence = float(np.clip(np.hypot(*total_vec) / (norm + 1e-9), 0, 1)) * float(min(1.0, norm / 2.0))
    az = _angle(total_vec)

    shadow_rgb = None
    if shadow_samples:
        ratio = np.median(shadow_samples, axis=0) / np.maximum(np.median(ref_samples, axis=0), 1.0)
        shadow_rgb = [float(v) for v in np.clip(ratio, 0.05, 1.0)]
    return LightEstimate(az=az, confidence=confidence, shadow_rgb=shadow_rgb, sources=len(votes))


def estimate_object_light(cut: Cutout) -> LightEstimate:
    """Which way the object's own shading says its shadow should fall.

    Bevels and rims facing the light are brighter than the ones facing away, and unlike the interior of the
    object (screens, labels, prints) a thin band along the silhouette is not dominated by the picture on it.
    """
    a = cut.alpha
    s = min(1.0, 500 / max(a.shape))
    a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb = cv2.resize(cut.rgb, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    inside = (a > 0.9).astype(np.uint8)
    if inside.sum() < 400:
        return LightEstimate(az=0.0, confidence=0.0, shadow_rgb=None, sources=0)
    # The photo's frame counts as the silhouette edge: a cutout that fills the whole frame would otherwise have an
    # infinite distance field and poison every later step with NaNs.
    dist_in = cv2.distanceTransform(cv2.copyMakeBorder(inside, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0), cv2.DIST_L2, 5)[1:-1, 1:-1]
    if float(dist_in.max()) < 4.0:
        return LightEstimate(az=0.0, confidence=0.0, shadow_rgb=None, sources=0)
    band_w = max(3.0, 0.04 * min(a.shape))
    band = (dist_in > 1.5) & (dist_in <= band_w + 1.5)
    # Outward normals of the silhouette, taken from the inside distance field (it grows inward).
    d = cv2.GaussianBlur(dist_in, (0, 0), 2.0)
    d = d - float(d.min())
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    n = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / n, -gy / n
    bins = ((np.arctan2(ny, nx) + np.pi) / (2 * np.pi) * NBINS).astype(int) % NBINS
    base = float(np.mean(lum[band]))
    score = np.zeros(NBINS)
    for k in range(NBINS):
        sel = band & (bins == k)
        if sel.sum() > 15:
            score[k] = float(np.mean(lum[sel])) - base
    centres = (np.arange(NBINS) + 0.5) / NBINS * 2 * np.pi - np.pi
    toward_light = np.array([(score * np.cos(centres)).sum(), (score * np.sin(centres)).sum()])
    mag = float(np.hypot(*toward_light))
    spread = float(np.abs(score).sum())
    if spread < 1e-3:
        return LightEstimate(az=0.0, confidence=0.0, shadow_rgb=None, sources=0)
    contrast = float(np.clip(spread / NBINS / 12.0, 0, 1))    # edges that barely differ say nothing
    conf = float(np.clip(mag / spread, 0, 1)) * contrast
    return LightEstimate(az=_angle(-toward_light), confidence=conf, shadow_rgb=None, sources=1)


def light_mismatch(scene: LightEstimate, obj: LightEstimate, min_scene: float = 0.35, min_obj: float = 0.5,
                   min_gap: float = 110.0):
    """Describe a clear disagreement between the object's shading and the scene's shadows, else None.

    The object-side threshold is higher because that estimate comes from bevels alone and a false alarm costs
    the user's trust in the notice. `fix` names a flip only when mirroring the object actually brings the two
    directions together; mirroring can just as easily make things worse, and a vertical flip would turn the
    object upside down.
    """
    if scene.confidence < min_scene or obj.confidence < min_obj:
        return None
    gap = angle_between(scene.az, obj.az)
    if gap < min_gap:
        return None
    flipped_gap = angle_between(scene.az, 180.0 - obj.az)
    fix = "flip_h" if flipped_gap < 60.0 and flipped_gap < gap - 40.0 else None
    return {"gap": gap, "scene_az": scene.az, "object_az": obj.az, "fix": fix}
