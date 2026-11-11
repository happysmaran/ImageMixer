"""Object cutout from a casual photo.

Object photos are usually taken on whatever surface was nearby (a bedsheet, a desk), so the cutout has to
work from a rough hint instead of a clean backdrop. A coarse box is enough: GrabCut does the rest, and a
second pass at higher resolution tightens the outline where the final edge quality is decided.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import segmenter


@dataclass
class Cutout:
    rgb: np.ndarray      # tight crop of the object photo, uint8 RGB
    alpha: np.ndarray    # float32 0..1, same size as rgb; softened so later resizing does not alias
    rect: tuple          # (x0, y0, x1, y1) of the crop inside the source photo, for UI overlays
    source_size: tuple   # (width, height) of the source photo
    method: str = "colour"   # "model" when the trained segmenter produced the outline, else "colour"


def _ellipse(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return mask
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.where(lab == i, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    # Screens and buttons make GrabCut punch holes in devices; a device is one solid silhouette.
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, -1)
    return out


def guess_rect(bgr: np.ndarray) -> tuple:
    """Propose a box around the object without asking the user to draw one.

    The photo's border is almost always background, so whatever differs strongly from the border's median
    colour is a good first guess for the object.
    """
    h, w = bgr.shape[:2]
    s = min(1.0, 640 / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(0.04 * min(small.shape[:2])))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    d = cv2.GaussianBlur(d, (0, 0), 2)
    d8 = np.clip(d / max(float(d.max()), 1.0) * 255, 0, 255).astype(np.uint8)
    thr1, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def bbox_of(thr: float):
        m = ((d8 > thr).astype(np.uint8)) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(7))
        m = _largest_component(m)
        ys, xs = np.where(m > 0)
        return (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None

    loose = bbox_of(thr1)
    if loose is None:
        return (int(0.1 * w), int(0.1 * h), int(0.9 * w), int(0.9 * h))
    # A second threshold, applied to the pixels that already differ from the background, separates the object
    # from its own cast shadow and side faces, which differ less. It is only trusted when it keeps most of the
    # loose box, since on a low-contrast object it would otherwise slice the object itself.
    vals = d8[d8 > thr1].reshape(-1, 1)
    box, pad_frac = loose, 0.02
    if vals.size > 100:
        thr2, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strict = bbox_of(float(thr2) if np.ndim(thr2) == 0 else float(thr2[0]))
        if strict is not None:
            area = lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
            if area(strict) >= 0.8 * area(loose):
                box, pad_frac = strict, 0.012
    pad = int(pad_frac * max(small.shape[:2]))
    x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
    x1, y1 = min(small.shape[1] - 1, box[2] + pad), min(small.shape[0] - 1, box[3] + pad)
    return (int(x0 / s), int(y0 / s), int(x1 / s), int(y1 / s))


def _grabcut_rect(img: np.ndarray, rect: tuple, iters: int) -> np.ndarray:
    gc = np.zeros(img.shape[:2], np.uint8)
    x0, y0, x1, y1 = rect
    r = (x0, y0, max(2, x1 - x0), max(2, y1 - y0))
    bgd = np.zeros((1, 65)); fgd = np.zeros((1, 65))
    cv2.grabCut(img, gc, r, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return _fill_holes(_largest_component(m))


def _grabcut_band(img: np.ndarray, prior: np.ndarray, band: int, iters: int) -> np.ndarray:
    # Only pixels near the coarse outline are re-decided; the interior is trusted, which keeps screens and
    # dark bezels from being reclassified as background.
    k = _ellipse(band)
    sure_fg = cv2.erode(prior, k)
    maybe = cv2.dilate(prior, k)
    gc = np.full(prior.shape, cv2.GC_BGD, np.uint8)
    gc[maybe > 0] = cv2.GC_PR_BGD
    gc[prior > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    bgd = np.zeros((1, 65)); fgd = np.zeros((1, 65))
    cv2.grabCut(img, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    m = _fill_holes(_largest_component(m))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, _ellipse(2))


def _model_mask(small: np.ndarray, rect_small: tuple | None) -> np.ndarray | None:
    """Coarse mask from the trained segmenter, or None if it is unavailable or found nothing sensible.

    With a user box the network sees only that box, because on a cluttered photo it would otherwise pick
    whichever object it finds most salient, which may not be the one the user means.
    """
    h, w = small.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if rect_small is None else rect_small
    pad = int(0.05 * max(x1 - x0, y1 - y0))
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    prob = segmenter.predict(small[cy0:cy1, cx0:cx1])
    if prob is None:
        return None
    m = np.zeros((h, w), np.uint8)
    m[cy0:cy1, cx0:cx1] = (prob > 0.5).astype(np.uint8) * 255
    if rect_small is not None:
        keep = np.zeros_like(m)
        keep[y0:y1, x0:x1] = 255
        m = cv2.bitwise_and(m, keep)
    m = _fill_holes(_largest_component(m))
    if m.sum() / 255 < 0.003 * h * w:
        return None
    return m


def cut_out(bgr: np.ndarray, rect: tuple | None = None, method: str = "auto") -> Cutout:
    """`method` is "auto" (trained model when available), "model" or "colour"."""
    h, w = bgr.shape[:2]
    user_rect = rect is not None
    if rect is None:
        rect = guess_rect(bgr)
    x0, y0, x1, y1 = [int(v) for v in rect]
    x0, x1 = sorted((max(0, x0), min(w - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(h - 1, y1)))

    s1 = min(1.0, 800 / max(h, w))
    small = cv2.resize(bgr, None, fx=s1, fy=s1, interpolation=cv2.INTER_AREA)
    rect_small = tuple(int(v * s1) for v in (x0, y0, x1, y1))
    coarse, used = None, "colour"
    if method in ("auto", "model"):
        coarse = _model_mask(small, rect_small if user_rect else None)
        used = "model" if coarse is not None else "colour"
    if coarse is None:
        coarse = _grabcut_rect(small, rect_small, 6)

    s2 = min(1.0, 3000 / max(h, w))
    mid = cv2.resize(bgr, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA)
    prior = cv2.resize(coarse, (mid.shape[1], mid.shape[0]), interpolation=cv2.INTER_LINEAR)
    prior = ((prior > 127).astype(np.uint8)) * 255
    band = max(4, int(0.004 * max(mid.shape[:2])))
    fine = _grabcut_band(mid, prior, band, 6)

    ys, xs = np.where(fine > 0)
    if len(xs) == 0:
        raise ValueError("No object found inside the selected box; try drawing a box around the object.")
    mx0, my0, mx1, my1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    fx0, fy0 = int(mx0 / s2), int(my0 / s2)
    fx1, fy1 = min(w, int(np.ceil(mx1 / s2))), min(h, int(np.ceil(my1 / s2)))

    crop = bgr[fy0:fy1, fx0:fx1]
    if max(crop.shape[:2]) > 4096:
        sc = 4096 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    soft = cv2.GaussianBlur(fine[my0:my1, mx0:mx1].astype(np.float32) / 255.0, (0, 0), 1.6)
    alpha = cv2.resize(soft, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_CUBIC)
    alpha = np.clip(alpha, 0, 1).astype(np.float32)
    return Cutout(rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), alpha=alpha,
                  rect=(fx0, fy0, fx1, fy1), source_size=(w, h), method=used)
