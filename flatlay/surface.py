"""Extending the photographed surface (cloth, wood, paper) past the frame edge.

Mirroring the edge leaves a visible seam and repeated structure, and a generative model would alter the
surface the real objects sit on. Image quilting re-tiles real texture from clean parts of the same photo,
joined along minimum-error seams, so the added area is made of the photo's own pixels. Lighting is handled
separately from texture: the slow brightness gradient is carried over from the edge, the fine pattern is
re-tiled, and the two are multiplied back together.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import inpaint

# Rotating so the requested side becomes "right" lets one implementation serve all four sides.
_ROT = {"right": 0, "top": -1, "left": 2, "bottom": 1}


def _surface_mask(bgr: np.ndarray, ref_rgb: np.ndarray | None) -> np.ndarray:
    """Pixels that belong to the surface rather than to objects lying on it."""
    s = min(1.0, 1200 / max(bgr.shape[:2]))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Pile and weave alternate bright and dark at pixel scale; classifying raw pixels would speckle the mask.
    lab = cv2.GaussianBlur(lab, (0, 0), max(1.0, 0.0025 * max(small.shape[:2])))
    if ref_rgb is None:
        # The surface is the largest region of colour in the photo. The strip along the edge is not a safe
        # reference: an object touching that edge (a phone, say) can fill most of it.
        pts = lab.reshape(-1, 3)
        pts = pts[np.random.default_rng(0).choice(len(pts), min(20000, len(pts)), replace=False)]
        cv2.setRNGSeed(1)
        _, labels, centers = cv2.kmeans(pts, 4, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                        3, cv2.KMEANS_PP_CENTERS)
        ref = centers[np.argmax(np.bincount(labels.ravel(), minlength=4))]
    else:
        px = np.uint8([[np.clip(ref_rgb, 0, 255)[::-1]]])
        ref = cv2.cvtColor(px, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    d = lab - ref
    # Lightness is nearly ignored because shadows change it a lot while the surface's hue stays put.
    dist = np.sqrt(0.06 * d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    m = (dist < 26).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.resize(m, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def _largest_rect(mask: np.ndarray, min_side: int):
    """Largest axis-aligned all-ones rectangle with both sides >= min_side. Returns (x, y, w, h) or None."""
    h, w = mask.shape
    heights = np.zeros(w, int)
    best, best_area = None, 0
    for y in range(h):
        heights = np.where(mask[y] > 0, heights + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                width = x - sx
                if width >= min_side and sh >= min_side and width * sh > best_area:
                    best_area = width * sh
                    best = (sx, y - sh + 1, width, sh)
                start = sx
            stack.append((start, cur))
    return best


def _pick_source(clean: np.ndarray, block: int):
    """Choose the rectangle of clean surface the new texture is sampled from."""
    cell = 8
    small = cv2.resize(clean, (clean.shape[1] // cell, clean.shape[0] // cell), interpolation=cv2.INTER_AREA)
    # Margin keeps patches away from object edges, where cast shadows would be mistaken for texture.
    r = max(1, int(0.015 * max(clean.shape[:2]) / cell))
    small = cv2.erode((small > 0.98).astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    for min_px in (3 * block, 2 * block, int(1.4 * block)):
        rect = _largest_rect(small, max(2, min_px // cell))
        if rect:
            x, y, w, h = [v * cell for v in rect]
            maxw, maxh = 2600, 600
            if w > maxw:
                x += (w - maxw) // 2; w = maxw
            if h > maxh:
                y += (h - maxh) // 2; h = maxh
            return x, y, w, h
    return None


def _mincut(err: np.ndarray) -> np.ndarray:
    """Boolean mask over err's columns: True where the incoming patch should win (right of the cut)."""
    h, w = err.shape
    e = err.copy()
    for i in range(1, h):
        left = np.r_[np.inf, e[i - 1, :-1]]
        right = np.r_[e[i - 1, 1:], np.inf]
        e[i] += np.minimum(np.minimum(left, e[i - 1]), right)
    path = np.zeros(h, int)
    path[-1] = int(np.argmin(e[-1]))
    for i in range(h - 2, -1, -1):
        j = path[i + 1]
        lo, hi = max(j - 1, 0), min(j + 2, w)
        path[i] = lo + int(np.argmin(e[i, lo:hi]))
    return np.arange(w)[None, :] >= path[:, None]


def _extend_right(img: np.ndarray, px: int, blend_px: int, falloff: float, seed: int,
                  surface_rgb, progress, use_model: bool = True) -> np.ndarray:
    H, W = img.shape[:2]
    A = img.astype(np.float32)
    big = max(H, W)
    sigma = max(2.0, 0.0045 * big)
    B = int(np.clip(round(0.018 * big / 8) * 8, 32, 128))
    O = B // 4
    step = B - O
    # On small photos a fixed-width blend would rewrite most of the image, so it is capped at a quarter of it.
    blend_px = int(min(blend_px, 0.25 * W))
    band = min(int(blend_px + 0.03 * big), int(0.4 * W))
    X0 = W - band

    cloth = _surface_mask(img, surface_rgb)
    cloth = cv2.erode(cloth, np.ones((5, 5), np.float32))
    # Normalised convolution estimates lighting from surface pixels only, so objects do not darken it.
    num = cv2.GaussianBlur(A * cloth[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(cloth, (0, 0), sigma)
    Lc = num / np.maximum(den, 0.03)[..., None]
    T = A / np.maximum(Lc, 1.0)
    Tg = T.mean(2)

    src = _pick_source(cloth, B)
    if src is None:
        raise ValueError("Could not find a clean patch of the surface to copy texture from; "
                         "the surface is too cluttered or too small.")
    sx, sy, sw, sh = src
    S = T[sy:sy + sh, sx:sx + sw]
    Sg = Tg[sy:sy + sh, sx:sx + sw].astype(np.float32)
    Hs, Ws = Sg.shape

    CW = (W + px - X0) + B
    nrows = int(np.ceil((H - O) / step)) + 1
    CH = nrows * step + O
    canvas = np.ones((CH, CW, 3), np.float32)
    seed_cols = T[:, X0:X0 + O].copy()
    seed_cols[cloth[:, X0:X0 + O] < 0.5] = 1.0   # neutral where an object sits, so no object texture is propagated
    canvas[:H, :O] = seed_cols
    rng = np.random.default_rng(seed)
    ncols = int(np.ceil((CW - B) / step)) + 1

    for r in range(nrows):
        for c in range(ncols):
            y, x = r * step, min(c * step, CW - B)
            left = canvas[y:y + B, x:x + O].mean(2).astype(np.float32)
            cost = cv2.matchTemplate(Sg, left, cv2.TM_SQDIFF)[:Hs - B + 1, :Ws - B + 1].copy()
            if r > 0:
                top = canvas[y:y + O, x:x + B].mean(2).astype(np.float32)
                cost += cv2.matchTemplate(Sg, top, cv2.TM_SQDIFF)[:Hs - B + 1, :Ws - B + 1]
            k = min(12, cost.size - 1)
            idx = np.argpartition(cost.ravel(), k)[:k + 1]
            iy, ix = np.unravel_index(rng.choice(idx), cost.shape)
            patch = S[iy:iy + B, ix:ix + B]
            exist = canvas[y:y + B, x:x + B]
            m = np.ones((B, B), bool)
            m[:, :O] &= _mincut(((exist[:, :O] - patch[:, :O]) ** 2).sum(2))
            if r > 0:
                m[:O, :] &= _mincut(((exist[:O, :] - patch[:O, :]) ** 2).sum(2).T).T
            canvas[y:y + B, x:x + B] = np.where(m[..., None], patch, exist)
        if progress:
            progress((r + 1) / nrows)

    Wn = W + px
    Tsyn = canvas[:H, :Wn - X0]

    # Lighting for the new area: the edge's profile, easing off by `falloff` so the area does not look pasted.
    edge_w = max(4, int(0.005 * big))
    edge = Lc[:, W - edge_w:W].mean(1)
    edge = cv2.GaussianBlur(edge.reshape(H, 1, 3), (0, 0), sigmaX=0.1, sigmaY=max(2.0, 0.0085 * big)).reshape(H, 3)
    L = np.zeros((H, Wn - X0, 3), np.float32)
    L[:, :W - X0] = Lc[:, X0:W]
    xs = np.arange(px, dtype=np.float32) / max(px, 1)
    fall = 1 - falloff * (xs * xs * (3 - 2 * xs))
    lowfreq = np.random.default_rng(3).normal(0, 1, (5, 8)).astype(np.float32)
    low = 1 + 0.025 * cv2.resize(lowfreq, (px, H), interpolation=cv2.INTER_CUBIC)
    L[:, W - X0:] = edge[:, None, :] * fall[None, :, None] * low[..., None]
    # Wood, stone and similar surfaces drift in colour and brightness in ways a repeated edge profile cannot
    # follow, so a trained model supplies that soft layer when it is installed; quilting still supplies the grain.
    soft = inpaint.soft_extension(img, cloth, px) if use_model else None
    if soft is not None:
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=max(3.0, 0.012 * big))
        # The network keeps the brightness structure it invents but drifts in hue as it moves away from known
        # pixels, so the structure is taken from it and the hue is held at the photo's own edge colour. Rows are
        # smoothed together so the structure is kept.
        def lab(x):
            return cv2.cvtColor(np.clip(x, 0, 255).astype(np.float32).reshape(-1, 1, 3) / 255, cv2.COLOR_BGR2LAB).reshape(x.shape)
        soft_lab = lab(soft.reshape(-1, 3)).reshape(soft.shape)
        edge_lab = lab(edge)
        band_w = max(edge_w, min(px // 6, 48))
        dl = edge_lab[:, 0] - soft_lab[:, :band_w, 0].mean(1)
        dl = cv2.GaussianBlur(dl.reshape(H, 1), (0, 0), sigmaX=0.1, sigmaY=max(2.0, 0.03 * big)).reshape(H)
        out_lab = np.empty_like(soft_lab)
        out_lab[..., 0] = soft_lab[..., 0] + dl[:, None]
        out_lab[..., 1:] = edge_lab[:, None, 1:]
        L[:, W - X0:] = cv2.cvtColor(out_lab.reshape(-1, 1, 3), cv2.COLOR_LAB2BGR).reshape(soft.shape) * 255
    new = L * Tsyn

    # Blend zone: where nothing touches the old edge the new texture fades in gradually (the original's edge
    # area can differ in sharpness); rows with an object keep the original up to the object and switch
    # straight to new texture beyond it.
    non = (cloth[:, X0:W] < 0.5)
    obj_rows = non.sum(1) >= max(8, int(0.087 * band))
    last_non = np.where(obj_rows, band - 1 - np.argmax(non[:, ::-1], axis=1), -1).astype(np.float32)
    wph = cv2.GaussianBlur(obj_rows.astype(np.float32).reshape(-1, 1), (0, 0), sigmaX=0.1,
                           sigmaY=max(2.0, 0.007 * big)).reshape(-1)
    xx = np.arange(Wn - X0, dtype=np.float32)[None, :]
    ramp = np.clip((xx - 0.02 * big) / max(blend_px, 1), 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)
    alpha = wph[:, None] + (1 - wph[:, None]) * ramp
    alpha[:, W - X0:] = 1.0
    keep = np.zeros((H, Wn - X0), bool)
    keep[:, :band] = obj_rows[:, None] & (np.arange(band)[None, :] <= (last_non[:, None] + 5))
    alpha = np.where(keep, 0.0, alpha)

    out = np.zeros((H, Wn, 3), np.float32)
    out[:, :W] = A
    orig = out[:, X0:Wn].copy()
    out[:, X0:Wn] = alpha[..., None] * new + (1 - alpha[..., None]) * orig
    return np.clip(out, 0, 255).astype(np.uint8)


def extend(scene_bgr: np.ndarray, side: str, px: int, blend_px: int = 320, falloff: float = 0.06,
           seed: int = 5, surface_rgb=None, progress=None, use_model: bool = True) -> np.ndarray:
    """Return the scene with `px` more surface added on `side` ('left', 'right', 'top' or 'bottom')."""
    if side not in _ROT:
        raise ValueError(f"unknown side {side!r}")
    px = int(px)
    if px < 16:
        raise ValueError("Extension must be at least 16 px.")
    k = _ROT[side]
    rot = np.ascontiguousarray(np.rot90(scene_bgr, k))
    out = _extend_right(rot, px, int(blend_px), float(falloff), seed, surface_rgb, progress, use_model)
    return np.ascontiguousarray(np.rot90(out, -k))
