"""Compositing: place the cut-out object into the scene so it reads as photographed there.

The model is deliberately physical rather than artistic, because every number can then be tied to something
visible in the target photo:

* Perspective: a camera looking straight at a surface shows the inner side face of objects far from the optical
  axis. The object's footprint on the surface is its front face scaled toward the optical axis by
  height/camera distance, and the side face is everything swept between the two.
* Shadow: cast from the footprint, not from the front face, because the footprint is where the object meets the
  surface. An object held off the surface (a hanging frame) casts a displaced, softer shadow instead, since the
  gap lengthens the shadow's path. The shadow is multiplied onto the surface so its texture shows through.
* Tone: black point, sheen and rim light are derived from the scene's own colours and the light direction.
* Glass: refraction cannot be recovered from one photo, so glass is drawn as a clearly approximate see-through
  placeholder rather than a confident but wrong imitation.
"""
from __future__ import annotations

import cv2
import numpy as np

from .cutout import Cutout
from .model import Params
from .surface import _surface_mask

# (depth along the side face, multiplier of local surface colour, constant offset). Depth 0 is the edge shared
# with the front face and 1 is where the object meets the surface.
MATERIALS = {
    "plastic_over_steel": [
        (0.00, 0.00, (58, 54, 52)), (0.05, 0.05, (105, 95, 90)), (0.10, 0.08, (22, 19, 19)),
        (0.45, 0.07, (16, 14, 14)), (0.52, 0.25, (60, 45, 40)), (0.58, 0.56, (0, 0, 0)),
        (0.80, 0.43, (0, 0, 0)), (1.00, 0.28, (0, 0, 0)),
    ],
    "black_plastic": [
        (0.00, 0.00, (50, 48, 46)), (0.06, 0.04, (85, 80, 76)), (0.15, 0.05, (24, 20, 20)),
        (1.00, 0.05, (14, 12, 12)),
    ],
    "brushed_metal": [
        (0.00, 0.30, (150, 150, 152)), (0.08, 0.25, (110, 110, 112)), (0.50, 0.35, (70, 70, 72)),
        (1.00, 0.30, (45, 45, 46)),
    ],
}


# (rim light, sheen) multipliers. Matte and woven surfaces scatter light, so edge highlights and the soft gradient
# across the face are weaker than on polished or metal surfaces.
FINISHES = {"matte": (0.5, 0.5), "glossy": (1.0, 1.0), "metal": (1.4, 1.2), "fabric": (0.4, 0.2), "glass": (1.0, 1.0)}


def _warp(m: np.ndarray, M: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(m, M.astype(np.float32), (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _grab(scene: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Window that may extend past the scene; edge pixels are repeated so blurs near the border stay sane."""
    H, W = scene.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return np.zeros((y1 - y0, x1 - x0, 3), scene.dtype)
    return cv2.copyMakeBorder(scene[cy0:cy1, cx0:cx1], cy0 - y0, y1 - cy1, cx0 - x0, x1 - cx1,
                              cv2.BORDER_REPLICATE)


def _window_mask(m: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(m.shape[1], x1), min(m.shape[0], y1)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = m[cy0:cy1, cx0:cx1]
    return out


def _smear_edge_colour(front_rgb: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Colour of the nearest edge pixel at every pixel. A rounded object's side is the same material as its rim
    seen at a grazing angle, so extending the rim colour outward is a better guess than any fixed palette."""
    src = np.where(edge, 0, 255).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    table = np.zeros((int(labels.max()) + 1, 3), np.float32)
    table[labels[edge]] = front_rgb[edge]
    return table[labels]


def object_black(cut: Cutout) -> np.ndarray:
    """The object's own black level, so the lift to the scene's black point is the difference, not a constant."""
    sel = cut.alpha > 0.9
    px = cut.rgb[sel][::7].astype(np.float32)
    return np.percentile(px, 1.0, axis=0) if len(px) else np.zeros(3, np.float32)


def _rotated_size(w: int, h: int, deg: float) -> tuple:
    if abs(deg) < 0.01:
        return w, h
    r = np.radians(deg)
    c, s = abs(np.cos(r)), abs(np.sin(r))
    return int(h * s + w * c) + 2, int(h * c + w * s) + 2


def _local_surface(scene: np.ndarray, box: tuple, margin: int) -> np.ndarray:
    """Surface colour around the object (RGB). Lighting varies across the frame, so a global median would
    tint glossy edges with the wrong brightness."""
    x0, y0, x1, y1 = box
    win = _grab(scene, x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    mask = np.ones(win.shape[:2], bool)
    mask[margin:win.shape[0] - margin, margin:win.shape[1] - margin] = False
    px = win[mask][::6]
    if len(px) < 50:
        px = scene.reshape(-1, 3)[::97]
    return np.median(px, axis=0)[::-1].astype(np.float32)


def _illuminant(rgb_px: np.ndarray) -> np.ndarray:
    """Colour of the light a photo was taken under, as a unit-mean RGB triple (shades-of-grey estimate).

    Strongly coloured pixels (wood, cloth) are left out when enough neutral ones exist, because a large orange
    surface would otherwise be read as an orange light and the correction would drain the colour from everything.
    """
    px = rgb_px.reshape(-1, 3).astype(np.float32)
    px = px[(px.max(1) > 25) & (px.max(1) < 245)]
    if len(px) < 200:
        return np.ones(3, np.float32)
    sat = (px.max(1) - px.min(1)) / np.maximum(px.max(1), 1.0)
    neutral = px[sat < 0.35]
    if len(neutral) >= 0.03 * len(px) and len(neutral) >= 200:
        px = neutral
    ill = np.power(np.mean(np.power(px / 255.0, 6.0), axis=0), 1 / 6.0)
    return (ill / max(float(ill.mean()), 1e-6)).astype(np.float32)


def _scene_light_map(scene_bgr: np.ndarray) -> np.ndarray:
    """Brightness falloff of the scene's light across the frame, from its surface only (objects would darken it).

    Returned at the scene's resolution with a mean near 1, so it can be sampled over any window.
    """
    s = min(1.0, 400 / max(scene_bgr.shape[:2]))
    small = cv2.resize(scene_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    m = cv2.erode(_surface_mask(small, None), np.ones((5, 5), np.float32))
    lum = small.astype(np.float32).mean(2)
    sig = max(6.0, 0.06 * max(small.shape[:2]))
    num = cv2.GaussianBlur(lum * m, (0, 0), sig)
    den = cv2.GaussianBlur(m, (0, 0), sig)
    L = num / np.maximum(den, 0.02)
    L = np.where(den > 0.02, L, np.mean(lum))
    L = L / max(float(np.mean(L)), 1e-3)
    return cv2.resize(L, (scene_bgr.shape[1], scene_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def _cached(cache, key, fn):
    if cache is None:
        return fn()
    if key not in cache:
        cache[key] = fn()
    return cache[key]


def _object_arrays(cut: Cutout, H: int, W: int, sigma: float, cache):
    key = ("obj", H, round(sigma, 2))
    if cache is not None and key in cache:
        rgb, alpha = cache[key]
        return rgb.copy(), alpha.copy()
    down = H < cut.rgb.shape[0]
    interp = cv2.INTER_AREA if down else cv2.INTER_CUBIC
    rgb = cv2.resize(cut.rgb, (W, H), interpolation=interp).astype(np.float32)
    a = cv2.resize(cut.alpha, (W, H), interpolation=interp)
    # Eroding slightly removes the original background's colour from the silhouette edge.
    alpha = np.clip((a - 0.56) / 0.18, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.45, sigma))
    if cache is not None:
        if len([k for k in cache if isinstance(k, tuple) and k[0] == "obj"]) > 6:
            for k in [k for k in cache if isinstance(k, tuple) and k[0] == "obj"][:3]:
                cache.pop(k)
        cache[key] = (rgb.copy(), alpha.copy())
    return rgb, alpha


def render(scene_bgr: np.ndarray, cut: Cutout, p: Params, scale: float = 1.0, cache: dict | None = None,
           obj_black: np.ndarray | None = None, behind: np.ndarray | None = None) -> np.ndarray:
    """Composite onto `scene_bgr`, which must already be resized by `scale`; `p` stays in full-res units.

    `behind` is a 0..1 mask at the same scale marking parts of the scene that must stay in front of the object.
    """
    S = float(scale)
    Hs, Ws = scene_bgr.shape[:2]
    rng = np.random.default_rng(7)

    H = max(8, int(round(p.height_px * S)))
    W = max(4, int(round(H * cut.rgb.shape[1] / cut.rgb.shape[0])))
    t_px = max(1.0, p.thickness_px * S)
    ow, oh = _rotated_size(W, H, p.rotation_deg)
    ox, oy = int(round(p.cx * S - ow / 2)), int(round(p.cy * S - oh / 2))

    az = np.radians(p.light_az)
    shadow_dir = np.array([np.cos(az), np.sin(az)], np.float32)
    light = -shadow_dir                       # unit vector pointing toward the light
    surf = _local_surface(scene_bgr, (ox, oy, ox + ow, oy + oh), max(20, int(0.12 * H)))

    # ---- object in its own frame ---------------------------------------------------------------
    rgb, alpha = _object_arrays(cut, H, W, p.softness * S, cache)
    if p.flip_h:
        # Mirroring is offered because a mirrored object keeps its silhouette but swaps which side its own
        # lighting comes from; it is only sensible for objects without text.
        rgb, alpha = rgb[:, ::-1].copy(), alpha[:, ::-1].copy()
    rim_k, sheen_k = FINISHES.get(p.finish, FINISHES["glossy"])
    glass = p.finish == "glass"
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if p.clear_edge_frac > 0:
        # Translucent trim lets the surface show through, which grey bedsheet pixels from the source would not.
        e = max(2.0, p.clear_edge_frac * H)
        strip = np.clip((yy - (H - e)) / max(1.0, 0.3 * e), 0, 1)[..., None]
        rgb = rgb * (1 - 0.4 * strip) + (surf * 1.0)[None, None, :] * 0.4 * strip

    if abs(p.rotation_deg) > 0.01:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), -p.rotation_deg, 1.0)
        M[0, 2] += ow / 2 - W / 2
        M[1, 2] += oh / 2 - H / 2
        rgb = cv2.warpAffine(rgb, M, (ow, oh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(alpha, M, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=0)
    oh, ow = alpha.shape[:2]
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)

    # ---- tone: match the scene's black point, light direction and colour -----------------------
    rgb_raw = rgb.copy()
    inner = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    mg = cv2.GaussianBlur(alpha, (0, 0), max(1.2, 0.00127 * oh))
    gx = cv2.Sobel(mg, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mg, cv2.CV_32F, 0, 1, ksize=3)
    nrm = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    facing = np.clip((-gx / nrm) * light[0] + (-gy / nrm) * light[1], 0, 1)
    ext = 0.5 * (abs(light[0]) * ow + abs(light[1]) * oh)
    along = ((xx - ow / 2) * light[0] + (yy - oh / 2) * light[1]) / max(ext, 1.0)

    ob = obj_black
    if ob is None:
        ob = cache.setdefault("obj_black", object_black(cut)) if cache is not None else object_black(cut)
    if p.color_match > 0:
        # The object was photographed under its own light; bringing its white balance toward the scene's is what
        # stops a silver lid from staying cool in a warm room (or the reverse).
        obj_ill = _cached(cache, ("obj_ill", id(cut)),
                          lambda: _illuminant(cut.rgb[cut.alpha > 0.9] if (cut.alpha > 0.9).sum() > 200 else cut.rgb))
        scene_ill = _cached(cache, ("scene_ill", Hs, Ws), lambda: _illuminant(scene_bgr[:, :, ::-1]))
        corr = np.clip(scene_ill / np.maximum(obj_ill, 1e-3), 0.8, 1.25)
        rgb = rgb * (1 + p.color_match * (corr - 1))[None, None, :]
    lift = np.clip(np.asarray(p.black_rgb, np.float32) - p.gain * ob, 0, 90)
    lum = rgb.mean(2, keepdims=True)
    dark = (1 - lum / 255.0) ** 3             # only dark areas are lifted, so the screen stays luminous
    rgb = rgb * p.gain + dark * lift
    g = np.clip(0.5 + 0.5 * along, 0, 1) ** 1.5
    rgb += dark * g[..., None] * (0.133 * surf) * p.sheen * sheen_k

    rim = np.exp(-inner / max(1.5, 0.0028 * oh)) * (0.25 + 0.9 * facing)
    rgb += rim[..., None] * (0.5 * surf) * p.rim * rim_k

    if p.light_grad > 0:
        # The scene's light is uneven across the frame; an object lit evenly looks pasted on, so it receives the same
        # falloff. Only the variation across the object is applied, so its overall brightness stays as set.
        lmap = _cached(cache, ("light_map", Hs, Ws), lambda: _scene_light_map(scene_bgr))
        win_l = _grab(np.repeat(lmap[..., None], 3, 2), ox, oy, ox + ow, oy + oh)[..., 0]
        inside = alpha > 0.5
        if inside.any():
            ratio = win_l / max(float(win_l[inside].mean()), 1e-3)
            rgb = rgb * (1 + p.light_grad * (np.clip(ratio, 0.6, 1.5) - 1))[..., None]

    rgb = cv2.GaussianBlur(rgb, (0, 0), max(0.3, p.softness * S))
    rgb += rng.normal(0, p.grain, rgb.shape[:2])[..., None].astype(np.float32)
    rgb = np.clip(rgb, 0, 255)

    # ---- shadow and perspective geometry ---------------------------------------------------------
    cam_px = max(1.0, p.cam_height * Ws)
    px_, py_ = p.principal(Ws / S, Hs / S)
    px_, py_ = px_ * S, py_ * S
    standoff = max(0.0, p.standoff_px * S)
    soft = max(0.05, p.shadow_soft)
    core_sigma = max(0.6, 0.03 * t_px * soft + 0.05 * standoff * soft)
    pen_sigma = max(1.5, 0.09 * t_px * soft + 0.15 * standoff * soft)   # the gap widens the penumbra
    far = max(abs(ox - px_), abs(ox + ow - px_)), max(abs(oy - py_), abs(oy + oh - py_))
    lean = float(np.hypot(*far)) / cam_px
    pad = int((t_px + standoff) * (p.shadow_len + 0.8 + lean) + 4 * pen_sigma + 40 * S)
    wx0, wy0, wx1, wy1 = ox - pad, oy - pad, ox + ow + pad, oy + oh + pad
    win = _grab(scene_bgr, wx0, wy0, wx1, wy1).astype(np.float32)
    win_orig = win.copy()
    full = np.zeros(win.shape[:2], np.float32)
    full[oy - wy0:oy - wy0 + oh, ox - wx0:ox - wx0 + ow] = alpha

    perspective = p.side_faces and not glass
    draws_sides = perspective and p.shape != "flat"
    h_front = standoff + t_px
    # A dome or ball is widest at its base, so seen off-axis its outline grows far less than a straight-sided
    # body of the same height does; only a fraction of the straight-sided parallax is applied.
    lean_k = 0.3 if p.shape == "round" else 1.0
    s_back = float(np.clip(1 - lean_k * t_px / cam_px, 0.5, 1.0)) if perspective else 1.0
    s_foot = float(np.clip(1 - h_front / cam_px, 0.4, 1.0)) if perspective else 1.0
    pxw, pyw = px_ - wx0, py_ - wy0
    side_sweep = full.copy()
    depth = np.ones(full.shape, np.float32)
    if draws_sides and s_back < 1.0:
        unset = np.ones(full.shape, bool)
        # One sweep step per pixel of side width keeps the depth gradient smooth instead of banded.
        reach = float(np.hypot(pxw - (ox - wx0 + ow / 2), pyw - (oy - wy0 + oh / 2))) * (1 - s_back)
        n = int(np.clip(np.ceil(reach), 28, 160))
        for i in range(n + 1):
            u = i / n
            sc = 1 - u * (1 - s_back)
            Mu = np.array([[sc, 0, (1 - sc) * pxw], [0, sc, (1 - sc) * pyw]])
            cur = _warp(full, Mu)
            side_sweep = np.maximum(side_sweep, cur)
            newly = unset & (cur > 0.5)
            depth[newly] = u
            unset &= ~newly
    has_sides = draws_sides and s_back < 1.0
    if s_foot < 1.0:
        Mf = np.array([[s_foot, 0, (1 - s_foot) * pxw], [0, s_foot, (1 - s_foot) * pyw]])
        base = _warp(full, Mf)
    else:
        base = full

    # Shadow of a body between heights `standoff` and `standoff + thickness`: the near end of the sweep belongs to
    # the back plane and the far end to the front plane, so a gap detaches the shadow from the object.
    v0 = shadow_dir * standoff * p.shadow_len
    v1 = shadow_dir * h_front * p.shadow_len
    nstep = int(np.clip(np.hypot(*(v1 - v0)) / 2, 4, 40))
    sweep = np.zeros_like(base)
    for i in range(nstep + 1):
        v = v0 + (v1 - v0) * (i / nstep)
        sweep = np.maximum(sweep, _warp(base, np.array([[1, 0, v[0]], [0, 1, v[1]]])))
    core = cv2.GaussianBlur(sweep, (0, 0), core_sigma)
    pen = cv2.GaussianBlur(sweep, (0, 0), pen_sigma)
    smap = np.clip(0.8 * core + 0.35 * pen, 0, 1) * p.shadow_strength
    r = max(1, int(round(0.04 * t_px)))
    ao_src = cv2.dilate(side_sweep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    # Contact darkening only exists where the object touches the surface.
    touching = float(np.exp(-standoff / (0.6 * t_px + 3.0 * S)))
    ao = np.clip(cv2.GaussianBlur(ao_src, (0, 0), max(1.0, 0.067 * t_px)), 0, 1) * p.contact * touching
    if glass:
        # Clear glass lets most light through, so its shadow is faint.
        smap, ao = smap * 0.35, ao * 0.4
    trans = np.clip(np.asarray(p.shadow_rgb, np.float32), 0.02, 1.0)
    shade = (1 - smap[..., None] * (1 - trans)) * (1 - ao[..., None] * (1 - trans ** 0.55))
    win = win * shade[..., ::-1]

    # ---- side face colours --------------------------------------------------------------------
    if has_sides:
        if p.shape == "round":
            front = np.zeros(win.shape, np.float32)
            front[oy - wy0:oy - wy0 + oh, ox - wx0:ox - wx0 + ow] = rgb
            din = cv2.distanceTransform((full > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
            lo = max(1.5, 0.004 * oh)
            edge = (din >= lo) & (din < lo + 2.0)
            if edge.sum() > 20:
                smear = _smear_edge_colour(front, edge)
                shade_u = 0.92 - 0.55 * np.power(depth, 0.8)
                side_rgb = smear * shade_u[..., None]
                side_rgb = side_rgb * (1 - 0.2 * depth[..., None]) + surf[None, None, :] * (0.06 * depth[..., None])
            else:
                side_rgb = np.zeros(win.shape, np.float32)
            side_rgb = side_rgb * p.side_brightness
        else:
            stops = MATERIALS.get(p.side_material, MATERIALS["black_plastic"])
            ds = [s_[0] for s_ in stops]
            chans = []
            for c in range(3):
                vals = [s_[1] * surf[c] + s_[2][c] for s_ in stops]
                chans.append(np.interp(depth, ds, vals))
            side_rgb = np.stack(chans, -1).astype(np.float32) * p.side_brightness
        lowf = np.random.default_rng(11).normal(0, 1, (6, 6)).astype(np.float32)
        side_rgb *= (1 + 0.12 * cv2.resize(lowf, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_CUBIC))[..., None]
        side_rgb = cv2.GaussianBlur(side_rgb, (0, 0), max(0.3, p.softness * S))
        side_rgb += rng.normal(0, p.grain, side_rgb.shape[:2])[..., None].astype(np.float32)
        sa = np.clip(side_sweep, 0, 1)[..., None]
        win = win * (1 - sa) + np.clip(side_rgb, 0, 255)[..., ::-1] * sa

    # ---- front face on top ---------------------------------------------------------------------
    y0o, x0o = oy - wy0, ox - wx0
    roi = win[y0o:y0o + oh, x0o:x0o + ow]
    if glass:
        tint = rgb_raw[alpha > 0.5].mean(0) if (alpha > 0.5).any() else surf
        # Frosting the backdrop under the glass hides the fact that nothing is being refracted.
        blurred = cv2.GaussianBlur(roi, (0, 0), max(1.0, 0.004 * oh))
        roi[:] = roi * (1 - 0.4 * alpha[..., None]) + blurred * 0.4 * alpha[..., None]
        edge_k = np.exp(-inner / max(2.0, 0.009 * oh)) * (0.4 + 0.6 * facing)
        streak = np.exp(-((along - 0.35) / 0.12) ** 2) * 0.18
        a_eff = alpha * np.clip(p.glass_opacity + 0.85 * edge_k + 0.6 * streak, 0, 0.92)
        body = tint * 0.85
        col = body[None, None, :] + (edge_k * 0.8 + streak)[..., None] * (np.array([235, 240, 245], np.float32) - body)[None, None, :]
        roi[:] = roi * (1 - a_eff[..., None]) + np.clip(col, 0, 255)[..., ::-1] * a_eff[..., None]
    else:
        a3 = alpha[..., None]
        roi[:] = roi * (1 - a3) + rgb[..., ::-1] * a3

    # ---- things that stay in front of the object -----------------------------------------------
    if behind is not None and p.use_behind:
        m = _window_mask(behind, wx0, wy0, wx1, wy1)[..., None]
        win = win * (1 - m) + win_orig * m

    out = scene_bgr.copy()
    cx0, cy0, cx1, cy1 = max(0, wx0), max(0, wy0), min(Ws, wx1), min(Hs, wy1)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0:cy1, cx0:cx1] = np.clip(win[cy0 - wy0:cy1 - wy0, cx0 - wx0:cx1 - wx0], 0, 255).astype(np.uint8)
    return out
