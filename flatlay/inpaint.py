"""Optional trained inpainting for the low-frequency part of an extended surface.

Quilting reproduces fine texture well but cannot invent what the surface does at larger scale: a colour drift,
a darker band, the way a grain pattern bends. A trained inpainting network (LaMa) does, but its output is soft.
The two are therefore combined by the caller: this module supplies the soft layer, quilting supplies the grain.
Everything degrades to plain quilting when torch or the model file is missing, so nothing here may raise
into the caller.
"""
from __future__ import annotations

import importlib.util
import os
import threading
import urllib.request
import warnings

import cv2
import numpy as np

from .segmenter import cache_dir

MODEL_FILE = "big-lama.pt"
MODEL_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
WORK_SIZE = 896          # network cost grows quickly with size, and only the soft layer is wanted from it

_lock = threading.Lock()
_net = None
_failed: str | None = None


def available() -> bool:
    if os.environ.get("FLATLAY_NO_MODEL"):
        return False
    return importlib.util.find_spec("torch") is not None and _failed is None


def last_error() -> str | None:
    return _failed


def _load():
    global _net, _failed
    with _lock:
        if _net is not None or _failed is not None:
            return _net
        try:
            import torch
            path = cache_dir() / MODEL_FILE
            if not path.exists():
                tmp = path.with_suffix(".part")      # an interrupted download must not look like a model
                urllib.request.urlretrieve(MODEL_URL, tmp)
                tmp.replace(path)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _net = torch.jit.load(str(path), map_location="cpu").eval()
        except Exception as e:
            _failed = str(e)
        return _net


def _fill(net, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    import torch
    h, w = hole.shape
    hp, wp = (h + 7) // 8 * 8, (w + 7) // 8 * 8
    # Reflected padding keeps the hole from being seen against black, which the network reads as a dark surface.
    img = cv2.copyMakeBorder(rgb, 0, hp - h, 0, wp - w, cv2.BORDER_REFLECT)
    msk = cv2.copyMakeBorder(hole, 0, hp - h, 0, wp - w, cv2.BORDER_CONSTANT)
    # The network reads whatever sits under the mask as evidence, and a black or smeared fill makes it
    # continue a dark surface. Mirrored surface is a plausible stand-in for the new columns; small gaps left by
    # removed objects are bridged by classical inpainting.
    right = np.flatnonzero(msk[:h].sum(0) == h)
    if right.size:
        x0 = int(right.min())
        mirror = img[:, max(0, x0 - (wp - x0)):x0][:, ::-1]
        img[:, x0:x0 + mirror.shape[1]] = mirror
        if mirror.shape[1] < wp - x0:
            img[:, x0 + mirror.shape[1]:] = img[:, x0 + mirror.shape[1] - 1:x0 + mirror.shape[1]]
        inner = msk.copy()
        inner[:, x0:] = 0
        if inner.any():
            img = np.where(inner[..., None] > 0, cv2.inpaint(img, inner, 5, cv2.INPAINT_TELEA), img)
    else:
        img = np.where(msk[..., None] > 0, cv2.inpaint(img, msk, 5, cv2.INPAINT_TELEA), img)
    x = torch.from_numpy(img.transpose(2, 0, 1)[None].copy()).float() / 255
    y = torch.from_numpy((msk > 0).astype(np.float32))[None, None]
    with torch.no_grad():
        out = net(x, y)[0].permute(1, 2, 0).numpy()
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)[:h, :w]


def soft_extension(bgr: np.ndarray, surface: np.ndarray, px: int, progress=None) -> np.ndarray | None:
    """Soft (low-frequency) colour for `px` new columns to the right of `bgr`, or None if unavailable.

    `surface` is 1 where the photo shows the surface; everything else is treated as unknown so objects at the
    edge are not continued into the new area. The result is (H, px, 3) float32 BGR at the photo's resolution.
    """
    if not available():
        return None
    net = _load()
    if net is None:
        return None
    H, W = bgr.shape[:2]
    ctx = min(W, int(1.3 * H))
    s = min(1.0, WORK_SIZE / max(H, ctx + px))
    h = max(8, int(round(H * s)))
    win = cv2.resize(bgr[:, W - ctx:], (max(8, int(round(ctx * s))), h), interpolation=cv2.INTER_AREA)
    sur = cv2.resize(surface[:, W - ctx:], win.shape[1::-1], interpolation=cv2.INTER_AREA)
    unknown = cv2.dilate((sur < 0.9).astype(np.uint8), np.ones((9, 9), np.uint8))
    cur_rgb = np.ascontiguousarray(win[:, :, ::-1])
    cur_unknown = unknown
    want = int(round(px * s))
    # A hole much wider than the visible surface comes back as an unrelated flat colour, so grow in steps.
    done, step_max = 0, max(8, int(0.4 * h))
    try:
        while done < want:
            step = min(step_max, want - done)
            ctx_w = min(cur_rgb.shape[1], int(1.0 * h))
            rgb = np.concatenate([cur_rgb[:, -ctx_w:], np.zeros((h, step, 3), np.uint8)], 1)
            hole = np.concatenate([cur_unknown[:, -ctx_w:], np.ones((h, step), np.uint8)], 1)
            filled = _fill(net, rgb, hole)
            cur_rgb = np.concatenate([cur_rgb, filled[:, ctx_w:]], 1)
            cur_unknown = np.concatenate([cur_unknown, np.zeros((h, step), np.uint8)], 1)
            done += step
            if progress:
                progress(done / want)
    except Exception as e:
        global _failed
        _failed = str(e)
        return None
    strip = cur_rgb[:, win.shape[1]:][:, :, ::-1]
    strip = cv2.resize(strip, (px, H), interpolation=cv2.INTER_CUBIC)
    return strip.astype(np.float32)
