"""Optional trained cutout model.

Colour-based cutting fails exactly where "any object" gets interesting: plants, fur, glassware and anything
close in colour to what it lies on. A small salient-object network handles those, so it is used when
`onnxruntime` is installed. Without it everything still works, just with the colour-based cutter, so the
dependency stays optional and nothing here may raise into the caller.
"""
from __future__ import annotations

import importlib.util
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODELS = {
    # name: (file, url). Both are U2-Net family networks released with the rembg project.
    "silueta": ("silueta.onnx", "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx"),
    "u2netp": ("u2netp.onnx", "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"),
    "u2net": ("u2net.onnx", "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"),
}
DEFAULT_MODEL = os.environ.get("FLATLAY_SEGMENTER", "silueta")
_SIZE = 320
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

_lock = threading.Lock()
_sessions: dict = {}
_failed: dict = {}


def cache_dir() -> Path:
    d = Path(os.environ.get("FLATLAY_MODEL_DIR", Path.home() / ".cache" / "flatlay"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def available() -> bool:
    # FLATLAY_NO_MODEL lets a user (or a test run) opt out of the model and its first-use download.
    if os.environ.get("FLATLAY_NO_MODEL"):
        return False
    return importlib.util.find_spec("onnxruntime") is not None


def _model_file(name: str) -> Path:
    fname, url = MODELS[name]
    path = cache_dir() / fname
    if not path.exists():
        # Downloaded to a temporary name first so an interrupted download is never mistaken for a model.
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def _session(name: str):
    with _lock:
        if name in _failed:
            return None
        if name not in _sessions:
            try:
                import onnxruntime as ort
                _sessions[name] = ort.InferenceSession(str(_model_file(name)), providers=["CPUExecutionProvider"])
            except Exception as e:    # missing network, corrupt file, unsupported runtime: fall back quietly
                _failed[name] = str(e)
                return None
        return _sessions[name]


def last_error(name: str = DEFAULT_MODEL) -> str | None:
    return _failed.get(name)


def predict(bgr: np.ndarray, name: str | None = None) -> np.ndarray | None:
    """Probability (0..1) that each pixel belongs to the main object, at the photo's resolution.

    Returns None when the model cannot be used, so callers can fall back.
    """
    name = name or DEFAULT_MODEL
    if name not in MODELS or not available():
        return None
    sess = _session(name)
    if sess is None:
        return None
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(bgr, (_SIZE, _SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    rgb = rgb / max(float(rgb.max()), 1e-6)
    x = ((rgb - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
    try:
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    except Exception as e:
        _failed[name] = str(e)
        return None
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    return cv2.resize(out.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
