"""Local web server for ImageMixer. Everything runs on this machine; photos are never uploaded anywhere."""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, Response

from flatlay.model import Params
from flatlay.session import Session, load_bgr

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "devices.json"

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024    # phone photos are 5-12 MB; allow batches and RAW-sized JPEGs

SESSIONS: dict[str, Session] = {}
JOBS: dict[str, dict] = {}
MAX_SESSIONS = 4    # each session holds full-resolution photos


def _session(sid: str) -> Session:
    s = SESSIONS.get(sid)
    if s is None:
        raise KeyError("Session expired. Reload the page and open the photos again.")
    return s


def _jpeg(img: np.ndarray, q: int = 88) -> Response:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.errorhandler(KeyError)
def _key_error(e):
    return jsonify(error=str(e.args[0]) if e.args else "Not found"), 404


@app.errorhandler(ValueError)
def _value_error(e):
    return jsonify(error=str(e)), 400


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/devices")
def devices():
    return jsonify(json.loads(DATA.read_text()))


@app.post("/api/session")
def create_session():
    f_scene, f_obj = request.files.get("scene"), request.files.get("object")
    if not f_scene or not f_obj:
        raise ValueError("Both a scene photo and an object photo are needed.")
    scene = load_bgr(f_scene.read())
    obj = load_bgr(f_obj.read())
    s = Session(scene, obj)
    while len(SESSIONS) >= MAX_SESSIONS:
        SESSIONS.pop(next(iter(SESSIONS)))
    SESSIONS[s.id] = s
    return jsonify(id=s.id, scene={"w": s.size[0], "h": s.size[1]}, aspect=s.aspect(),
                   params=s.defaults().to_dict(), analysis=s.analysis(), cutout=s.cutout_info(),
                   object={"w": s.object_photo.shape[1], "h": s.object_photo.shape[0]})


@app.post("/api/session/<sid>/render")
def render_preview(sid):
    s = _session(sid)
    body = request.get_json(force=True)
    p = Params.from_dict(body.get("params"))
    max_w = int(np.clip(body.get("max_w", 1400), 200, 3000))
    return _jpeg(s.render_preview(p, max_w))


@app.get("/api/session/<sid>/scene.jpg")
def scene_jpg(sid):
    return _jpeg(_session(sid).scene_preview(int(request.args.get("max_w", 1400))))


@app.post("/api/session/<sid>/export")
def export(sid):
    s = _session(sid)
    body = request.get_json(force=True)
    p = Params.from_dict(body.get("params"))
    img = s.render_full(p)
    fmt = body.get("format", "jpg")
    if fmt == "png":
        ok, buf = cv2.imencode(".png", img)
        mime, name = "image/png", "composite.png"
    else:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mime, name = "image/jpeg", "composite.jpg"
    return Response(buf.tobytes(), mimetype=mime,
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})


# ---- lighting and occlusion --------------------------------------------------------------------------
@app.get("/api/session/<sid>/analysis")
def analysis(sid):
    return jsonify(_session(sid).analysis())


@app.post("/api/session/<sid>/behind")
def paint_behind(sid):
    s = _session(sid)
    b = request.get_json(force=True)
    cover = s.paint_behind(b.get("points") or [], float(b.get("radius", 12)), bool(b.get("erase", False)),
                           bool(b.get("snap", True)))
    return jsonify(coverage=cover)


@app.post("/api/session/<sid>/behind/clear")
def clear_behind(sid):
    _session(sid).clear_behind()
    return jsonify(coverage=0.0)


@app.get("/api/session/<sid>/behind.png")
def behind_png(sid):
    ok, buf = cv2.imencode(".png", _session(sid).behind_overlay(int(request.args.get("max_w", 1400))))
    return Response(buf.tobytes(), mimetype="image/png", headers={"Cache-Control": "no-store"})


# ---- cutout ----------------------------------------------------------------------------------------
@app.get("/api/session/<sid>/object.jpg")
def object_overlay(sid):
    img, scale = _session(sid).object_overlay()
    r = _jpeg(img)
    r.headers["X-Scale"] = str(scale)
    return r


@app.post("/api/session/<sid>/cutout")
def recut(sid):
    s = _session(sid)
    rect = request.get_json(force=True).get("rect")
    if not rect or len(rect) != 4:
        raise ValueError("Draw a box around the object.")
    s.recut(tuple(rect))
    return jsonify(aspect=s.aspect(), rect=list(s.cutout.rect), cutout=s.cutout_info(), analysis=s.analysis())


# ---- calibration -----------------------------------------------------------------------------------
@app.post("/api/session/<sid>/calibrate/shadow")
def calibrate_shadow(sid):
    b = request.get_json(force=True)
    return jsonify(shadow_rgb=_session(sid).shadow_from_samples(b["lit"], b["shadow"]))


# ---- canvas extension (slow, so it runs as a job the page polls) ------------------------------------
@app.post("/api/session/<sid>/extend")
def extend(sid):
    s = _session(sid)
    b = request.get_json(force=True)
    side, px = b.get("side"), int(b.get("px", 0))
    if side not in ("left", "right", "top", "bottom"):
        raise ValueError("Choose a side to extend.")
    if not 16 <= px <= 4000:
        raise ValueError("Extend by between 16 and 4000 px.")
    for old in [k for k, v in JOBS.items() if v["state"] != "running"][:-20]:    # long sessions would otherwise keep every job
        JOBS.pop(old)
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"state": "running", "progress": 0.0}

    def work():
        try:
            def prog(f):
                JOBS[jid]["progress"] = f
            res = s.extend(side, px, int(b.get("blend_px", 320)), float(b.get("falloff", 0.06)), prog)
            JOBS[jid].update(state="done", progress=1.0, shift=res["shift"], size=res["size"])
        except Exception as e:     # surfaced to the page instead of dying silently in the thread
            JOBS[jid].update(state="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify(job=jid)


@app.get("/api/job/<jid>")
def job(jid):
    if jid not in JOBS:
        raise KeyError("Unknown job.")
    return jsonify(JOBS[jid])


@app.post("/api/session/<sid>/undo_extend")
def undo_extend(sid):
    res = _session(sid).undo_extend()
    if res is None:
        raise ValueError("Nothing to undo.")
    return jsonify(res)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ImageMixer (local web app)")
    ap.add_argument("--port", type=int, default=5057)
    ap.add_argument("--host", default="127.0.0.1", help="Keep the default unless you trust everyone on the network.")
    a = ap.parse_args()
    print(f"Open http://{a.host}:{a.port}")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
