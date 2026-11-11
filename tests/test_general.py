"""Tests for the parts that make the tool work on things other than flat-lay devices: shapes, standoff,
glass, measured light, occlusion and the optional cutout model."""
import io
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from flatlay import segmenter
from flatlay.cutout import Cutout, cut_out
from flatlay.lighting import (LightEstimate, angle_between, estimate_object_light, estimate_scene_light,
                              light_mismatch)
from flatlay.render import render
from flatlay.session import Session
from test_pipeline import make_object, make_scene


def textured_scene(w=900, h=600, seed=3):
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0)
    noise /= noise.std()
    base = np.array([70, 110, 190], np.float32)
    return np.clip(base[None, None, :] * (1 + 0.1 * noise)[..., None], 0, 255).astype(np.uint8)


def scene_with_shadow(dx, dy):
    """A dark block on a textured surface whose shadow is offset by (dx, dy), as a real lit scene would show."""
    img = textured_scene().astype(np.float32)
    block = np.zeros(img.shape[:2], np.float32)
    block[200:420, 300:480] = 1
    sh = cv2.warpAffine(block, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]))
    sh = cv2.GaussianBlur(sh, (0, 0), 4) * (1 - block)
    img *= (1 - 0.5 * sh)[..., None]
    img[block > 0] = (22, 20, 20)
    return np.clip(img, 0, 255).astype(np.uint8)


def beveled(shadow_az, w=420, h=900):
    """Rounded slab shaded by a known light, the way a lit device looks in its own photo."""
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (30, 30), (w - 30, h - 30), 255, -1)
    m = (cv2.GaussianBlur(m, (0, 0), 9) > 128).astype(np.uint8)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    height = np.sin(np.clip(d / (0.06 * min(h, w)), 0, 1) * np.pi / 2)
    gx, gy = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
    nz = 1 / np.sqrt(1 + (gx * 8) ** 2 + (gy * 8) ** 2)
    az = np.radians(shadow_az)
    lam = np.clip(-gx * 8 * nz * -np.cos(az) * 0.8 + -gy * 8 * nz * -np.sin(az) * 0.8 + nz * 0.6, 0, 1)
    img = np.repeat((40 + 160 * lam)[..., None], 3, 2).astype(np.uint8)
    return Cutout(rgb=img, alpha=cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2), rect=(0, 0, w, h),
                  source_size=(w, h))


# ---- lighting ------------------------------------------------------------------------------------
@pytest.mark.parametrize("dx,dy,az", [(30, 22, 36), (-30, 20, 146), (28, -26, -43), (-26, -26, -135)])
def test_scene_light_follows_existing_shadows(dx, dy, az):
    est = estimate_scene_light(scene_with_shadow(dx, dy))
    assert est.confidence > 0.4
    assert angle_between(est.az, az) < 35
    assert est.shadow_rgb is not None and all(0.2 < v < 0.9 for v in est.shadow_rgb)


def test_scene_without_objects_reports_no_light():
    est = estimate_scene_light(textured_scene())
    assert est.confidence == 0.0


@pytest.mark.parametrize("az", [-135, -45, 45, 135])
def test_object_light_from_bevel_shading(az):
    est = estimate_object_light(beveled(az))
    assert est.confidence > 0.4
    assert angle_between(est.az, az) < 25


def test_unshaded_object_reports_low_confidence():
    flat = Cutout(rgb=np.full((300, 200, 3), 120, np.uint8), alpha=np.ones((300, 200), np.float32),
                  rect=(0, 0, 200, 300), source_size=(200, 300))
    assert estimate_object_light(flat).confidence < 0.2


def test_mismatch_suggests_a_flip_only_when_it_helps():
    est = lambda az, conf=0.8: LightEstimate(az=az, confidence=conf, shadow_rgb=None, sources=1)
    scene = est(20)
    # Shadow falls left in the object's photo but right in the scene: mirroring puts them in agreement.
    assert light_mismatch(scene, est(160))["fix"] == "flip_h"
    # Light from opposite sides vertically: a horizontal mirror cannot fix it, and a vertical flip would invert it.
    assert light_mismatch(est(-60), est(120))["fix"] is None
    assert light_mismatch(scene, est(30)) is None              # already agree
    assert light_mismatch(scene, est(160, 0.3)) is None        # object estimate too unsure to act on
    assert light_mismatch(est(20, 0.2), est(160)) is None      # scene estimate too unsure to act on


# ---- rendering of other kinds of object ----------------------------------------------------------
@pytest.fixture(scope="module")
def sess():
    return Session(make_scene(), make_object())


def place(sess, **kw):
    p = sess.defaults()
    p.cx, p.cy, p.height_px = 650, 300, 260
    p.length_mm, p.thickness_mm = 100, 30
    p.light_az = 0.0                                     # shadow to the right
    p.cam_height = 0.6
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_flat_shape_draws_no_side_face(sess):
    left = int(650 - 260 * sess.aspect() / 2)
    patch = lambda im: im[240:360, left - 12:left - 2].astype(int)
    slab = sess.render_full(place(sess, shape="slab"))
    flat = sess.render_full(place(sess, shape="flat"))
    assert np.abs(patch(slab) - patch(flat)).mean() > 15            # the side face is what differs
    # With no thickness to speak of, a flat object and a straight-sided one without perspective coincide.
    thin_flat = sess.render_full(place(sess, shape="flat", thickness_mm=0.4))
    thin_off = sess.render_full(place(sess, shape="slab", thickness_mm=0.4, side_faces=False))
    assert np.abs(patch(thin_flat) - patch(thin_off)).mean() < 4


def test_round_sides_are_narrower_than_straight_sides(sess):
    def side_width(shape):
        a = sess.render_full(place(sess, shape=shape)).astype(int)
        b = sess.render_full(place(sess, shape=shape, side_faces=False)).astype(int)
        diff = np.abs(a - b).sum(2)[300, :650] > 40
        return int(diff[:int(650 - 260 * sess.aspect() / 2)].sum())
    assert 0 < side_width("round") < side_width("slab")


def test_standoff_detaches_and_displaces_the_shadow(sess):
    right = int(650 + 260 * sess.aspect() / 2)
    strip = lambda im, a, b: im[280:320, right + a:right + b].astype(float).mean()
    bare = sess.scene
    touching = sess.render_full(place(sess, shape="flat", thickness_mm=6, standoff_mm=0, shadow_len=1.0))
    hanging = sess.render_full(place(sess, shape="flat", thickness_mm=6, standoff_mm=30, shadow_len=1.0))
    far_a, far_b = 20, 40                                            # where the gap's shadow lands, clear of contact shade
    # Touching: shade right beside the object, none out where the hanging shadow lands.
    assert strip(touching, 2, 8) < 0.85 * strip(bare, 2, 8)
    assert strip(touching, far_a, far_b) > 0.93 * strip(bare, far_a, far_b)
    # Hanging: the shadow has moved out; the strip beside the object is no longer in contact shade.
    assert strip(hanging, far_a, far_b) < 0.85 * strip(bare, far_a, far_b)
    assert strip(hanging, 2, 8) < 0.9 * strip(bare, 2, 8)   # still shaded, but by the displaced shadow
    assert strip(hanging, 60, 90) > 0.95 * strip(bare, 60, 90)


def test_glass_lets_the_scene_show_through(sess):
    p = place(sess, finish="glass", shape="slab")
    out = sess.render_full(p)
    cx, cy = 650, 300
    inner = out[cy - 20:cy + 20, cx - 10:cx + 10].astype(int)
    bare = sess.scene[cy - 20:cy + 20, cx - 10:cx + 10].astype(int)
    assert np.abs(inner - bare).mean() < 40                # still mostly the surface behind it
    solid = sess.render_full(place(sess, finish="glossy"))[cy - 20:cy + 20, cx - 10:cx + 10].astype(int)
    assert np.abs(solid - bare).mean() > 2 * np.abs(inner - bare).mean()


def test_flip_matches_a_pre_mirrored_cutout(sess):
    p = place(sess)
    p.side_faces = False
    flipped_param = sess.render_full(Params_with(p, flip_h=True))
    c = sess.cutout
    mirrored = Cutout(rgb=c.rgb[:, ::-1].copy(), alpha=c.alpha[:, ::-1].copy(), rect=c.rect, source_size=c.source_size)
    ref = render(sess.scene, mirrored, p, 1.0, {}, sess.obj_black)
    assert np.abs(flipped_param.astype(int) - ref.astype(int)).mean() < 0.5


def Params_with(p, **kw):
    from flatlay.model import Params
    q = Params.from_dict(p.to_dict())
    for k, v in kw.items():
        setattr(q, k, v)
    return q


def test_finishes_scale_edge_light(sess):
    def edge(finish):
        out = sess.render_full(place(sess, finish=finish, shape="flat")).astype(int)
        ref = sess.render_full(place(sess, finish="glossy", shape="flat")).astype(int)
        return np.abs(out - ref).mean()
    assert edge("glossy") == 0 and edge("metal") > 0 and edge("fabric") > 0


# ---- things in front of the object -----------------------------------------------------------------
def test_painted_region_stays_in_front(sess):
    s = Session(make_scene(), make_object())
    p = place(s, shape="flat", side_faces=False)
    before = s.render_full(p)
    x0 = int(650 - 260 * s.aspect() / 2)
    cov = s.paint_behind([[x0 + 20, 250], [x0 + 20, 350]], radius=25, snap=False)
    assert cov > 0
    after = s.render_full(p)
    region = (slice(260, 340), slice(x0 + 15, x0 + 25))
    assert np.abs(after[region].astype(int) - s.scene[region].astype(int)).mean() < 3     # scene shows again
    assert np.abs(before[region].astype(int) - s.scene[region].astype(int)).mean() > 10   # it was covered before
    s.clear_behind()
    assert np.array_equal(s.render_full(p), before)


def test_erase_removes_paint_and_snap_never_leaks_far():
    s = Session(make_scene(), make_object())
    s.paint_behind([[300, 300]], radius=20, snap=False)
    assert s.behind[300, 300] > 0.5
    s.paint_behind([[300, 300]], radius=30, erase=True)
    assert s.behind[300, 300] == 0
    s.paint_behind([[160, 350], [200, 350]], radius=12, snap=True)      # over the dark device in the scene
    ys, xs = np.where(s.behind > 0.5)
    assert xs.min() > 40 and xs.max() < 330                              # stays near the stroke
    assert s.behind[350, 180] > 0.5


def test_mask_follows_the_scene_when_it_grows_and_shrinks():
    s = Session(make_scene(), make_object())
    s.paint_behind([[200, 300]], radius=15, snap=False)
    s.extend("left", 100)
    assert s.behind.shape == s.scene.shape[:2] and s.behind[300, 300] > 0.5 and s.behind[300, 200] == 0
    s.undo_extend()
    assert s.behind.shape == s.scene.shape[:2] and s.behind[300, 200] > 0.5


# ---- API -------------------------------------------------------------------------------------------
def test_api_light_cutout_and_brush():
    import app.server as srv
    c = srv.app.test_client()
    ok_s, buf_s = cv2.imencode(".jpg", scene_with_shadow(30, 22))
    ok_o, buf_o = cv2.imencode(".jpg", make_object())
    r = c.post("/api/session", data={"scene": (io.BytesIO(buf_s.tobytes()), "s.jpg"),
                                      "object": (io.BytesIO(buf_o.tobytes()), "o.jpg")},
               content_type="multipart/form-data")
    info = r.get_json()
    assert info["analysis"]["scene_light"]["confidence"] > 0.4
    assert angle_between(info["params"]["light_az"], 36) < 35          # defaults start from the measured light
    assert info["cutout"]["method"] == "colour" and info["cutout"]["coverage"] < 0.6
    sid = info["id"]
    assert c.get(f"/api/session/{sid}/analysis").status_code == 200
    r = c.post(f"/api/session/{sid}/behind", json={"points": [[300, 300], [320, 320]], "radius": 15, "snap": False})
    assert r.get_json()["coverage"] > 0
    r = c.get(f"/api/session/{sid}/behind.png?max_w=300")
    assert r.status_code == 200 and r.mimetype == "image/png"
    assert c.post(f"/api/session/{sid}/behind/clear", json={}).get_json()["coverage"] == 0.0
    assert c.post(f"/api/session/{sid}/behind", json={"points": []}).status_code == 400
    p = info["params"]; p.update(shape="round", finish="glass", standoff_mm=10, flip_h=True)
    assert c.post(f"/api/session/{sid}/render", json={"params": p, "max_w": 400}).status_code == 200


# ---- optional trained cutout -----------------------------------------------------------------------
CACHED = Path(os.environ.get("FLATLAY_MODEL_DIR", Path.home() / ".cache" / "flatlay")) / "silueta.onnx"


@pytest.mark.skipif(not (segmenter.importlib.util.find_spec("onnxruntime") and CACHED.exists()),
                    reason="needs onnxruntime and a cached model; the suite never downloads one")
def test_trained_model_cuts_a_scene_the_colour_method_cannot(monkeypatch):
    monkeypatch.delenv("FLATLAY_NO_MODEL", raising=False)
    rng = np.random.default_rng(0)
    h, w = 500, 700
    bg = np.clip(rng.normal(120, 55, (h, w, 3)), 0, 255).astype(np.uint8)       # busy, object-coloured background
    bg = cv2.GaussianBlur(bg, (0, 0), 6)
    cv2.ellipse(bg, (350, 250), (150, 90), 0, 0, 360, (235, 235, 235), -1)       # one clear salient object
    cv2.ellipse(bg, (350, 250), (150, 90), 0, 0, 360, (60, 60, 60), 6)
    out = cut_out(bg, method="model")
    assert out.method == "model"
    area = (out.alpha > 0.5).sum() * ((out.rect[2] - out.rect[0]) * (out.rect[3] - out.rect[1])) / out.alpha.size
    assert 0.45 * (np.pi * 150 * 90) < area < 2.0 * (np.pi * 150 * 90)


def test_model_failure_falls_back_to_colour(monkeypatch):
    monkeypatch.setattr(segmenter, "predict", lambda *a, **k: None)
    assert cut_out(make_object(), method="auto").method == "colour"


# ---- trained inpainting for the extender -------------------------------------------------------------
def test_extension_falls_back_to_quilting_without_the_model():
    # Users without torch or the model file must still get an extension, not an error.
    from flatlay import inpaint, surface
    assert not inpaint.available()                 # conftest sets FLATLAY_NO_MODEL
    out = surface.extend(textured_scene(300, 200), "right", 80)
    assert out.shape == (200, 380, 3)


def test_soft_extension_is_optional_and_never_raises():
    from flatlay import inpaint
    sc = textured_scene(200, 120)
    assert inpaint.soft_extension(sc, np.ones(sc.shape[:2], np.float32), 40) is None
