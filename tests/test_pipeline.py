import io
import time

import cv2
import numpy as np
import pytest

from flatlay import surface
from flatlay.cutout import cut_out
from flatlay.session import Session


def make_scene(w=900, h=600, seed=1):
    """Warm textured surface with a lighting gradient and one dark device, like a real flat-lay."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.0)
    noise = noise / noise.std()
    grad = np.linspace(1.15, 0.85, w, dtype=np.float32)[None, :]
    base = np.array([60, 90, 200], np.float32)                    # BGR orange-red
    img = base[None, None, :] * (grad * (1 + 0.12 * noise))[..., None]
    img[200:520, 80:250] = (18, 16, 16)                           # a black device
    return np.clip(img, 0, 255).astype(np.uint8)


def make_object(w=700, h=1000):
    """Dark rounded device with a bright screen, photographed on a grey sheet."""
    img = np.full((h, w, 3), (150, 150, 150), np.uint8)
    rng = np.random.default_rng(2)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (200, 200), (500, 800), (12, 12, 12), -1)
    cv2.rectangle(img, (230, 240), (470, 380), (230, 200, 140), -1)
    return img


@pytest.fixture(scope="module")
def session():
    return Session(make_scene(), make_object())


def test_cutout_finds_the_object():
    cut = cut_out(make_object())
    a = cut.alpha > 0.5
    h, w = a.shape
    assert 560 <= h <= 640 and 280 <= w <= 330, (h, w)
    assert a[h // 2, w // 2] and a.mean() > 0.9           # solid, no holes from the screen


def test_render_only_touches_the_neighbourhood_of_the_object(session):
    p = session.defaults()
    p.cx, p.cy, p.height_px = 650, 300, 260
    out = session.render_full(p)
    diff = np.abs(out.astype(int) - session.scene.astype(int)).sum(2) > 0
    ys, xs = np.where(diff)
    assert diff[300, 650]                                   # the object is there
    assert xs.min() > 350 and not diff[:, :300].any()       # nothing changes far from it


def test_object_can_hang_off_the_edge(session):
    p = session.defaults()
    p.cx, p.cy, p.height_px = 880, 580, 400
    assert session.render_full(p).shape == session.scene.shape


def test_rotation_changes_footprint(session):
    p = session.defaults()
    p.cx, p.cy, p.height_px = 600, 300, 300
    a = session.render_full(p)
    p.rotation_deg = 45
    b = session.render_full(p)
    assert np.abs(a.astype(int) - b.astype(int)).sum() > 0


def test_side_face_appears_on_the_side_facing_the_optical_axis(session):
    p = session.defaults()
    p.cx, p.cy, p.height_px = 760, 300, 300
    p.length_mm, p.thickness_mm = 100, 25                   # exaggerated so the effect is unmistakable
    p.cam_height = 0.6
    on = session.render_full(p)
    p.side_faces = False
    off = session.render_full(p)
    left_edge = int(760 - 300 * session.aspect() / 2)
    near = lambda a, b: np.abs(a[250:350, left_edge - 12:left_edge - 2].astype(int)
                               - b[250:350, left_edge - 12:left_edge - 2].astype(int)).mean()
    assert near(on, off) > 20                               # the object is right of centre, so its left side shows


def test_shadow_falls_away_from_the_light(session):
    p = session.defaults()
    p.cx, p.cy, p.height_px = 600, 300, 200
    p.length_mm, p.thickness_mm = 100, 30
    p.side_faces = False
    p.light_az = 0.0                                        # shadow falls to the right
    out = session.render_full(p)
    half_w = 200 * session.aspect() / 2
    right = out[280:320, int(600 + half_w + 8):int(600 + half_w + 20)].mean()
    left = out[280:320, int(600 - half_w - 20):int(600 - half_w - 8)].mean()
    assert right < 0.8 * left


def test_preview_and_full_agree_on_geometry(session):
    p = session.defaults()
    p.cx, p.cy, p.height_px = 600, 300, 300
    full = cv2.resize(session.render_full(p), (450, 300), interpolation=cv2.INTER_AREA)
    prev = session.render_preview(p, max_w=450)
    assert prev.shape == full.shape
    assert np.abs(prev.astype(int) - full.astype(int)).mean() < 6


@pytest.mark.parametrize("side", ["right", "left", "top", "bottom"])
def test_extension_adds_surface_on_every_side(side):
    scene = make_scene(500, 350, seed=4)
    out = surface.extend(scene, side, 120)
    dh, dw = (120, 0) if side in ("top", "bottom") else (0, 120)
    assert out.shape == (350 + dh, 500 + dw, 3)
    # Pixels well away from the join are untouched.
    far = {"right": (slice(None), slice(0, 250)), "left": (slice(None), slice(120 + 250, None)),
           "top": (slice(120 + 200, None), slice(None)), "bottom": (slice(0, 200), slice(None))}[side]
    assert np.array_equal(out[far], scene[:, :250] if side == "right" else
                          scene[:, 250:] if side == "left" else scene[200:] if side == "top" else scene[:200])
    # The new area continues the surface. Compare strips on either side of the join, over rows/columns that
    # contain no device, because brightness legitimately drifts with the lighting gradient.
    clean = (slice(0, 150), slice(None)) if side in ("left", "right") else (slice(None), slice(300, 500))
    o = {"right": out[:, :500], "left": out[:, 120:], "top": out[120:], "bottom": out[:350]}[side][clean]
    n = {"right": out[:, 500:], "left": out[:, :120], "top": out[:120], "bottom": out[350:]}[side][clean]
    join = {"right": (o[:, -60:], n[:, :60]), "left": (o[:, :60], n[:, -60:]),
            "top": (o[:60], n[-60:]), "bottom": (o[-60:], n[:60])}[side]
    m_old, m_new = [x.reshape(-1, 3).mean(0) for x in join]
    assert np.all(np.abs(m_new - m_old) < 0.2 * m_old)
    s_old, s_new = [x.reshape(-1, 3).std(0).mean() for x in join]
    assert 0.4 < s_new / s_old < 2.5                         # texture amplitude, not flat and not noisy


def test_extension_rejects_tiny_requests():
    with pytest.raises(ValueError):
        surface.extend(make_scene(300, 200), "right", 4)


def test_api_roundtrip():
    import app.server as srv
    c = srv.app.test_client()
    ok_s, buf_s = cv2.imencode(".jpg", make_scene())
    ok_o, buf_o = cv2.imencode(".jpg", make_object())
    r = c.post("/api/session", data={"scene": (io.BytesIO(buf_s.tobytes()), "s.jpg"),
                                      "object": (io.BytesIO(buf_o.tobytes()), "o.jpg")},
               content_type="multipart/form-data")
    assert r.status_code == 200, r.data
    info = r.get_json()
    sid, params = info["id"], info["params"]
    params.update(cx=600, cy=300, height_px=250)
    r = c.post(f"/api/session/{sid}/render", json={"params": params, "max_w": 450})
    assert r.status_code == 200 and r.mimetype == "image/jpeg"
    r = c.post(f"/api/session/{sid}/calibrate/shadow", json={"lit": [500, 100], "shadow": [150, 300]})
    assert all(0 < v <= 1 for v in r.get_json()["shadow_rgb"])
    r = c.post(f"/api/session/{sid}/export", json={"params": params, "format": "png"})
    assert r.status_code == 200 and r.mimetype == "image/png"
    r = c.post(f"/api/session/{sid}/extend", json={"side": "left", "px": 100})
    job = r.get_json()["job"]
    for _ in range(100):
        j = c.get(f"/api/job/{job}").get_json()
        if j["state"] != "running":
            break
        time.sleep(0.2)
    assert j["state"] == "done" and j["size"] == [1000, 600] and j["shift"] == [100, 0]
    r = c.post(f"/api/session/{sid}/undo_extend", json={})
    assert r.get_json()["size"] == [900, 600]


def test_bad_inputs_get_readable_errors():
    import app.server as srv
    c = srv.app.test_client()
    r = c.post("/api/session", data={}, content_type="multipart/form-data")
    assert r.status_code == 400 and "error" in r.get_json()
    r = c.post("/api/session/nope/render", json={})
    assert r.status_code == 404
