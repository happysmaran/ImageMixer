# ImageMixer

ImageMixer puts an object from one photo into another photo taken head on so it looks like it was there: matching scale, light direction, shadow, perspective and tone. It runs entirely on your computer; photos are never uploaded.

It works for any photo taken **head on** to a surface: a table seen from above, a wall, a shelf front. Objects can lie on the surface, hang or stand off it (a print, a frame), or be boxes, cans, balls or glass. It is not a general-purpose scene editor, and it does not use generative AI: the object's pixels are your photo's pixels, resized and re-toned.

## Run it

Python 3.10 or newer.

```
pip install -r requirements.txt
python -m app.server
```

Then open http://127.0.0.1:5057.

Tested with Python 3.13, NumPy 2.5, OpenCV 5.0 and Flask 3.1.

## Using it

1. Open the **scene** (the photo the object goes into) and the **object** (a photo of it on any background). The cutout takes 10 to 15 seconds on phone-sized photos.
2. **Place it.** Drag the object, scroll over it to resize, or use the Placement sliders. Arrow keys nudge by one screen pixel (hold Shift for ten).
3. **Set the real size.** Choose the device, or enter its length and thickness. Then use *Set scale from something in the photo* and click both ends of something whose size you know (a device in the scene, a credit card). *Resize to real size* applies the result.
4. **Match the light.** Drag the sun on the dial to the direction the existing shadows imply. *Sample from a shadow in the photo* reads the real shadow colour from one lit and one shadowed point.
5. **Check the side.** An object far from the middle of a top-down photo shows its inner side face, as the other objects in the scene do. *Camera height* controls how much; lower values show more side.
6. **Tone.** Black level, brightness, rim light, sheen, softness and grain bring the object to the look of the scene.
7. **Export** a full-resolution JPEG or PNG.

Double-click a slider to reset it. If the cutout includes shadow or background, use *Fix the cutout* and draw a tighter box.

### Beyond flat-lays

- **Shape** (flat, slab, round) sets how the sides are drawn; **standoff** is the gap between object and surface, which detaches and displaces the shadow (a frame on a wall).
- **Finish** (matte, glossy, metal, fabric, glass) scales sheen, rim light and shadow softness. **Glass is a placeholder**: a frosted, tinted stand-in with a faint shadow; refraction is not modelled.
- **Light** is measured from the shadows already in the scene and from the object's own bevels. If they clearly disagree the app says so, and offers a mirror when that fixes it.
- **In front of the object** is a brush: paint what should pass in front of the inserted object (a plant leaf, a cable). *Snap* follows edges.
- Optional better cutout: `pip install -r requirements-optional.txt`. A small trained model (about 44 MB, downloaded on first use to `~/.cache/flatlay`, or `FLATLAY_MODEL_DIR`) replaces GrabCut when it is available. Set `FLATLAY_NO_MODEL=1` to disable it and `FLATLAY_SEGMENTER=u2netp` or `u2net` to choose another. It can miss parts of an object, so check the cutout.

### Not enough room?

*Canvas → Add surface* grows the photo on any side by re-tiling the photo's own surface texture (cloth, wood, paper) along minimum-error seams, and carries the lighting gradient across. It takes about a minute on a 5000-pixel-wide photo and can be undone. With `torch` installed (see `requirements-optional.txt`) a trained inpainting network (LaMa, about 200 MB, downloaded on first use to `~/.cache/flatlay`) also supplies the soft layer: colour drift, shading and where larger marks such as cracks lead. Quilting still supplies the fine grain, and the hue is held at the photo's own edge colour. Objects at the edge are not continued into the new area. `FLATLAY_NO_MODEL=1` turns it off. The optical axis stays on the original photo's centre, so side faces remain correct on the enlarged canvas.

## How the render works

| Piece | What it does |
| --- | --- |
| Cutout | GrabCut from a box guessed from the photo border, refined near the outline at higher resolution. |
| Scale | Length in mm and a pixels-per-mm reference give the object's pixel size and, from its thickness, its pixel thickness. |
| Side face | The footprint on the surface is the front face scaled toward the optical axis by thickness / camera height. The band swept between the two is the side face, shaded by depth using a material profile and the local surface colour. |
| Shadow | The footprint is swept along the shadow direction, blurred into a core and a penumbra, and multiplied onto the surface so its texture shows through. A tight contact shadow runs around the base. |
| Tone | The object's blacks are lifted to the scene's own black point; sheen and rim light come from the light direction and the surface colour. |
| Surface extension | Illumination (low frequency) and texture (high frequency) are separated; texture is re-tiled by image quilting and multiplied by the carried-over illumination. |

All parameters are in full-resolution pixels, so the preview and the export produce the same picture.

## Limits

- Reflections on glossy faces come from the object's original photo, not from the new scene.
- The scene photo must be head on. Oblique shots, and objects standing on a floor in perspective, are out of scope.
- Light estimates are heuristics; check the dial. Rounded-object parallax is approximate. Glass and mirrors are placeholders.
- The cutout works best when the object differs clearly from its background in the object photo. Cast shadows that are as dark as the object are the usual failure; tighten the box.
- Camera height is a single number estimated by eye. A good value is the one at which the side faces of objects already in the scene match the ones the new object shows.
- Device dimensions in `data/devices.json` are published figures and are editable. Add entries for anything else you photograph.
- Surface extension needs a clean area of the surface to copy from, and works best on fine, even textures.

## Project layout

```
app/server.py        Flask API
app/static/          browser UI (plain HTML, CSS, JS; no build step)
flatlay/cutout.py    object cutout
flatlay/scene.py     measurements taken from the scene photo
flatlay/render.py    perspective, shadows, tone, compositing
flatlay/surface.py   surface extension by image quilting
flatlay/lighting.py  light direction from scene and object
flatlay/segmenter.py optional trained cutout model
flatlay/session.py   state for one editing session
data/devices.json    device sizes and reference objects
tests/               pytest suite
```

## Using the library directly

```python
import cv2
from flatlay.session import Session

s = Session(cv2.imread("scene.jpg"), cv2.imread("object.jpg"))
p = s.defaults()
p.cx, p.cy, p.height_px = 3000, 1800, 1500
p.length_mm, p.thickness_mm = 90, 6.9
cv2.imwrite("out.jpg", s.render_full(p))
```

## Tests

```
pip install -r requirements-dev.txt
python -m pytest
```
