#!/usr/bin/env python3
"""Genjutsu on Evolink: swap the person in a real clip for a character, keeping the scene, the motion and the lip sync.

Port of sirioberati/Genjustsu-Open-Source-Workflow (MIT). Same recipe, different plumbing:
  - the person becomes a colored depth silhouette (identity gone, pose kept), background stays real
  - optional face mesh drawn on top (lip / eye / brow guide)
  - vocals isolated (no music bed) and pitched +3 semitones, embedded in the video for lip sync
  - Seedance 2.5 video-edit on Evolink (not Enhancor), 480p draft first, 1080p only on a picked run
  - the ORIGINAL soundtrack goes back on at the end, generated audio is thrown away
Everything before the Evolink call runs locally and free (Depth Anything V2, MediaPipe mask + face mesh, Demucs, Rubber Band).

  gj.py prep SOURCE JOB [--start S] [--len L] [--mode depth_mesh|depth|mesh] [--pitch 3]
  gj.py ref JOB NAME PROMPT_FILE                 Nano Banana 2.1 character reference (text only)
  gj.py submit JOB --ref IMG [--ref IMG2] --character "..." [--name r] [--audio]   one run at a time
  gj.py fetch JOB [--wait 900]                   poll, download, conform, put the original audio back, build compare video
  gj.py hd JOB RUN                               1080p of a picked draft (seedance-2.5-draft-to-video, within 24 h)
  gj.py grade JOB RUN                            color-match a finished swap back to the source's light -> runs/RUN-graded.mp4
  gj.py check JOB RUN                            lip-sync score: mouth opening original vs result (correlation, lag)
  gj.py status JOB

Run with this folder's venv: .venv/bin/python gj.py ...
Needs EVOLINK_API_KEY in this folder's .env (the studio writes it) or the environment."""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')  # a few torch ops have no Apple-GPU kernel; run those on CPU instead of crashing

ROOT = Path(__file__).resolve().parent
API ='https://api.evolink.ai/v1'
FPS = 24  # Seedance renders at 24 fps; prepping at 24 keeps frames 1:1
PREP_VERSION = 2  # bump when prep output changes; the studio keys its job folders on it
SHARED_ENV = ''


def env():
    """EVOLINK_API_KEY etc. from this folder's .env (the studio writes it), then SHARED_ENV if set. Real env vars win."""
    for path in (ROOT / '.env', Path(os.path.expanduser(SHARED_ENV)) if SHARED_ENV else None):
        if not path or not path.exists(): continue
        for line in open(path):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.replace('export ', '').strip(), v.strip().strip('"').strip("'"))


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def probe(path):
    out = run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,width,height:format=duration', '-of', 'json', str(path)]).stdout
    j = json.loads(out); v = next(s for s in j['streams'] if s['codec_type'] == 'video')
    return v['width'], v['height'], float(j['format']['duration']), any(s['codec_type'] == 'audio' for s in j['streams'])


T0 = time.time()


def say(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg + (f'  (+{time.time() - T0:.0f}s)' if msg.startswith('[') else ''), flush=True)


# ---------------------------------------------------------------- frame IO (ffmpeg pipes; PyAV + cv2 in one process clash on macOS)

def read_frames(path, w, h):
    import numpy as np
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
    n = w * h * 3; frames = []
    while True:
        b = p.stdout.read(n)
        if len(b) < n: break
        frames.append(np.frombuffer(b, np.uint8).reshape(h, w, 3).copy())
    p.wait()
    return frames


def write_video(path, frames, audio=None, seconds=None):
    h, w = frames[0].shape[:2]
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-']
    if audio:
        cmd += ['-i', str(audio), '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '192k', '-af', 'apad']
        cmd += ['-t', f'{seconds or len(frames) / FPS:.4f}']
    cmd += ['-c:v', 'libx264', '-crf', '15', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames: p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait(): sys.exit(f'ffmpeg failed writing {path}')


# ---------------------------------------------------------------- prep

def device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def depth_maps(frames):
    """Depth Anything V2 per frame. Returns float32 inverse depth (bigger = closer)."""
    import numpy as np, torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = os.environ.get('GENJUTSU_DEPTH_MODEL', 'depth-anything/Depth-Anything-V2-Small-hf')  # Small is Apache-2.0; Base/Large are non-commercial
    proc = AutoImageProcessor.from_pretrained(name); model = AutoModelForDepthEstimation.from_pretrained(name).to(device()).eval()
    h, w = frames[0].shape[:2]; out = []
    with torch.no_grad():
        for i in range(0, len(frames), 8):
            batch = [Image.fromarray(f) for f in frames[i:i + 8]]
            x = proc(images=batch, return_tensors='pt').to(device())
            d = model(**x).predicted_depth.unsqueeze(1)
            d = torch.nn.functional.interpolate(d, size=(h, w), mode='bilinear', align_corners=False)[:, 0]  # bicubic has no MPS kernel (CPU fallback was most of prep time)
            out.extend(d.half().cpu().numpy())  # float16: a 720p clip's depth was ~1 GB in float32
            say(f'  depth {min(i + 8, len(frames))}/{len(frames)}')
    del model
    if device() == 'mps': torch.mps.empty_cache()
    return out


FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def skip_frames(spec, n):
    """'0-1.42,6.5-7' (clip seconds) -> set of frame indexes to leave untouched."""
    out = set()
    for part in (spec or '').split(','):
        if '-' in part:
            a, b = (float(x) for x in part.split('-', 1))
            out.update(range(max(0, int(a * FPS)), min(n, int(round(b * FPS)) + 1)))
    return out


def head_masks(marks, hair, shape):
    """Head-only masks from face landmarks: the face oval (covers jaw and beard, which the segmenter calls 'not face'), a touch wider,
    plus the hair touching it. Frames with no face (helmet shut, other shots) get nothing, so armor and props stay untouched.
    The class-based version painted the armor under the chin because red armor reads as skin (Iron Man 2, 10-08)."""
    import cv2, numpy as np
    h, w = shape; out = []
    for mk, hr in zip(marks, hair):
        m = np.zeros((h, w), np.uint8)
        if mk:
            pts = np.array([(mk[k][0] * w, mk[k][1] * h) for k in FACE_OVAL], np.float32)
            c = pts.mean(0); poly = (c + (pts - c) * 1.1).astype(np.int32)
            cv2.fillPoly(m, [poly], 1)
            fw, fh = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
            near = np.zeros_like(m); y0 = int(max(0, pts[:, 1].min() - 0.7 * fh))
            near[y0:int(c[1]), int(max(0, c[0] - fw)):int(min(w, c[0] + fw))] = 1
            hm = ((hr > 0.5) & (near > 0)).astype(np.uint8)
            if hm.any():  # keep only hair blobs that touch the face
                n, lab = cv2.connectedComponents(hm)
                touch = cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0
                for k in range(1, n):
                    if (touch & (lab == k)).any(): m[lab == k] = 1
        out.append(m.astype(np.float32))
    return out


def person_masks(frames, outfit=None, heads=None):
    """MediaPipe selfie-multiclass segmenter -> soft person masks 0..1 (hair, skin, clothes, accessories all count).
    ~20 ms a frame. Low-res (256 px) edges, which is fine: the mask gets grown and Seedance repaints the figure anyway.
    If `outfit` is a dict it gets the body-skin vs clothes pixel counts (classes 2 and 4): covered arms = long sleeves."""
    import cv2, numpy as np, mediapipe as mp
    v = mp.tasks.vision
    path = ROOT / 'models/selfie_multiclass_256x256.tflite'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        run(['curl', '-sfL', '-o', str(path), 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'])
    opts = v.ImageSegmenterOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(path)), running_mode=v.RunningMode.VIDEO,
                                   output_confidence_masks=True, output_category_mask=False)
    h, w = frames[0].shape[:2]; out = []
    with v.ImageSegmenter.create_from_options(opts) as seg:
        for i, f in enumerate(frames):
            r = seg.segment_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)), round(i * 1000 / FPS))
            bg = r.confidence_masks[0].numpy_view().astype(np.float32)
            out.append(cv2.resize(1 - bg, (w, h), interpolation=cv2.INTER_LINEAR))
            if heads is not None: heads.append(cv2.resize(np.squeeze(r.confidence_masks[1].numpy_view()).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR))  # hair
            if outfit is not None:
                outfit['skin'] = outfit.get('skin', 0) + int((r.confidence_masks[2].numpy_view() > 0.5).sum())
                outfit['clothes'] = outfit.get('clothes', 0) + int((r.confidence_masks[4].numpy_view() > 0.5).sum())
    return out


def person_masks_deeplab(frames):
    """torchvision DeepLabV3-ResNet101, COCO 'person' class. Catches fast raised arms and held props that MediaPipe drops."""
    import numpy as np, torch
    from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
    wts = DeepLabV3_ResNet101_Weights.DEFAULT; model = deeplabv3_resnet101(weights=wts).to(device()).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device()).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225], device=device()).view(1, 3, 1, 1)
    h, w = frames[0].shape[:2]; s = 520 / min(h, w); size = (round(h * s), round(w * s)); out = []
    with torch.no_grad():
        for i in range(0, len(frames), 4):
            x = torch.from_numpy(np.stack(frames[i:i + 4])).to(device()).permute(0, 3, 1, 2).float() / 255
            x = torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
            p = model((x - mean) / std)['out'].softmax(1)[:, 15:16]
            out.extend((torch.nn.functional.interpolate(p, size=(h, w), mode='bilinear', align_corners=False)[:, 0] * 255).byte().cpu().numpy())
            if i % 48 == 0: say(f'  deeplab {min(i + 4, len(frames))}/{len(frames)}')
    del model
    if device() == 'mps': torch.mps.empty_cache()
    return out  # uint8 0..255


def clean_mask(soft, grow):
    """Binary mask, small specks dropped, grown a little so loose hair and sleeves sit inside the silhouette."""
    import cv2, numpy as np
    m = (soft > (127 if soft.dtype == np.uint8 else 0.5)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < big * 0.02: m[lab == k] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    core = m.astype(bool)
    if grow: m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return core, m.astype(bool)


def colorize(depth, core, grow, prev):
    """Inferno depth, normalized on the subject only so the body gets the full color range (yellow = near).
    Per-frame percentiles cancel Depth Anything's per-frame scale drift; a light blend with the previous frame calms flicker.
    The body's edge pixels carry blurred background depth (a black outline Seedance copies), so the edge band and the grown ring
    take the nearest interior depth. The body never goes below dark magenta: near-black is reserved for the painted mouth and eyes."""
    import cv2, numpy as np
    depth = depth.astype(np.float32)
    vals = depth[core] if core.any() else depth.ravel()
    lo, hi = np.percentile(vals, 2), np.percentile(vals, 98)
    if grow:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        inner = cv2.erode(core.astype(np.uint8), k).astype(bool)
        if inner.any():
            filled = cv2.dilate(np.where(inner, depth, np.float32(lo)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * grow + 3, 4 * grow + 3)))
            depth = np.where(inner, depth, np.maximum(depth, filled))
    n = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    if prev is not None: n = np.where(core, 0.65 * n + 0.35 * prev, n)  # blending the ring with last frame's background drew a thin dark outline
    rgb = cv2.applyColorMap(((0.3 + 0.7 * n) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    return rgb, n


def face_model():
    path = ROOT / 'models/face_landmarker.task'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        run(['curl', '-sfL', '-o', str(path), 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'])
    return path


def head_box(mask, last):
    """Square crop (x0, y0, side) likely to hold the face: around the last tracked face, else the top of the person mask."""
    import numpy as np
    h, w = mask.shape if mask is not None else (None, None)
    if last is not None:
        (cx, cy), size = last
    elif mask is not None and mask.any():
        ys, xs = np.nonzero(mask); top = ys.min()
        row = min(h - 1, top + max(8, int(0.05 * h))); cols = np.nonzero(mask[row])[0]
        if not len(cols): return None
        size = max(40, (cols.max() - cols.min()) * 1.2); cx = cols.mean(); cy = top + 0.6 * size
    else:
        return None
    return cx, cy, size * 3


def detect_crop(det, frame, box):
    """Run the IMAGE-mode landmarker on an upscaled square crop; landmarks come back in full-frame pixels."""
    import cv2, numpy as np, mediapipe as mp
    h, w = frame.shape[:2]; cx, cy, side = box; side = int(min(side, h, w))
    x0 = int(np.clip(cx - side / 2, 0, w - side)); y0 = int(np.clip(cy - side / 2, 0, h - side))
    crop = cv2.resize(frame[y0:y0 + side, x0:x0 + side], (512, 512), interpolation=cv2.INTER_CUBIC)
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))
    if not r.face_landmarks: return None
    return [(x0 + l.x * side, y0 + l.y * side) for l in r.face_landmarks[0]]


# MediaPipe face-mesh contours (478-point topology)
LIPS_OUT = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_IN = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
BROW_L = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]


def paint_features(im, pts):
    """'features' style: no foreign colors. Lips a darker shade of the face, the mouth opening and eyes near-black, brows dark.
    Reads as a real face's dark mouth and eyes, so Seedance has nothing neon to copy into the character."""
    import cv2, numpy as np
    P = lambda idx: np.array([pts[i] for i in idx], np.int32)
    fh = abs(pts[152][1] - pts[10][1]); t = max(1, round(fh / 45))
    lips = np.zeros(im.shape[:2], np.uint8); cv2.fillPoly(lips, [P(LIPS_OUT)], 1, cv2.LINE_AA)
    im[lips > 0] = (im[lips > 0] * 0.55).astype(np.uint8)
    for poly in (LIPS_IN, EYE_L, EYE_R): cv2.fillPoly(im, [P(poly)], (18, 8, 16), cv2.LINE_AA)
    for brow in (BROW_L, BROW_R): cv2.polylines(im, [P(brow[:5])], False, (40, 16, 30), t + 1, cv2.LINE_AA)


def draw_mesh(frames, base, masks=None, style='features', given=None):
    """MediaPipe face mesh tracked on the ORIGINAL frames, drawn onto `base`. Colors match the original repo:
    cyan wireframe, orange lips, green eyes. Faces too small for the full-frame pass get a second, zoomed look.
    Returns (frames, tracking report, landmarks)."""
    import cv2, numpy as np, mediapipe as mp
    v = mp.tasks.vision; C = v.FaceLandmarksConnections
    conf = dict(num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
    opts = v.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model())), running_mode=v.RunningMode.VIDEO, min_tracking_confidence=0.2, **conf)
    opts2 = v.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model())), running_mode=v.RunningMode.IMAGE, **conf)
    h, w = frames[0].shape[:2]; lw = max(1, round(min(w, h) / 720)); out, marks, missing, zoomed = [], [], [], 0; last, last_i = None, -99
    with v.FaceLandmarker.create_from_options(opts) as det, v.FaceLandmarker.create_from_options(opts2) as det2:
        for i, (f, b) in enumerate(zip(frames, base)):
            im = b.copy()
            if given is not None:  # landmarks from an earlier pass: same points, no second detection
                fpts = [(x * w, y * h) for x, y in given[i]] if given[i] else None
                if fpts is None: missing.append(i); marks.append([]); out.append(im); continue
            else:
                r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)), round(i * 1000 / FPS))
                fpts = [(l.x * w, l.y * h) for l in r.face_landmarks[0]] if r.face_landmarks else None
            if fpts is None:
                box = head_box(masks[i] if masks is not None else None, last if i - last_i <= 12 else None)
                fpts = detect_crop(det2, f, box) if box else None
                if fpts is None and box and last is not None and masks is not None:
                    box = head_box(masks[i], None); fpts = detect_crop(det2, f, box) if box else None
                zoomed += fpts is not None
            if fpts is None: missing.append(i); marks.append([]); out.append(im); continue
            xs, ys = [p[0] for p in fpts], [p[1] for p in fpts]
            last, last_i = (((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2), max(max(xs) - min(xs), max(ys) - min(ys))), i
            pts = [(round(x), round(y)) for x, y in fpts]
            marks.append([[round(x / w, 5), round(y / h, 5)] for x, y in fpts])
            if style == 'wire':
                for c in C.FACE_LANDMARKS_TESSELATION: cv2.line(im, pts[c.start], pts[c.end], (50, 170, 205), lw, cv2.LINE_AA)
                for name, col in [('FACE_LANDMARKS_LIPS', (255, 135, 100)), ('FACE_LANDMARKS_LEFT_EYE', (130, 255, 150)), ('FACE_LANDMARKS_RIGHT_EYE', (130, 255, 150)),
                                  ('FACE_LANDMARKS_LEFT_EYEBROW', (130, 255, 150)), ('FACE_LANDMARKS_RIGHT_EYEBROW', (130, 255, 150))]:
                    for c in getattr(C, name): cv2.line(im, pts[c.start], pts[c.end], col, lw * 2, cv2.LINE_AA)
            else:
                paint_features(im, pts)
                if style == 'lips':  # one neon cue where it matters most: the lip contour
                    fh = abs(pts[152][1] - pts[10][1]); t = max(1, round(fh / 60))
                    cv2.polylines(im, [np.array([pts[i] for i in LIPS_OUT], np.int32)], True, (40, 200, 255), t + 1, cv2.LINE_AA)  # cyan: orange vanished on the orange heat map
            out.append(im)
    return out, {'frames': len(frames), 'tracked': len(frames) - len(missing), 'zoomed': zoomed, 'missing_frames': missing}, marks


def separate_vocals(job, pitch):
    """Demucs htdemucs -> vocals only (no music bed), then Rubber Band pitch shift, same length as the clip."""
    import numpy as np, soundfile as sf
    src = job / 'original-audio.wav'
    run([sys.executable, '-m', 'demucs', '--two-stems', 'vocals', '-n', 'htdemucs', '-d', 'cpu', '-o', str(job / 'demucs'), str(src)])
    vocals = job / 'demucs/htdemucs/original-audio/vocals.wav'
    shutil.copy(vocals, job / 'vocals.wav')
    if not pitch: return job / 'vocals.wav'
    out = job / f'vocals-pitch{pitch:+d}.wav'
    run(['rubberband', '--fine', '-p', str(pitch), '-t', '1', str(vocals), str(out)])
    a, sr = sf.read(out); n = sf.info(str(src)).frames
    a = np.pad(a, ((0, max(0, n - len(a))), (0, 0)))[:n] if a.ndim == 2 else np.pad(a, (0, max(0, n - len(a))))[:n]
    sf.write(out, a, sr)
    return out


def contact_sheet(rows, path, cols=6):
    import numpy as np
    from PIL import Image
    n = len(rows[0]); idx = [round(i * (n - 1) / (cols - 1)) for i in range(cols)]
    h, w = rows[0][0].shape[:2]; s = 300 / h
    tiles = [np.concatenate([np.asarray(Image.fromarray(r[i]).resize((round(w * s), 300))) for i in idx], axis=1) for r in rows]
    Image.fromarray(np.concatenate(tiles, axis=0)).save(path, quality=88)


def prep(o):
    import numpy as np
    job = Path(o.job).resolve(); job.mkdir(parents=True, exist_ok=True)
    src = Path(o.source).resolve(); W, H, D, has_audio = probe(src)
    if not has_audio: sys.exit('source has no audio track — lip sync needs the vocals')
    length = o.len or (D - o.start)
    if not 4 <= length <= 30: sys.exit(f'clip is {length:.2f}s; Seedance video-edit takes 4–30 s')
    say(f'[1/6] cut {o.start:.2f}–{o.start + length:.2f}s, 24 fps, short side {o.short}px')
    scale = f"scale='if(gt(iw,ih),-2,{o.short})':'if(gt(iw,ih),{o.short},-2)':flags=lanczos"
    run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{o.start}', '-i', str(src), '-t', f'{length}', '-vf', f'fps={FPS},{scale}', '-c:v', 'libx264', '-crf', '14',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '256k', str(job / 'original.mp4')])
    run(['ffmpeg', '-v', 'error', '-y', '-i', str(job / 'original.mp4'), '-vn', '-ac', '2', '-ar', '44100', str(job / 'original-audio.wav')])
    w, h, _, _ = probe(job / 'original.mp4'); frames = read_frames(job / 'original.mp4', w, h); seconds = len(frames) / FPS
    say(f'      {len(frames)} frames {w}x{h}')

    say('[2/6] vocals: Demucs, no music bed' + (f', pitch {o.pitch:+d} semitones' if o.pitch else ''))
    vocals = separate_vocals(job, o.pitch)

    report = {'source': str(src), 'start': o.start, 'length': seconds, 'size': [w, h], 'fps': FPS, 'mode': o.mode, 'swap': o.swap, 'mesh_style': o.mesh_style, 'prep_version': PREP_VERSION, 'mask_model': o.mask, 'pitch': o.pitch, 'vocals': vocals.name}
    if o.mode in ('depth', 'depth_mesh'):
        say(f'[3/6] person mask: {"head only" if o.swap == "head" else o.mask}'); outfit = {}
        if o.swap == 'head':  # movie scenes: keep the costume/armor, swap the head
            hair = []; person_masks(frames, None, hair)
            _, head_track, head_marks = draw_mesh(frames, frames, None, o.mesh_style)
            soft = head_masks(head_marks, hair, frames[0].shape[:2])
        else:
            if True:
                mpm = [(m * 255).astype(np.uint8) for m in person_masks(frames, outfit)]  # uint8 keeps a 720p clip's masks ~4x smaller
                soft = [np.maximum(a, b) for a, b in zip(mpm, person_masks_deeplab(frames))]; del mpm
        if outfit.get('clothes'):
            # Bare body skin (neck, hands, arms) as a share of skin+clothes. Measured 10-08: sweatshirt 14%, t-shirt 28%.
            outfit['skin_share'] = round(outfit['skin'] / (outfit['skin'] + outfit['clothes']), 3)
            outfit['sleeves'] = 'long' if outfit['skin_share'] < 0.20 else 'short'
            report['outfit'] = outfit; say(f"      outfit: {outfit['sleeves']} sleeves (bare skin {outfit['skin_share']:.0%} of body)")
        grow = round(min(w, h) * o.grow); pairs = [clean_mask(s, grow) for s in soft]; del soft
        skip = skip_frames(o.skip, len(frames))
        if skip:  # shots of other people (dancers, crowd): no figure, no guide, original pixels
            empty = np.zeros(frames[0].shape[:2], bool)
            pairs = [(empty, empty) if i in skip else pr for i, pr in enumerate(pairs)]; report['skip'] = o.skip
        masks = [m for _, m in pairs]
        cover = [float(m.mean()) for m in masks]
        report['mask'] = {'grow_px': grow, 'coverage_min': min(cover), 'coverage_max': max(cover), 'empty_frames': [i for i, c in enumerate(cover) if c < 0.002]}
        say('[4/6] depth: Depth Anything V2'); depth = depth_maps(frames)
        comp, prev = [], None
        for f, d, (core, m) in zip(frames, depth, pairs):
            rgb, prev = colorize(d, core, grow, prev)
            comp.append(np.where(m[:, :, None], rgb, f).astype(np.uint8))
        write_video(job / 'mask.mp4', [np.repeat(m[:, :, None].astype(np.uint8) * 255, 3, axis=2) for m in masks])
    else:
        say('[3/6] mesh only: keeping the original video'); comp = frames; masks = None
    if o.mode in ('mesh', 'depth_mesh'):
        say('[5/6] face mesh: MediaPipe')
        comp, track, marks = draw_mesh(frames, comp, masks if o.mode == 'depth_mesh' else None, o.mesh_style,
                                       given=head_marks if o.swap == 'head' and o.mode != 'mesh' else None)
        for i in skip_frames(o.skip, len(frames)): comp[i] = frames[i]  # no face guide on skipped shots either
        report['face'] = track; (job / 'face-landmarks.json').write_text(json.dumps({'fps': FPS, 'frames': marks}))
        if track['missing_frames']: say(f"      no face on {len(track['missing_frames'])} of {track['frames']} frames (mesh skipped there)")
    say('[6/6] encode Seedance input with the vocals embedded')
    write_video(job / 'input.mp4', comp, audio=vocals, seconds=seconds)
    contact_sheet([frames, comp], job / 'review.jpg')
    report['made'] = time.strftime('%Y-%m-%d %H:%M:%S')
    (job / 'manifest.json').write_text(json.dumps(report, indent=2))
    say(f'done -> {job}/input.mp4  (review.jpg = original over input)')


# ---------------------------------------------------------------- Evolink

def api(method, path, body=None):
    cmd = ['curl', '-s', '-X', method, API + path, '-H', 'Authorization: Bearer ' + os.environ['EVOLINK_API_KEY'], '-H', 'Content-Type: application/json']
    if body is not None: cmd += ['-d', json.dumps(body)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def upload(job, f):
    """Public URL for a local file. Default: Evolink's file service (same API key, files expire after 72 h).
    GENJUTSU_UPLOAD=r2 uses your own Cloudflare R2 (R2_* vars). Cached per file hash; refuses a URL that doesn't answer 200
    (a run submitted with a dead URL still bills)."""
    f = Path(f); digest = hashlib.sha1(f.read_bytes()).hexdigest()[:16]
    cache_path = job / 'uploads.json'; cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    hit = cache.get(digest)
    if isinstance(hit, str): return hit  # older R2 entries: permanent
    if hit and hit.get('expires', 0) > time.time() + 3600: return hit['url']
    if os.environ.get('GENJUTSU_UPLOAD') == 'r2':
        e = dict(os.environ, AWS_ACCESS_KEY_ID=os.environ['R2_ACCESS_KEY'], AWS_SECRET_ACCESS_KEY=os.environ['R2_SECRET_KEY'], AWS_DEFAULT_REGION='auto')
        ct = {'.mp4': 'video/mp4', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}.get(f.suffix.lower(), 'application/octet-stream')
        key = f"genjutsu/{int(time.time())}-{digest}-{f.name.replace(' ', '_')}"
        run(['aws', 's3', 'cp', str(f), f"s3://{os.environ['R2_BUCKET']}/{key}", '--endpoint-url', f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com", '--content-type', ct], env=e)
        url, expires = f"{os.environ['R2_PUBLIC_URL']}/{key}", time.time() + 10 * 365 * 86400
    else:
        out = subprocess.run(['curl', '-s', '-X', 'POST', 'https://files-api.evolink.ai/api/v1/files/upload/stream',
                              '-H', 'Authorization: Bearer ' + os.environ['EVOLINK_API_KEY'], '-F', f'file=@{f}'], capture_output=True, text=True).stdout
        try: d = json.loads(out)['data']; url = d['file_url']
        except Exception: sys.exit('Evolink upload failed: ' + out[:300])
        expires = time.time() + 70 * 3600
    code = run(['curl', '-sI', '-A', 'Mozilla/5.0', '-o', '/dev/null', '-w', '%{http_code}', url]).stdout
    if code != '200': sys.exit(f'upload not reachable ({code}): {url}')
    cache[digest] = {'url': url, 'expires': expires}; cache_path.write_text(json.dumps(cache, indent=2))
    return url


def small_ref(job, img):
    """Evolink times out fetching heavy refs; send a ≤1536px JPEG."""
    from PIL import Image
    im = Image.open(img).convert('RGB'); im.thumbnail((1536, 1536))
    out = job / 'refs' / (Path(img).stem + '.jpg'); out.parent.mkdir(exist_ok=True); im.save(out, quality=90)
    return out


HEAD_LOCK = """HEAD LOCK, the most important rule: the character's head never changes. Face, skin tone, hair, facial hair and headwear come only from {refs} and stay identical in every frame. Any hat is worn exactly as in {refs}: same hat, same direction (a cap that faces forward in {refs} faces forward in every frame, with its front logo or text visible), same fit. Do not copy the performer's hat, hairstyle or headwear from @video1: the head shape in @video1 only says where the head is and how it moves. If the performer's hat sticks out somewhere the character's hat does not, show the background there instead."""


PROMPT = """Edit @video1. Replace the person in @video1 with the character from {refs}, in every frame, as the same performer doing the exact same performance.

How to read @video1: the person's body is painted as a colored depth map (bright yellow = closer to camera, dark purple = farther){mesh}. These colors and lines are ONLY a motion guide. None of them may appear in the result: no depth colors, no mesh lines, no outline or glow around the body.

The character: {character}. Take the face, hair, skin, outfit and every design detail from {refs}. Replace the WHOLE figure from head to toe, including hair, hands and clothing, even where the depth shape is rough at its edges.

HEAD LOCK, the most important rule: the character's head never changes. Face, skin tone, hair, facial hair and headwear come only from {refs} and stay identical in every frame. Any hat is worn exactly as in {refs}: same hat, same direction (a cap that faces forward in {refs} faces forward in every frame, with its front logo or text visible), same fit. Do not copy the performer's hat, hairstyle or headwear from @video1: the head shape in @video1 only says where the head is and how it moves. If the performer's hat sticks out somewhere the character's hat does not, show the background there instead.

Keep exactly from @video1: the background, the camera framing and camera movement, the lighting direction, and the body pose, gestures, head turns and timing, frame by frame. Everything outside the figure stays exactly as it is in @video1. Keep the color of the light exactly as in @video1: strongly colored light (yellow, pink, blue rooms, neon) stays just as strong on the walls AND on the character; never white-balance, neutralize or brighten the scene.

Lip sync: the audio in @video1 is this performer's vocals. The character raps these exact vocals: the mouth opens and closes on every syllable at exactly the same moments as the {mouth} in @video1. Eyes, brows and expression follow the performer.

No extra people, no cuts, no slow motion, no zoom, no text, no logos."""

PROMPT_MESH_ONLY = """Edit @video1. Replace the person in @video1 with the character from {refs}, in every frame, as the same performer doing the exact same performance.

How to read @video1: a thin face mesh is drawn over the person's face (cyan wireframe, orange lips, green eyes and brows). The mesh is ONLY a motion guide and must not appear in the result.

The character: {character}. Replace the ENTIRE identity of the person with {refs}: face, hair, skin and outfit. Nothing of the original person's face may remain.

HEAD LOCK, the most important rule: the character's head never changes. Face, skin tone, hair, facial hair and headwear come only from {refs} and stay identical in every frame. Any hat is worn exactly as in {refs}: same hat, same direction (a cap that faces forward in {refs} faces forward in every frame, with its front logo or text visible), same fit. Do not copy the performer's hat, hairstyle or headwear from @video1: the head shape in @video1 only says where the head is and how it moves. If the performer's hat sticks out somewhere the character's hat does not, show the background there instead.

Keep exactly from @video1: the background, the camera framing and camera movement, the lighting, and the body pose, gestures, head turns and timing, frame by frame.

Lip sync: the audio in @video1 is this performer's vocals. The character raps these exact vocals: the mouth opens and closes on every syllable at exactly the same moments as the orange lip lines in @video1.

No extra people, no cuts, no slow motion, no zoom, no text, no logos."""


OUTFIT_LONG = """

Clothing: the performer in @video1 wears loose, long-sleeved clothing, so the arms and torso of the depth shape are mostly fabric, not body. Dress the character to match: a loose long-sleeve version of their top from {refs}, same color and style (if their top has short sleeves, make it long-sleeved), so the arms read as sleeves and the character keeps their own natural build. Never fill the shape with bare skin or muscle. This changes clothes only; the head lock above still holds."""


PROMPT_HEAD = """Edit @video1. Swap ONLY the head of the person in @video1 for the head of the character from {refs}, in every frame where that head is visible, as the same performer doing the exact same performance.

How to read @video1: only the person's head (face, hair and neck) is painted as a colored depth map (bright yellow = closer to camera, dark purple = farther){mesh}. These colors and lines are ONLY a motion guide. None of them may appear in the result.

The character's head: {character} Use only their head from {refs}; ignore any clothing in that description.

{head}

Keep everything else exactly as it is in @video1: the body, the armor, suit or costume and every piece of it, the hands, every other person and object, the background, the camera framing and movement, the lighting and the timing, frame by frame. Keep the color of the light exactly as in @video1: strongly colored light (yellow, pink, blue rooms, neon) stays just as strong on the walls AND on the character; never white-balance, neutralize or brighten the scene. When the head is covered (a helmet closing over it, turned away), show exactly what @video1 shows.

Lip sync: the audio in @video1 is this performer's voice. The character's mouth opens and closes on every syllable at exactly the same moments as the {mouth} in @video1.

No extra people, no cuts, no slow motion, no zoom, no text, no logos."""


def build_prompt(job, character, nrefs, match_outfit=True):
    man = json.loads((job / 'manifest.json').read_text()); mode = man['mode']; style = man.get('mesh_style', 'wire')
    refs = '@image1' if nrefs == 1 else ' and '.join(f'@image{i + 1}' for i in range(nrefs))
    long_sleeves = match_outfit and man.get('swap') != 'head' and (man.get('outfit') or {}).get('sleeves') == 'long'
    if long_sleeves:  # a "t-shirt" in the character text would fight the sleeve instruction
        import re
        character = re.sub(r'\b(t-?shirt|tee|tank top)\b', 'long-sleeve shirt', character, flags=re.I)
    if man.get('swap') == 'head':
        mesh, mouth = _guide_words(mode, style)
        head = HEAD_LOCK.format(refs=refs)
        return PROMPT_HEAD.format(refs=refs, character=character.rstrip('.') + '.', mesh=mesh, mouth=mouth, head=head)
    p = _prompt(man, mode, style, refs, character)
    if man.get('skip'):
        head, tail = p.rsplit('\n\nNo extra people', 1)
        p = head + "\n\nShots with no colored figure in @video1 show other people, not the performer: leave those shots and those people exactly as they are." + '\n\nNo extra people' + tail
    if long_sleeves:
        head, tail = p.rsplit('\n\nNo extra people', 1)
        p = head + OUTFIT_LONG.format(refs=refs) + '\n\nNo extra people' + tail
    return p


def _guide_words(mode, style):
    if mode == 'depth_mesh' and style in ('features', 'lips'):
        lip = ', a thin cyan line tracing the lips' if style == 'lips' else ''
        return (f', and the face shows only a dark open mouth{lip}, dark eye shapes and dark brows marking where they are and how they move. '
                "In the result these become the character's own real lips, teeth, tongue, eyes and brows, with the character's own colors"), \
               ('cyan lip line and dark mouth opening' if style == 'lips' else 'dark mouth opening')
    if mode == 'depth_mesh':
        return ', and a thin face mesh is drawn on the face (cyan wireframe, orange lips, green eyes and brows)', 'orange lip lines'
    return '', "performer's mouth"


def _prompt(man, mode, style, refs, character):
    if mode == 'mesh': return PROMPT_MESH_ONLY.format(refs=refs, character=character)
    if mode == 'depth_mesh' and style in ('features', 'lips'):
        lip = ', a thin cyan line tracing the lips' if style == 'lips' else ''
        mesh, mouth = (f', and the face shows only a dark open mouth{lip}, dark eye shapes and dark brows marking where they are and how they move. '
                       "In the result these become the character's own real lips, teeth, tongue, eyes and brows, with the character's own colors"), \
                      ('cyan lip line and dark mouth opening' if style == 'lips' else 'dark mouth opening')
    elif mode == 'depth_mesh':
        mesh, mouth = ', and a thin face mesh is drawn on the face (cyan wireframe, orange lips, green eyes and brows)', 'orange lip lines'
    else:
        mesh, mouth = '', "performer's mouth"
    return PROMPT.format(refs=refs, character=character, mesh=mesh, mouth=mouth)


def ledger(job, entry):
    with open(job / 'ledger.jsonl', 'a') as f: f.write(json.dumps({'t': time.strftime('%Y-%m-%d %H:%M:%S'), **entry}) + '\n')


def submit(o):
    job = Path(o.job).resolve(); runs = job / 'runs'; runs.mkdir(exist_ok=True)
    video = upload(job, job / 'input.mp4'); refs = [upload(job, small_ref(job, r)) for r in o.ref]
    prompt = open(o.prompt_file).read() if o.prompt_file else build_prompt(job, o.character, len(refs), not o.no_match_outfit)
    body = {'model': 'seedance-2.5-video-edit', 'video_urls': [video], 'image_urls': refs, 'draft': True, 'quality': o.quality,
            'generate_audio': o.audio, 'prompt': prompt}
    if o.relaxed: body['content_filter'] = False  # Evolink's own relaxed channel, billed 1.1x; illegal content still enforced
    (job / 'last-prompt.txt').write_text(prompt)
    for i in range(o.runs):
        name = f'{o.name}{"abcdefghij"[i]}'
        if (runs / f'{name}.json').exists(): sys.exit(f'{name} already exists — pick a new --name, never resubmit over a run')
        resp = api('POST', '/videos/generations', body)
        json.dump({'body': body, 'resp': resp}, open(runs / f'{name}.json', 'w'), indent=1)
        try: r = json.loads(resp); say(name, r.get('id'), 'reserved', (r.get('usage') or {}).get('credits_reserved', r.get('credits_reserved')))
        except Exception: say(name, 'BAD RESPONSE', resp[:300])
        ledger(job, {'run': name, 'kind': 'draft', 'resp': resp[:400]})


def conform(job, raw, out, shift=0):
    """Stretch evenly to the clip length (Seedance comes back a few % off) and copy the ORIGINAL audio under it.
    shift = frames to pull the picture earlier (Seedance lips land ~2 frames late); the last frame is held to keep the length."""
    L = probe(job / 'original.mp4')[2]; D = probe(raw)[2]
    vf = f'setpts=PTS*{L}/{D},fps={FPS}' + (f',trim=start_frame={shift},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop={shift}' if shift else '')
    final, out = Path(out), Path(out).with_suffix('.part.mp4')
    run(['ffmpeg', '-v', 'error', '-y', '-i', str(raw), '-i', str(job / 'original.mp4'), '-filter:v', vf,
         '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-t', f'{L}', '-movflags', '+faststart', str(out)])
    os.replace(out, final)
    return D, L


def compare(job, final, out):
    w, h, _, _ = probe(final); hh = 720 if h >= 720 else h
    dest, out = Path(out), Path(out).with_suffix('.part.mp4')
    run(['ffmpeg', '-v', 'error', '-y', '-i', str(job / 'original.mp4'), '-i', str(job / 'input.mp4'), '-i', str(final), '-filter_complex',
         f'[0:v]scale=-2:{hh}[a];[1:v]scale=-2:{hh}[b];[2:v]scale=-2:{hh}[c];[a][b][c]hstack=3[v]', '-map', '[v]', '-map', '0:a',
         '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-movflags', '+faststart', str(out)])
    os.replace(out, dest)


def fetch(o):
    job = Path(o.job).resolve(); t0 = time.time()
    while True:
        pending = 0
        for p in sorted(glob.glob(str(job / 'runs/*.json'))):
            if p.endswith('-sync.json'): continue  # lip-sync scores, not runs
            j = json.load(open(p)); name = Path(p).stem
            if j.get('done'): continue
            try: tid = json.loads(j['resp'])['id']
            except Exception: say(name, 'no task id:', j['resp'][:200]); j['done'] = True; json.dump(j, open(p, 'w'), indent=1); continue
            r = json.loads(api('GET', '/tasks/' + tid) or '{}'); st = r.get('status')
            if st == 'completed':
                u = (r.get('results') or [None])[0]
                raw = job / 'runs' / f'{name}.mp4'
                run(['curl', '-sfL', '-A', 'Mozilla/5.0', '-o', str(raw), u])  # results expire in 24 h; urllib gets 403
                res = finish(job, name, raw)
                say(f"{name} DONE  raw {res['raw_seconds']}s -> {res['clip_seconds']}s  lip sync {res['mouth_corr']} (shift {res['shift_applied']} frames)  runs/{name}-final.mp4  runs/{name}-compare.mp4")
                j.update(done=True, final=r); json.dump(j, open(p, 'w'), indent=1); ledger(job, {'run': name, 'status': 'completed', 'usage': r.get('usage')})
            elif st == 'failed':
                say(name, 'FAILED', json.dumps(r.get('error'))[:400]); j.update(done=True, final=r); json.dump(j, open(p, 'w'), indent=1)
                ledger(job, {'run': name, 'status': 'failed', 'error': r.get('error'), 'usage': r.get('usage')})
            else: pending += 1; say(name, st, r.get('progress', ''))
        if not pending or time.time() - t0 > o.wait: break
        time.sleep(20)
    say('pending:', pending)


def hd(o):
    job = Path(o.job).resolve(); src = job / 'runs' / f'{o.run}.json'
    tid = json.loads(json.load(open(src))['resp'])['id']; name = f'{o.run}-hd'
    if (job / 'runs' / f'{name}.json').exists(): sys.exit(f'{name} already submitted — run fetch')
    body = {'model': 'seedance-2.5-draft-to-video', 'source_task_id': tid}  # only these fields; must be within 24 h of the draft
    resp = api('POST', '/videos/generations', body); json.dump({'body': body, 'resp': resp}, open(job / 'runs' / f'{name}.json', 'w'), indent=1)
    ledger(job, {'run': name, 'kind': 'hd', 'resp': resp[:400]}); say(name, resp[:200])


def ref(o):
    job = Path(o.job).resolve(); (job / 'refs').mkdir(parents=True, exist_ok=True)
    body = {'model': 'gemini-nano-banana-2.1', 'prompt': open(o.prompt_file).read(), 'size': o.size, 'quality': '2K'}
    r = json.loads(api('POST', '/images/generations', body)); tid = r.get('id')
    if not tid: sys.exit('ref gen rejected: ' + json.dumps(r)[:300])
    for _ in range(60):
        t = json.loads(api('GET', '/tasks/' + tid))
        if t.get('status') == 'completed':
            u = t['results'][0]; out = job / 'refs' / f'{o.name}{Path(u.split("?")[0]).suffix or ".png"}'
            run(['curl', '-sfL', '-A', 'Mozilla/5.0', '-o', str(out), u]); ledger(job, {'ref': o.name, 'usage': t.get('usage')}); say(out); return
        if t.get('status') == 'failed': sys.exit('ref gen failed: ' + json.dumps(t.get('error')))
        time.sleep(5)
    sys.exit('ref gen still pending: ' + tid)


def mouth_series(path):
    """Inner-lip gap / face height per frame (NaN where no face). Uses the same zoomed fallback as prep."""
    import numpy as np, mediapipe as mp
    v = mp.tasks.vision
    w, h, _, _ = probe(path); frames = read_frames(path, w, h)
    opts = v.FaceLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(face_model())), running_mode=v.RunningMode.IMAGE,
                                   num_faces=1, min_face_detection_confidence=0.2, min_face_presence_confidence=0.2)
    out, last = [], None
    with v.FaceLandmarker.create_from_options(opts) as det:
        for f in frames:
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
            pts = [(l.x * w, l.y * h) for l in r.face_landmarks[0]] if r.face_landmarks else None
            if pts is None and last is not None: pts = detect_crop(det, f, (last[0], last[1], last[2] * 3))
            if pts is None: out.append(np.nan); continue
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]; last = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(max(xs) - min(xs), max(ys) - min(ys)))
            out.append(abs(pts[14][1] - pts[13][1]) / max(1e-6, abs(pts[152][1] - pts[10][1])))
    return np.array(out)


def sync_score(job, final, a=None):
    """Correlation of mouth opening, original vs result, at the best lag within ±6 frames (positive lag = result is late)."""
    import numpy as np
    a = mouth_series(job / 'original.mp4') if a is None else a; b = mouth_series(final)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]; best = None
    for lag in range(-6, 7):
        x, y = (a[max(0, -lag):n - max(0, lag)], b[max(0, lag):n - max(0, -lag)])
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() > 24:
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            if best is None or r > best[0]: best = (r, lag, int(ok.sum()))
    return {'frames': n, 'face_in_original': int((~np.isnan(a)).sum()), 'face_in_result': int((~np.isnan(b)).sum()),
            'mouth_corr': round(best[0], 3) if best else None, 'best_lag_frames': best[1] if best else None, 'frames_compared': best[2] if best else 0}, a


def finish(job, name, raw):
    """Conform + original audio, measure lip sync, and if the lips are consistently 1-3 frames late, re-conform pulled earlier."""
    final, comp = job / 'runs' / f'{name}-final.mp4', job / 'runs' / f'{name}-compare.mp4'
    D, L = conform(job, raw, final); res, a = sync_score(job, final); res['shift_applied'] = 0
    if res['mouth_corr'] is not None and res['mouth_corr'] >= 0.3 and 1 <= res['best_lag_frames'] <= 3:
        conform(job, raw, final, shift=res['best_lag_frames']); res['shift_applied'] = res['best_lag_frames']
        res['after_shift'] = sync_score(job, final, a)[0]
    if (job / 'mask.mp4').exists():  # Seedance flattens strong colored light; grade it back to the source (ungraded copy kept)
        ungraded = job / 'runs' / f'{name}-ungraded.mp4'; os.replace(final, ungraded)
        grade(job, ungraded, final); res['graded'] = True
    compare(job, final, comp); res.update(run=name, raw_seconds=round(D, 3), clip_seconds=round(L, 3))
    (job / 'runs' / f'{name}-sync.json').write_text(json.dumps(res, indent=2))
    return res


def grade(job, video, out):
    """Color-match a finished swap back to the source, frame by frame: LAB mean/std of the BACKGROUND (outside the person mask) in the
    original vs the result, applied to the whole result frame, so the scene's light (and the light on the character) comes back.
    Seedance flattens strongly colored light: Hotline Bling's yellow room came back beige twice, prompt or not (10-08)."""
    import cv2, numpy as np
    w, h, _, _ = probe(video); ow, oh, _, _ = probe(job / 'original.mp4')
    res = read_frames(video, w, h); org = read_frames(job / 'original.mp4', ow, oh)
    mw, mh, _, _ = probe(job / 'mask.mp4') if (job / 'mask.mp4').exists() else (ow, oh, 0, 0)
    masks = read_frames(job / 'mask.mp4', mw, mh) if (job / 'mask.mp4').exists() else [np.zeros((oh, ow, 3), np.uint8)] * len(org)
    n = min(len(res), len(org), len(masks)); k = np.ones((25, 25), np.uint8); stats = []
    for i in range(n):
        o = cv2.cvtColor(cv2.resize(org[i], (w, h)), cv2.COLOR_RGB2LAB).astype(np.float32)
        r = cv2.cvtColor(res[i], cv2.COLOR_RGB2LAB).astype(np.float32)
        bg = cv2.erode((cv2.resize(masks[i][:, :, 0], (w, h)) < 128).astype(np.uint8), k).astype(bool)  # away from the person's edges
        if bg.mean() < 0.15: bg = np.ones_like(bg)  # tight close-up: use the whole frame
        stats.append([o[bg].mean(0), o[bg].std(0) + 1e-3, r[bg].mean(0), r[bg].std(0) + 1e-3])
    st = np.array(stats)  # (n, 4, 3)
    sm = st.copy()  # smooth over 5 frames, never across a cut (big jump in the source's mean color)
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        win = [j for j in range(lo, hi) if np.abs(st[j, 0] - st[i, 0]).max() < 12]
        sm[i] = np.median(st[win], axis=0)
    graded = []
    for i in range(n):
        mo, so, mr, sr = sm[i]
        r = cv2.cvtColor(res[i], cv2.COLOR_RGB2LAB).astype(np.float32)
        scale = np.clip(so / sr, 0.7, 1.5); scale[0] = 1.0  # L: shift the level only; scaling it by the flat wall's spread killed contrast
        g = (r - mr) * scale + mo
        graded.append(cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    tmp = Path(out).with_suffix('.part.mp4')
    write_video(tmp, graded, audio=video, seconds=n / FPS); os.replace(tmp, out)
    return out


def check(o):
    """Lip-sync score for a finished run: correlation of mouth opening, original vs result."""
    job = Path(o.job).resolve(); res = sync_score(job, job / 'runs' / f'{o.run}-final.mp4')[0]; say(json.dumps(res))


def status(o):
    job = Path(o.job).resolve(); say((job / 'manifest.json').read_text())
    for p in sorted(glob.glob(str(job / 'runs/*.json'))):
        if p.endswith('-sync.json'): continue
        j = json.load(open(p)); f = j.get('final') or {}
        say(Path(p).stem, f.get('status', 'pending'), json.dumps(f.get('error') or '')[:200])


if __name__ == '__main__':
    env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prep'); p.add_argument('source'); p.add_argument('job'); p.add_argument('--start', type=float, default=0); p.add_argument('--len', type=float)
    p.add_argument('--mode', choices=['depth_mesh', 'depth', 'mesh'], default='depth_mesh'); p.add_argument('--pitch', type=int, default=3)
    p.add_argument('--skip', default='', help="clip-second ranges to leave untouched, e.g. '0-1.42' (shots of other people)")
    p.add_argument('--swap', choices=['body', 'head'], default='body', help='body = whole person; head = head only, costume/armor kept (movie scenes)')
    p.add_argument('--mesh-style', choices=['features', 'lips', 'wire'], default='features',
                   help='features = dark mouth/eyes painted on (no neon to leak); lips = features + cyan lip line; wire = original repo neon wireframe')
    p.add_argument('--mask', choices=['union'], default='union', help='MediaPipe + DeepLab person mask'); p.add_argument('--short', type=int, default=720, help='short side in px'); p.add_argument('--grow', type=float, default=0.006, help='mask growth, share of short side (0.012 made baggy clothes read as huge arms)')
    p = sub.add_parser('ref'); p.add_argument('job'); p.add_argument('name'); p.add_argument('prompt_file'); p.add_argument('--size', default='9:16')
    p = sub.add_parser('submit'); p.add_argument('job'); p.add_argument('--ref', action='append', required=True); p.add_argument('--character', default='the character in the reference image')
    p.add_argument('--prompt-file'); p.add_argument('--runs', type=int, default=1); p.add_argument('--name', default='r'); p.add_argument('--quality', default='480p')
    p.add_argument('--no-match-outfit', action='store_true', help="don't dress the character to match the performer's sleeves")
    p.add_argument('--relaxed', action='store_true', help="Evolink's relaxed content filter (content_filter=false), +10% cost")
    p.add_argument('--audio', action='store_true', help='generate_audio true (off by default: Evolink fails the run if it thinks the audio is a copyrighted song)')
    p = sub.add_parser('fetch'); p.add_argument('job'); p.add_argument('--wait', type=int, default=0)
    p = sub.add_parser('hd'); p.add_argument('job'); p.add_argument('run')
    p = sub.add_parser('status'); p.add_argument('job')
    p = sub.add_parser('check'); p.add_argument('job'); p.add_argument('run')
    p = sub.add_parser('grade'); p.add_argument('job'); p.add_argument('run')
    o = ap.parse_args(); {'prep': prep, 'ref': ref, 'submit': submit, 'fetch': fetch, 'hd': hd, 'status': status, 'check': check,
     'grade': lambda o: say(grade(Path(o.job).resolve(), Path(o.job).resolve() / 'runs' / f'{o.run}-final.mp4', Path(o.job).resolve() / 'runs' / f'{o.run}-graded.mp4'))}[o.cmd](o)
