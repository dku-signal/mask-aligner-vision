"""Run opencv / sam2 / hybrid detectors on the same frames and compare.

  python test_detectors.py                  # synthetic frame with known truth
  python test_detectors.py img1.png img2.jpg
  python test_detectors.py --camera 0       # one webcam snapshot
  python test_detectors.py --random 50      # randomized frames: success rate / error / time
Overlays are written to ./out/.
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from make_dataset import render
from sam2_keys import make_detector

OUT = Path(__file__).resolve().parent/'out'


def synthetic(mask_center=(320, 240), wafer_offset=(12.0, -8.0), wafer_angle=6.0, seed=0):
    """Mask: 4 dark squares on a 2x2 grid. Wafer: dark cross (arm = size/3), shifted and rotated."""
    rng = np.random.default_rng(seed)
    h, w = 480, 640
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    img = 175+25*(xx/w)-15*(yy/h)

    wx, wy = mask_center[0]+wafer_offset[0], mask_center[1]+wafer_offset[1]
    size = 80.0
    r = math.radians(wafer_angle)
    u = (xx-wx)*math.cos(r)+(yy-wy)*math.sin(r)
    v = -(xx-wx)*math.sin(r)+(yy-wy)*math.cos(r)
    half, arm = size/2, size/6
    cross = ((abs(u) < arm) & (abs(v) < half)) | ((abs(v) < arm) & (abs(u) < half))
    img[cross] = 95

    side, gap = 22, 24
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = mask_center[0]+sx*gap, mask_center[1]+sy*gap
            img[int(cy-side/2):int(cy+side/2), int(cx-side/2):int(cx+side/2)] = 25

    img += rng.normal(0, 4, img.shape)
    gray = np.clip(img, 0, 255).astype(np.uint8)
    frame = cv2.GaussianBlur(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (3, 3), 0)
    return frame, dict(mask=mask_center, wafer=(wx, wy), dx=wafer_offset[0], dy=wafer_offset[1])


def draw(frame, md, wd, detector, name):
    out = frame.copy()
    if md:
        for q in md['quads']:
            cv2.polylines(out, [np.asarray(q, np.int32)], True, (0, 255, 0), 1)
        cv2.drawMarker(out, tuple(int(c) for c in md['center']), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    if wd:
        x, y, w, h = wd['rect']
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.drawMarker(out, tuple(int(c) for c in wd['center']), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
    for b in getattr(detector, 'boxes', []):
        cv2.rectangle(out, tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:]), (0, 255, 255), 1)
    for s in getattr(detector, 'sam_squares', []):
        cv2.polylines(out, [s['quad'].astype(np.int32)], True, (255, 160, 0), 1)
    cv2.putText(out, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return out


def fmt(p):
    return f'({p[0]:.2f}, {p[1]:.2f})' if p is not None else '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='*')
    ap.add_argument('--camera', type=int)
    ap.add_argument('--detectors', default='opencv,sam2,hybrid')
    ap.add_argument('--random', type=int, default=0, help='evaluate on N randomized frames')
    ap.add_argument('--runs', type=int, default=3, help='timed repeats per frame')
    args = ap.parse_args()

    frames = []
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        for _ in range(10):
            ok, f = cap.read()
        cap.release()
        if not ok:
            raise SystemExit('camera read failed')
        frames.append(('camera', f, None))
    for p in args.images:
        f = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
        if f is None:
            raise SystemExit(f'cannot read {p}')
        if f.shape[1] != 640:
            f = cv2.resize(f, (640, int(f.shape[0]*640/f.shape[1])), interpolation=cv2.INTER_AREA)
        frames.append((Path(p).stem, f, None))
    if not frames:
        frames.append(('synthetic', *synthetic()))

    OUT.mkdir(exist_ok=True)
    detectors = {}
    for name in args.detectors.split(','):
        t = time.monotonic()
        detectors[name] = make_detector(name, center_tolerance=6.0)
        print(f'[load] {name}: {(time.monotonic()-t)*1000:.0f} ms'
              + (f' on {detectors[name].device}' if hasattr(detectors[name], 'device') else ''))

    if args.random:
        evaluate(detectors, args.random)
        return

    for label, frame, truth in frames:
        print(f'\n=== {label} {frame.shape[1]}x{frame.shape[0]}')
        if truth:
            print(f'truth  mask {fmt(truth["mask"])}  wafer {fmt(truth["wafer"])}  '
                  f'dx,dy = {truth["dx"]:+.2f}, {truth["dy"]:+.2f}')
        for name, det in detectors.items():
            times = []
            for i in range(args.runs):
                t = time.monotonic()
                md, wd, ma, wa, finish = det.process(frame, 'THETA', token=(label, i), stationary=False)
                times.append((time.monotonic()-t)*1000)
            mc = md['center'] if md else None
            wc = wd['center'] if wd else None
            line = f'{name:7s} mask {fmt(mc):18s} wafer {fmt(wc):18s}'
            if mc and wc:
                line += f' dx,dy = {wc[0]-mc[0]:+.2f}, {wc[1]-mc[1]:+.2f}'
            if truth and mc and wc:
                ex = (wc[0]-mc[0])-truth['dx']
                ey = (wc[1]-mc[1])-truth['dy']
                line += f'  err {math.hypot(ex, ey):.2f}px'
            line += f'  angle mask {ma[0]:.1f}' if ma else ''
            line += f' wafer {wa[0]:.1f}' if wa else ''
            line += f'  | {np.median(times):.0f} ms'
            if hasattr(det, 'yolo_ms'):
                line += f' (yolo {det.yolo_ms:.0f} ms)'
            if hasattr(det, 'opencv_ms'):
                line += f' (opencv {det.opencv_ms:.0f} ms)'
            if hasattr(det, 'sam_ms'):
                line += f' (sam2 {det.sam_ms:.0f} ms, {det.sam_mask_count} masks -> {len(det.sam_squares)} squares)'
            print(line)
            cv2.imwrite(str(OUT/f'{label}_{name}.png'), draw(frame, md, wd, det, name))
    print(f'\noverlays: {OUT}')



def evaluate(detectors, n):
    rng = np.random.default_rng(99)  # disjoint from the train (1) / val (2) seeds
    frames = [render(rng) for _ in range(n)]
    print(f'\n=== {n} randomized frames (success = both keys found, dx/dy error < 2 px)')
    for name, det in detectors.items():
        errors, times, found = [], [], 0
        for i, (frame, _, truth) in enumerate(frames):
            t = time.monotonic()
            md, wd, *_ = det.process(frame, 'THETA', token=('rand', i), stationary=False)
            times.append((time.monotonic()-t)*1000)
            if md and wd:
                found += 1
                ex = (wd['center'][0]-md['center'][0])-truth['dx']
                ey = (wd['center'][1]-md['center'][1])-truth['dy']
                errors.append(math.hypot(ex, ey))
            if i < 3:
                cv2.imwrite(str(OUT/f'rand{i}_{name}.png'), draw(frame, md, wd, det, name))
        errors = np.array(errors)
        ok = int((errors < 2).sum())
        print(f'{name:9s} success {ok}/{n} ({ok/n:.0%})  found {found}/{n}  '
              f'median err {np.median(errors) if len(errors) else float("nan"):.2f}px  '
              f'median time {np.median(times[1:]):.0f} ms')


if __name__ == '__main__':
    main()
