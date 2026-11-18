"""Pre-align: find the wafer disc and its notch on the loader camera (OpenCV only).

  python prealign.py photo.jpg [more.jpg ...]     # print centre / radius / notch angle, save *_notch.jpg

Angles are image angles in degrees: 0 = right, 90 = down (clockwise on screen), range [0, 360).
The disc is a bright circle on a dark background with one cut-out (the notch) on its rim.
1. Hough circle on a downscaled frame gives a rough centre and radius.
2. Radial edge search at every degree refines the circle by least squares; rays whose edge
   is missing or far off (the notch, a bright background touching the rim) are dropped.
3. The notch is the dark arc just inside the rim: its brightness-weighted centroid gives
   the angle to a fraction of a degree.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

WORK_WIDTH = 640          # Hough runs at this width
RING = (0.88, 0.97)       # radii (fraction of r) sampled for the notch
NOTCH_MIN_CONTRAST = 0.25  # notch arc (peak degree) must be this much darker than the rim median
NOTCH_MAX_WIDTH_DEG = 40   # a wider dark arc is a hand / shadow over the rim, not the notch
DISC_MIN_EDGE = 0.6        # the rim edge must be found on this fraction of the 360 rays
DISC_MIN_STEP = 40         # and the disc must be this much brighter than just outside it


def wrap180(a):
    return (a+180) % 360-180


def _ray_samples(gray, cx, cy, r, lo, hi, n):
    """Brightness along each of the 360 rays from lo*r to hi*r: array (360, n)."""
    t = np.radians(np.arange(360))[:, None]
    rr = np.linspace(lo*r, hi*r, n)[None, :]
    xs = (cx+rr*np.cos(t)).astype(np.float32)
    ys = (cy+rr*np.sin(t)).astype(np.float32)
    return cv2.remap(gray, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float64), rr[0]


def _fit_circle(pts):
    a = np.c_[2*pts, np.ones(len(pts))]
    s = np.linalg.lstsq(a, (pts**2).sum(1), rcond=None)[0]
    return s[0], s[1], float(np.sqrt(s[2]+s[0]**2+s[1]**2))


def _refine(gray, cx, cy, r):
    """Least-squares circle on the rim edge near (cx, cy, r): (cx, cy, r, edge fraction, step) or None."""
    prof, radii = _ray_samples(gray, cx, cy, r, 0.85, 1.15, 120)
    prof = cv2.GaussianBlur(prof, (5, 1), 0)
    step = prof[:, :-2]-prof[:, 2:]
    k = step.argmax(1)
    strength = step[np.arange(360), k]
    edge_r = radii[k+1]
    t = np.radians(np.arange(360))
    edge = np.c_[cx+edge_r*np.cos(t), cy+edge_r*np.sin(t)]
    keep = strength > 0.5*np.median(strength)
    for _ in range(3):
        if keep.sum() < 60:
            return None
        cx, cy, r = _fit_circle(edge[keep])
        resid = np.hypot(edge[:, 0]-cx, edge[:, 1]-cy)-r
        keep &= abs(resid) < max(1.5, 3*np.median(abs(resid[keep])))
    # A real disc has a bright face and a rim all round; clutter circles do not.
    inside, _ = _ray_samples(gray, cx, cy, r, 0.80, 0.95, 4)
    outside, _ = _ray_samples(gray, cx, cy, r, 1.05, 1.20, 4)
    return float(cx), float(cy), float(r), float(keep.mean()), float(np.median(inside.mean(1)-outside.mean(1)))


def find_disc(gray):
    """(cx, cy, r) in full-resolution pixels, or None."""
    h, w = gray.shape
    s = min(1.0, WORK_WIDTH/w)
    small = cv2.medianBlur(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 5)
    m = min(small.shape)
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, 1.5, m*0.05, param1=80, param2=40,
                               minRadius=int(m*0.15), maxRadius=int(m*0.6))
    if circles is None:
        return None
    best = None
    for cx, cy, r in circles[0][:12]/s:  # Hough's ranking is unreliable in clutter: score each
        c = _refine(gray, cx, cy, r)
        if c is None or c[3] < DISC_MIN_EDGE or c[4] < DISC_MIN_STEP:
            continue
        if best is None or c[3]*c[4] > best[3]*best[4]:
            best = c
    return None if best is None else best[:3]


def find_notch(frame):
    """dict(center, radius, angle, contrast) or None when no disc / no clear notch."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    disc = find_disc(gray)
    if disc is None:
        return None
    cx, cy, r = disc
    prof, _ = _ray_samples(gray, cx, cy, r, *RING, 8)
    ring = prof.mean(1)
    base = np.median(ring)
    dark = np.clip(base-ring, 0, None)
    smooth = np.convolve(np.r_[dark, dark, dark], np.ones(3)/3, 'same')[360:720]
    peak = int(smooth.argmax())
    contrast = smooth[peak]/max(base, 1.0)
    width = int((np.roll(smooth, 180-peak) >= 0.5*smooth[peak]).sum())
    if contrast < NOTCH_MIN_CONTRAST or width > NOTCH_MAX_WIDTH_DEG:
        return None
    # Centroid of the dark arc around the peak (values above half the peak).
    offs = np.arange(-30, 31)
    vals = dark[(peak+offs) % 360]
    vals = np.where(vals >= 0.5*dark[peak], vals, 0)
    angle = (peak+float((offs*vals).sum()/max(vals.sum(), 1e-9))) % 360
    return {'center': (cx, cy), 'radius': r, 'angle': angle, 'contrast': float(contrast)}


def draw(frame, result, target=None):
    out = frame.copy()
    if result is None:
        cv2.putText(out, 'NO WAFER / NOTCH', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return out
    (cx, cy), r, a = result['center'], result['radius'], result['angle']
    c = (int(round(cx)), int(round(cy)))
    cv2.circle(out, c, int(round(r)), (0, 255, 0), 2)
    tip = (int(cx+r*np.cos(np.radians(a))), int(cy+r*np.sin(np.radians(a))))
    cv2.line(out, c, tip, (0, 0, 255), 3)
    if target is not None:
        tt = (int(cx+r*np.cos(np.radians(target))), int(cy+r*np.sin(np.radians(target))))
        cv2.line(out, c, tt, (255, 200, 0), 2)
    text = f'notch {a:.1f} deg' + ('' if target is None else f'  err {wrap180(a-target):+.1f}')
    cv2.putText(out, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return out


if __name__ == '__main__':
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        res = find_notch(img)
        if res is None:
            print(f'{path}: no wafer/notch')
        else:
            print(f"{path}: centre ({res['center'][0]:.1f}, {res['center'][1]:.1f}) r {res['radius']:.1f} "
                  f"notch {res['angle']:.2f} deg contrast {res['contrast']:.2f}")
        cv2.imwrite(str(Path(path).with_name(Path(path).stem+'_notch.jpg')), draw(img, res))
