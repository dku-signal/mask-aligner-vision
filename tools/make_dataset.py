"""Randomized synthetic mask/wafer frames with YOLO labels and ground truth.

  python make_dataset.py --train 600 --val 120      # writes ./dataset + dataset.yaml
Classes: 0 = mask square, 1 = wafer cross.
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import math
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 480
ROOT = Path(__file__).resolve().parent


def rotate(pts, angle, center):
    r = math.radians(angle)
    rot = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    return pts@rot.T+center


def cross_poly(size, arm):
    h, a = size/2, arm/2
    return np.array([(-a, -h), (a, -h), (a, -a), (h, -a), (h, a), (a, a),
                     (a, h), (-a, h), (-a, a), (-h, a), (-h, -a), (-a, -a)], np.float64)


def fill(img, poly, value, ss=1):
    # ss>1: img is supersampled; map pixel-centre coordinates onto the fine grid.
    if ss > 1:
        cv2.fillPoly(img, [np.round(((poly+.5)*ss-.5)*16).astype(np.int32)], int(value), cv2.LINE_8, shift=4)
    else:
        cv2.fillPoly(img, [np.round(poly*16).astype(np.int32)], int(value), cv2.LINE_AA, shift=4)


def box(poly):
    x0, y0 = np.clip(poly.min(axis=0), 0, [W-1, H-1])
    x1, y1 = np.clip(poly.max(axis=0), 0, [W-1, H-1])
    return (x0+x1)/2/W, (y0+y1)/2/H, (x1-x0)/W, (y1-y0)/H


def render(rng, ss=1):
    """ss>1 draws the keys on an ss-times finer grid for exact subpixel ground truth."""
    side = rng.uniform(14, 34)
    gap = side*rng.uniform(.95, 1.3)
    mask_size = 2*gap+side
    margin = mask_size*1.3
    mc = np.array([rng.uniform(margin, W-margin), rng.uniform(margin, H-margin)])
    mask_angle = rng.uniform(-20, 20)
    wafer_size = mask_size*rng.uniform(.95, 1.15)
    wafer_angle = mask_angle+rng.uniform(-30, 30)
    offset = rng.uniform(-.45, .45, 2)*mask_size
    wc = mc+offset

    bg = rng.uniform(120, 220)
    img = np.full((H, W), bg, np.float32)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    g = rng.uniform(-30, 30, 2)
    img += g[0]*(xx/W-.5)+g[1]*(yy/H-.5)
    if rng.random() < .5:  # vignette / uneven illumination
        img -= rng.uniform(0, 40)*(((xx-W/2)/W)**2+((yy-H/2)/H)**2)*2
    canvas = np.clip(img, 0, 255).astype(np.uint8)

    for _ in range(rng.integers(0, 6)):  # distractors: blobs, lines, specks
        c = rng.uniform([0, 0], [W, H]).astype(int)
        v = int(rng.uniform(40, bg-10))
        if rng.random() < .5:
            cv2.circle(canvas, tuple(c), int(rng.uniform(3, 25)), v, -1, cv2.LINE_AA)
        else:
            d = rng.uniform(-80, 80, 2).astype(int)
            cv2.line(canvas, tuple(c), tuple(c+d), v, int(rng.uniform(1, 5)), cv2.LINE_AA)

    labels = []
    if ss > 1:
        canvas = cv2.resize(canvas, (W*ss, H*ss), interpolation=cv2.INTER_NEAREST)
    cross = rotate(cross_poly(wafer_size, wafer_size/3), wafer_angle, wc)
    fill(canvas, cross, bg-rng.uniform(40, 100), ss)
    labels.append((1, *box(cross)))
    ink = rng.uniform(10, 60)
    sq = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], np.float64)*side/2
    squares = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            poly = rotate(sq+[sx*gap, sy*gap], mask_angle, mc)
            fill(canvas, poly, ink, ss)
            labels.append((0, *box(poly)))
            squares.append(poly)

    if ss > 1:
        canvas = cv2.resize(canvas.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    img = canvas.astype(np.float32)+rng.normal(0, rng.uniform(2, 8), (H, W))
    k = rng.choice([0, 3, 5])
    if k:
        img = cv2.GaussianBlur(img, (k, k), 0)
    frame = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    truth = dict(mask=tuple(mc), wafer=tuple(wc), dx=float(offset[0]), dy=float(offset[1]),
                 mask_angle=mask_angle % 90, wafer_angle=wafer_angle % 90, squares=squares)
    return frame, labels, truth


def write_split(name, n, seed):
    img_dir, lbl_dir = ROOT/'dataset'/'images'/name, ROOT/'dataset'/'labels'/name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        frame, labels, _ = render(rng)
        cv2.imwrite(str(img_dir/f'{i:05d}.png'), frame)
        (lbl_dir/f'{i:05d}.txt').write_text(
            ''.join(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n' for c, x, y, w, h in labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', type=int, default=600)
    ap.add_argument('--val', type=int, default=120)
    args = ap.parse_args()
    write_split('train', args.train, 1)
    write_split('val', args.val, 2)
    (ROOT/'dataset.yaml').write_text(
        f'path: {ROOT/"dataset"}\ntrain: images/train\nval: images/val\nnames:\n  0: square\n  1: cross\n')
    print(f'wrote {args.train}+{args.val} frames to {ROOT/"dataset"}')


if __name__ == '__main__':
    main()
