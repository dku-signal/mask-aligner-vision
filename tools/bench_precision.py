"""Measurement precision against exact subpixel ground truth.

  python bench_precision.py --n 60
Frames come from make_dataset.render(ss=8): keys drawn on an 8x finer grid then
area-averaged, so the true centres are known to ~0.01 px.
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import math

import numpy as np

from make_dataset import render
from mask_aligner_optimized_v4 import CornerKeys
from subpixel import PreciseKeys

# Rough scale from the 2026-09-02 demo video (acrylic disc ~320 px wide at 640x480).
# Replace with a real calibration once a ruler/AprilTag frame is captured.
MM_PER_PX = {'640x480': (.31, .47), '1920x1080': (.31/3, .47/3)}


def stats(v):
    v = np.asarray(v)
    return f'median {np.median(v):.3f}  p95 {np.percentile(v, 95):.3f}  max {v.max():.3f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    args = ap.parse_args()
    rng = np.random.default_rng(123)
    frames = [render(rng, ss=8) for _ in range(args.n)]
    results = {}
    for name, cls in (('integer (v4)', CornerKeys), ('subpixel', PreciseKeys)):
        det = cls(center_tolerance=6.0)
        m_err, w_err, d_err, a_err, miss = [], [], [], [], 0
        for i, (frame, _, t) in enumerate(frames):
            md, wd, ma, wa, _ = det.process(frame, 'THETA', token=i, stationary=False)
            if not (md and wd):
                miss += 1
                continue
            mc, wc = np.array(md['center']), np.array(wd['center'])
            m_err.append(np.linalg.norm(mc-t['mask']))
            w_err.append(np.linalg.norm(wc-t['wafer']))
            d_err.append(np.linalg.norm((wc-mc)-[t['dx'], t['dy']]))
            a_err.append(abs(det.angle_delta(ma[0], t['mask_angle'])))
        results[name] = d_err
        print(f'\n[{name}]  found {args.n-miss}/{args.n}')
        print(f'  mask centre error px : {stats(m_err)}')
        print(f'  wafer centre error px: {stats(w_err)}')
        print(f'  dx,dy error px       : {stats(d_err)}')
        print(f'  mask angle error deg : {stats(a_err)}')
    print('\n[dx,dy error in um, p95 / median]  (scale is an estimate, see MM_PER_PX)')
    for name, d in results.items():
        for res, (lo, hi) in MM_PER_PX.items():
            p95, med = np.percentile(d, 95), np.median(d)
            print(f'  {name:13s} {res:9s}: p95 {p95*lo*1000:5.0f}-{p95*hi*1000:4.0f} um   '
                  f'median {med*lo*1000:4.0f}-{med*hi*1000:4.0f} um')


if __name__ == '__main__':
    main()
