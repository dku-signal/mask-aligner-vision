"""Characterise the continuous-rotation theta servo with the alignment keys.

  python tools/theta_char.py --camera 0
Stop the web app first. Needs both keys in view.

For each (speed, duration) it sends one + pulse then one - pulse and measures the
change of theta = wafer key angle - mask key angle (median of several frames).
Small pulses first; it stops early once a pulse turns more than MAX_STEP_DEG so
the 90-degree-symmetric cross never wraps between two measurements.
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import time

import cv2
import numpy as np

from calibrate_xy import find_arduino_port
from subpixel import PreciseKeys

MAX_STEP_DEG = 20.0
PLAN = [(60, 40), (60, 100), (80, 40), (80, 100), (100, 40), (100, 100), (150, 40)]


def delta(a, b):
    return (a-b+45) % 90-45


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--port', default=None)
    args = ap.parse_args()

    import serial
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    det = PreciseKeys(center_tolerance=6.0)

    def theta(n=7):
        for _ in range(4):
            cap.read()
        vals = []
        for _ in range(n*3):
            ok, f = cap.read()
            if not ok:
                continue
            md, wd, ma, wa, _ = det.process(f, 'THETA', token=0, stationary=False)
            if ma and wa:
                vals.append(delta(wa[0], ma[0]))
            if len(vals) >= n:
                break
        if len(vals) < 3:
            raise SystemExit('keys not visible (need both mask and wafer key)')
        return float(np.median(vals)), float(np.std(vals))

    port = args.port or find_arduino_port()
    ser = serial.Serial(port, 9600, timeout=.1)
    time.sleep(2.5)
    ser.reset_input_buffer()

    def pulse(i, d, ms, speed):
        ser.write(f'T {i} {d} {ms} {speed}\n'.encode())
        t = time.time()
        while time.time()-t < 2:
            l = ser.readline().decode(errors='replace').strip()
            if l.startswith('DONE') or l.startswith('ERR') or l.startswith('STOP'):
                return l
        return 'no reply'

    t0, s0 = theta()
    print(f'start theta {t0:+.2f} deg (frame-to-frame std {s0:.3f})')
    rows, i = [], 1
    try:
        for speed, ms in PLAN:
            res = {}
            for d in (1, -1):
                before, _ = theta()
                r = pulse(i, d, ms, speed)
                i += 1
                time.sleep(.5)
                after, _ = theta()
                res[d] = delta(after, before)
                print(f'  speed {speed:3d}us {ms:3d}ms dir {d:+d}: {r:10s} change {res[d]:+7.2f} deg', flush=True)
            rows.append((speed, ms, res[1], res[-1]))
            if max(abs(res[1]), abs(res[-1])) > MAX_STEP_DEG:
                print('  (large step: stopping before the cross can wrap)')
                break
    finally:
        ser.write(b's\n')
        ser.close()
        cap.release()
    print('\nspeed  ms   +dir deg  -dir deg   +deg/s   -deg/s')
    for speed, ms, p, m in rows:
        print(f'{speed:5d} {ms:4d} {p:+9.2f} {m:+9.2f} {p/ms*1000:+8.1f} {m/ms*1000:+8.1f}')


if __name__ == '__main__':
    main()
