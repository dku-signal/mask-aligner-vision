"""Does the theta servo actually turn the wafer disc? Measure it with the camera.

  python tools/theta_probe.py               # camera 0, auto port
  python tools/theta_probe.py --camera 0 --port /dev/cu.usbmodem11201

Needs no alignment keys: the disc is found as a circle and its rotation between
frames is estimated by ECC (rotation + translation) inside the disc only, so the
static rails/frame do not dilute the estimate. Stop the web app first.

Runs two tests and reports the measured disc rotation after each command:
  1. positional servo: P 1500/1600/1400/1500 (holds a pulse width)
  2. continuous servo: T pulses of 250 ms, 3x each direction
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import math
import time

import cv2
import numpy as np


def find_disc(gray):
    """(cx, cy, r) of the largest circle - the wafer disc."""
    blur = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w,
                               param1=80, param2=40, minRadius=int(.15*min(h, w)), maxRadius=int(.6*max(h, w)))
    if circles is None:
        return None
    return max(circles[0], key=lambda c: c[2])


class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise SystemExit(f'camera {index} not available')

    def gray(self, n=5):
        """Mean of n fresh frames (less sensor noise)."""
        for _ in range(3):
            self.cap.read()
        acc = None
        for _ in range(n):
            ok, f = self.cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
            acc = g if acc is None else acc+g
        return acc/n


def rotation_deg(ref, cur, mask):
    """Rotation of cur relative to ref inside mask, degrees, counter-clockwise + (ECC, Euclidean)."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    a = cv2.GaussianBlur(ref, (5, 5), 0)
    b = cv2.GaussianBlur(cur, (5, 5), 0)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_EUCLIDEAN, criteria, mask, 5)
    except cv2.error:
        return float('nan')
    # ECC's warp maps ref onto cur in the opposite sense; report counter-clockwise as +.
    return -math.degrees(math.atan2(warp[1, 0], warp[0, 0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--port', default=None)
    ap.add_argument('--speed', type=int, default=300, help='T pulse speed offset in us (firmware 50..500)')
    args = ap.parse_args()

    import serial
    from calibrate_xy import find_arduino_port
    port = args.port or find_arduino_port()
    if port is None:
        raise SystemExit('Arduino port not found; pass --port')
    cam = Camera(args.camera)
    ref = cam.gray()
    disc = find_disc(ref.astype(np.uint8))
    if disc is None:
        raise SystemExit('wafer disc not found in the camera image')
    cx, cy, r = disc
    mask = np.zeros(ref.shape, np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r*.9), 255, -1)  # stay inside the rim
    print(f'disc at ({cx:.0f}, {cy:.0f}) r={r:.0f} px')

    ser = serial.Serial(port, 9600, timeout=.1)
    time.sleep(2.5)
    ser.reset_input_buffer()

    def cmd(line, wait=3):
        ser.write((line+'\n').encode())
        t = time.time()
        while time.time()-t < wait:
            l = ser.readline().decode(errors='replace').strip()
            if l.startswith('DONE') or l.startswith('ERR') or l.startswith('STOP'):
                return l
        return 'no reply'

    def measure(label, reply):
        time.sleep(.6)
        a = rotation_deg(ref, cam.gray(), mask)
        print(f'  {label:28s} {reply:14s} disc rotation {a:+7.2f} deg (vs start)')
        return a

    try:
        print('[1] positional test (P = hold pulse width)')
        for us in (1500, 1600, 1400, 1500):
            measure(f'P {us}', cmd(f'P {us}'))
        print(f'[2] continuous test (T 250 ms, speed {args.speed} us)')
        i = 1
        for d in (1, 1, 1, -1, -1, -1):
            measure(f'T {i} {d:+d} 250 {args.speed}', cmd(f'T {i} {d} 250 {args.speed}'))
            i += 1
    finally:
        ser.write(b's\n')
        ser.close()
    print('\nReading it:')
    print('  P moves the disc and it stays  -> positional servo: control by angle (P)')
    print('  T accumulates rotation         -> continuous servo: control by timed pulses (T)')
    print('  nothing moves (|angle| < 0.2)  -> servo not driving the disc: power / D30 / coupling')


if __name__ == '__main__':
    main()
