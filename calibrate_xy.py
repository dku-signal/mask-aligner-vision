"""Measure the XY stage with the camera and write calibration.json for the app.

  python calibrate_xy.py                 # camera 0, auto port
  python calibrate_xy.py --camera 0 --port /dev/cu.usbmodem11201   (Windows: --port COM3)

Stop the web app first (it holds the serial port and camera).
Per Arduino axis it moves +n then -n steps, starting small and doubling until the
wafer key moves >= TARGET_PX, and measures the wafer centre each time. From that:
  - which image axis each Arduino axis drives (SWAP_XY_COMMANDS)
  - MOTOR_SIGN_X/Y and STEPS_PER_PIXEL_X/Y in the app's convention
  - backlash (forward vs return travel) and stage/camera skew
The stage always returns to where it started; every move goes through the
firmware's soft limit.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent/'calibration.json'
TARGET_PX = 8.0      # displacement that gives a usable px/step estimate (TB6560 microsteps: ~0.004 px/step)
MAX_SAFE_PX = 120.0  # a single probe moving further than this means something is wrong
START_STEPS = 20
MAX_STEPS = 6400


# USB vendor ids: Arduino, Arduino.org, CH340 clones, FTDI, CP210x
ARDUINO_VIDS = (0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4)
NO_PONG_HELP = ('Arduino did not answer PING. Check in this order:\n'
                '  1) close anything else using the port (Arduino IDE serial monitor, the web app)\n'
                '  2) is it the right port? pass it explicitly: --port COM3  or  --port /dev/cu.usbmodemXXXX\n'
                '  3) is firmware/xy_tb6560 on the board? (Serial monitor 9600, Newline, send PING -> PONG)')


def find_arduino_port():
    """macOS names first (unchanged behaviour), then any Arduino-like USB port (Windows COMx)."""
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        if 'usbmodem' in p.device or 'usbserial' in p.device:
            return p.device
    for p in ports:
        text = f'{p.description} {p.manufacturer or ""}'.lower()
        if p.vid in ARDUINO_VIDS or 'arduino' in text or 'ch340' in text or 'usb serial' in text:
            return p.device
    return None


class Rig:
    """Real hardware: TB6560 firmware over serial + camera + PreciseKeys."""

    def __init__(self, camera, port):
        import cv2
        import serial
        from subpixel import PreciseKeys
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise SystemExit(f'camera {camera} not available')
        self.det = PreciseKeys(center_tolerance=6.0)
        self.ser = serial.Serial(port, 9600, timeout=.1)
        time.sleep(2.5)  # opening the port resets the Mega
        boot = self.ser.read(self.ser.in_waiting or 1).decode(errors='replace').strip()
        if boot:
            print(f'[arduino] {boot}')
        if self.command('PING', 'PONG', 3) != 'PONG':
            raise SystemExit(NO_PONG_HELP)
        print(f'[arduino] {self.command("POS", "POS", 3)}')

    def command(self, line, expect='DONE', wait=30):
        self.ser.reset_input_buffer()
        self.ser.write((line+'\n').encode())
        deadline = time.time()+wait
        while time.time() < deadline:
            r = self.ser.readline().decode(errors='replace').strip()
            if r.startswith(expect) or r.startswith('ERR') or r.startswith('STOP'):
                return r
        self.ser.write(b's\n')
        return 'TIMEOUT'

    def move(self, axis, steps):
        r = self.command(f'{axis} {steps}')
        if r != 'DONE':
            self.ser.write(b's\n')
            raise SystemExit(f'move {axis} {steps} failed: {r}')
        time.sleep(.5)

    def measure(self, n=8):
        """Median wafer and mask centres over n frames with both keys found."""
        w, m = [], []
        for _ in range(4):
            self.cap.read()
        for _ in range(n*4):
            ok, f = self.cap.read()
            if not ok:
                continue
            md, wd, *_ = self.det.process(f, 'XY', token=0, stationary=False)
            if md and wd:
                m.append(md['center'])
                w.append(wd['center'])
            if len(w) >= n:
                break
        if len(w) < n//2:
            raise SystemExit('keys not visible; check camera/lighting before calibrating')
        return np.median(w, axis=0), np.median(m, axis=0)

    def close(self):
        self.ser.write(b's\n')
        self.ser.close()
        self.cap.release()


def probe(rig, axis):
    """Move axis +n/-n with growing n; return (n, forward_px_vector, return_px_vector)."""
    n = START_STEPS
    while True:
        w0, m0 = rig.measure()
        rig.move(axis, n)
        w1, m1 = rig.measure()
        rig.move(axis, -n)
        w2, _ = rig.measure()
        fwd, back = w1-w0, w2-w1
        moved = float(np.linalg.norm(fwd))
        mask_shift = float(np.linalg.norm(m1-m0))
        print(f'  {axis} {n:+5d} steps: wafer moved {fwd.round(2)} px, '
              f'return {back.round(2)} px, mask {mask_shift:.2f} px')
        if mask_shift > 2.0:
            # One glitchy frame set (shadow, reflection, wrong square) must not abort; a real move persists.
            _, m_again = rig.measure(12)
            if float(np.linalg.norm(m_again-m0)) > 2.0:
                raise SystemExit('the MASK key moved: the stage carries the mask, not the wafer; stop and check')
            print('    (mask jump was a one-off detection glitch; continuing)')
            continue
        if moved > MAX_SAFE_PX:
            raise SystemExit(f'{axis}: {n} steps moved {moved:.0f} px, more than expected; stopping')
        if moved >= TARGET_PX:
            return n, fwd, back
        if n >= MAX_STEPS:
            raise SystemExit(f'{axis}: {n} steps moved only {moved:.2f} px; motor not moving?')
        n *= 2


def solve(results):
    """App convention: steps = dx * STEPS_PER_PIXEL * MOTOR_SIGN, dx = wafer - mask,
    and a correct move shifts the wafer by -dx."""
    vec = {a: fwd/n for a, (n, fwd, back) in results.items()}  # px per +1 step
    # Arduino X drives mostly image Y -> commands are swapped.
    swap = abs(vec['X'][1]) > abs(vec['X'][0])
    drive = {'X': 'Y', 'Y': 'X'} if swap else {'X': 'X', 'Y': 'Y'}  # image axis -> arduino axis
    cal = {'SWAP_XY_COMMANDS': bool(swap)}
    for img_axis, k in (('X', 0), ('Y', 1)):
        v = vec[drive[img_axis]][k]
        cal[f'MOTOR_SIGN_{img_axis}'] = -1 if v > 0 else 1
        cal[f'STEPS_PER_PIXEL_{img_axis}'] = round(1/abs(v), 4)
    px_per_step = max(np.linalg.norm(v) for v in vec.values())
    # One move may cover ~60 px; dead band is two steps but never below 0.5 px.
    cal['MAX_STEPS_PER_AXIS'] = int(60*max(cal['STEPS_PER_PIXEL_X'], cal['STEPS_PER_PIXEL_Y']))
    cal['DEAD_BAND_PX'] = round(max(.5, 2*px_per_step), 3)
    cal['MANUAL_STEPS'] = int(round(10*max(cal['STEPS_PER_PIXEL_X'], cal['STEPS_PER_PIXEL_Y'])))
    report = {}
    for a, (n, fwd, back) in results.items():
        main = int(np.argmax(abs(fwd)))
        report[a] = dict(
            steps=n, px_per_step=round(float(np.linalg.norm(fwd))/n, 5),
            image_axis='XY'[main],
            backlash_px=round(float(abs(fwd[main])-abs(back[main])), 3),
            skew_deg=round(math.degrees(math.atan2(fwd[1-main], abs(fwd[main]))), 2))
    return cal, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', default='auto', help="index or 'auto' (camera that sees both keys)")
    ap.add_argument('--limit-px', type=float, default=80.0,
                    help='after calibrating, set the soft limit to this many px of travel (0 = leave it)')
    ap.add_argument('--port', default=None)
    ap.add_argument('--zero', action='store_true', help='set the current (centred) position as 0 and exit')
    ap.add_argument('--limit', type=int, help='set the soft limit in steps (half travel) and exit')
    args = ap.parse_args()
    port = args.port
    if port is None:
        port = find_arduino_port()
        if port is None:
            raise SystemExit('Arduino port not found: check the USB cable, or pass --port COM3 / /dev/cu.usbmodemXXXX')
    if args.camera == 'auto':
        from mask_aligner_optimized_v4 import find_key_camera
        args.camera = find_key_camera()
    rig = Rig(int(args.camera), port)
    if args.zero or args.limit:
        if args.zero:
            print('ZERO ->', rig.command('ZERO'))
        if args.limit:
            print(f'LIM {args.limit} ->', rig.command(f'LIM {args.limit}'))
        print(rig.command('POS', 'POS', 3))
        rig.close()
        return
    try:
        results = {}
        for axis in ('X', 'Y'):
            print(f'[probe] Arduino {axis}')
            results[axis] = probe(rig, axis)
        cal, report = solve(results)
        if args.limit_px > 0:
            # Without the rail stroke, bound travel by image distance from the ZERO point.
            limit = int(args.limit_px*max(cal['STEPS_PER_PIXEL_X'], cal['STEPS_PER_PIXEL_Y']))
            print(f'soft limit -> +-{limit} steps (~{args.limit_px:.0f} px):', rig.command(f'LIM {limit}'))
            cal['_soft_limit_steps'] = limit
    finally:
        rig.close()
    cal['_report'] = report
    cal['_measured_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    OUT.write_text(json.dumps(cal, indent=2, ensure_ascii=False))
    print(json.dumps(cal, indent=2, ensure_ascii=False))
    for a, r in report.items():
        print(f'Arduino {a}: 1 step = {r["px_per_step"]:.4f} px -> 1000 steps = {1000*r["px_per_step"]:.0f} px '
              f'(~{1000*r["px_per_step"]*.31:.0f}-{1000*r["px_per_step"]*.47:.0f} mm at the 0.31-0.47 mm/px estimate)')
    print('Set the soft limit to fit the rail stroke: python calibrate_xy.py --limit <steps>')
    print(f'\nsaved {OUT}; the app loads it on start.')


if __name__ == '__main__':
    main()
