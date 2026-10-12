"""Find the real travel of each rail by driving to both hard ends (no limit switches).

  python tools/find_limits.py --axis Y          # then --axis X
Stop the web app first and stand by the SMPS switch.

Moves in CHUNK steps at a slow step rate and compares camera frames before/after
each chunk. When a chunk no longer changes the image the carriage is at the end
(the stepper stalls briefly - a short grinding noise). It backs off, does the other
end, then moves to the midpoint, sets it as 0 and sets the soft limit to the
measured half-stroke minus MARGIN. Errors land on the safe side: a carriage that
leaves the camera view reads as "stopped", which only shortens the measured travel.
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import re
import time

import cv2
import numpy as np

from calibrate_xy import find_arduino_port

CHUNK = 2000            # steps per probe move
SEARCH_SPD = 200        # firmware half-period in us (100 = normal speed); slower = gentler stall
SEARCH_LIM = 400000     # soft limit while searching; restored afterwards
MAX_TRAVEL = 300000     # give up after this many steps in one direction
MARGIN = 3000           # final limit stays this far inside each measured end
STALL_RATIO = .25       # chunk motion below this fraction of normal motion = stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', choices=('X', 'Y'), required=True)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--port', default=None)
    args = ap.parse_args()

    import serial
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def frame():
        for _ in range(3):
            cap.read()
        acc = None
        for _ in range(3):
            ok, f = cap.read()
            g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
            acc = g if acc is None else acc+g
        return acc/3

    ser = serial.Serial(args.port or find_arduino_port(), 9600, timeout=.1)
    time.sleep(2.5)
    ser.reset_input_buffer()

    def cmd(line, wait=60):
        ser.reset_input_buffer()
        ser.write((line+'\n').encode())
        t = time.time()
        while time.time()-t < wait:
            l = ser.readline().decode(errors='replace').strip()
            if l.startswith(('DONE', 'ERR', 'STOP', 'POS')):
                return l
        ser.write(b's\n')
        return 'TIMEOUT'

    def pos():
        x, y, lim = map(int, re.findall(r'X=(-?\d+) Y=(-?\d+) LIM=(\d+)', cmd('POS'))[0])
        return (x if args.axis == 'X' else y), lim

    start, old_lim = pos()
    noise = float(np.mean(np.abs(frame()-frame())))
    print(f'{args.axis} start pos {start}, limit {old_lim}, frame noise {noise:.2f}')
    print('LIM', cmd(f'LIM {SEARCH_LIM}'), '| SPD', cmd(f'SPD {SEARCH_SPD}'))

    def run_to_end(direction):
        normal = []
        travelled = 0
        prev = frame()
        while travelled < MAX_TRAVEL:
            r = cmd(f'{args.axis} {direction*CHUNK}')
            if r != 'DONE':
                raise SystemExit(f'move refused: {r}')
            travelled += CHUNK
            time.sleep(.3)
            cur = frame()
            motion = float(np.mean(np.abs(cur-prev)))
            prev = cur
            ref = np.median(normal) if len(normal) >= 3 else None
            print(f'  {direction*travelled:+8d} steps  motion {motion:6.2f}' + (f'  (normal {ref:.2f})' if ref else ''), flush=True)
            if ref is not None and motion < max(STALL_RATIO*ref, 2*noise):
                p, _ = pos()
                print(f'  end reached near pos {p}')
                cmd(f'{args.axis} {-direction*CHUNK}')  # back off the end stop
                return p
            normal.append(motion)
        raise SystemExit('no end found within MAX_TRAVEL')

    try:
        print(f'[1] toward -')
        end_neg = run_to_end(-1)
        print(f'[2] toward +')
        end_pos = run_to_end(+1)
        stroke = end_pos-end_neg
        centre = (end_pos+end_neg)//2
        now, _ = pos()
        print(f'stroke {stroke} steps; moving to centre {centre}')
        rem = centre-now
        while rem:
            step = max(-20000, min(20000, rem))
            if cmd(f'{args.axis} {step}') != 'DONE':
                raise SystemExit('move to centre failed')
            rem -= step
        print('ZERO', cmd('ZERO'))
        limit = stroke//2-MARGIN
        print(f'LIM {limit}', cmd(f'LIM {limit}'))
        print(f'\n{args.axis}: stroke {stroke} steps, now at its centre (pos 0), soft limit +-{limit}')
        print('LM-2015R stroke is 200 mm -> about', round(stroke/200), 'steps/mm if the full stroke was reached')
    except BaseException:
        ser.write(b's\n')
        print('aborted - restoring the previous limit', cmd(f'LIM {old_lim}'))
        raise
    finally:
        cmd('SPD 100')
        ser.close()
        cap.release()


if __name__ == '__main__':
    main()
