"""Hardware-free rehearsal: fake TB6560 firmware on a pty + synthetic camera.

  python sim_rig.py
Runs calibrate_xy.py's probe/solve against the simulated stage, then the real
web-app code (MaskAlignerApp, skip_theta) closes the XY loop on it, and reports
the TRUE remaining offset. The simulated stage copies the 2026-09-30 jog test:
~0.47 px/step, Arduino X -> image Y, ~4 deg skew, 2-step backlash, step noise.
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from make_dataset import H, W, cross_poly, fill, rotate

ROOT = Path(__file__).resolve().parent
SIM_CAL = ROOT/'out'/'sim_calibration.json'


class World:
    """Stage state in steps -> wafer key position in the image."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.lock = threading.Lock()
        self.mask = np.array([330.0, 200.0])
        self.wafer0 = self.mask+[-47.0, 119.0]  # offset seen on the real rig
        # px per +1 step for Arduino X / Y (columns), from the jog test.
        self.A = np.array([[0.40, -14.34], [13.77, -0.98]])/30.0
        self.backlash = 2.0  # steps lost on each direction reversal
        self.pos = np.zeros(2)      # commanded steps
        self.eff = np.zeros(2)      # effective (after backlash/noise) steps
        self.last_dir = np.zeros(2)

    def step(self, axis, n):
        i = 0 if axis == 'X' else 1
        with self.lock:
            d = np.sign(n)
            lost = self.backlash if self.last_dir[i] not in (0, d) else 0.0
            self.last_dir[i] = d
            self.pos[i] += n
            self.eff[i] += n-d*min(lost, abs(n))+self.rng.normal(0, .3)

    def wafer(self):
        with self.lock:
            return self.wafer0+self.A@self.eff

    def frame(self):
        ss = 4
        canvas = np.full((H*ss, W*ss), 180, np.uint8)
        wc = self.wafer()
        fill(canvas, rotate(cross_poly(80, 80/3), 1.5, wc), 95, ss)
        sq = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], np.float64)*11
        for sx in (-1, 1):
            for sy in (-1, 1):
                fill(canvas, rotate(sq+[sx*24, sy*24], .5, self.mask), 25, ss)
        img = cv2.resize(canvas.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        img += self.rng.normal(0, 3, img.shape)
        return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class FakeCap:
    def __init__(self, world):
        self.world = world

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: W, cv2.CAP_PROP_FRAME_HEIGHT: H}.get(prop, 30.0)

    def read(self):
        time.sleep(1/30)
        return True, self.world.frame()

    def grab(self):
        time.sleep(1/30)
        return True

    def retrieve(self):
        return True, self.world.frame()

    def release(self):
        pass


class FakeFirmware(threading.Thread):
    """Same line protocol and soft limit as firmware/xy_tb6560 (+ aligner_mega loader lines)."""

    def __init__(self, world, limit=4000):
        super().__init__(daemon=True)
        self.world, self.limit = world, limit
        self.master, slave = os.openpty()
        self.port = os.ttyname(slave)
        self.log = []

    def reply(self, text):
        os.write(self.master, (text+'\r\n').encode())

    def run(self):
        buf = b''
        while True:
            buf += os.read(self.master, 256)
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                self.handle(raw.decode(errors='replace').strip())

    def handle(self, line):
        if not line:
            return
        self.log.append(line)
        if line in ('s', 'S', 'STOP'):
            self.reply('STOP')
        elif line == 'PING':
            self.reply('PONG')
        elif line == 'POS':
            p = self.world.pos
            self.reply(f'POS X={int(p[0])} Y={int(p[1])} LIM={self.limit}')
        elif line[:2] in ('X ', 'Y '):
            axis, n = line[0], int(line[2:])
            i = 0 if axis == 'X' else 1
            if abs(self.world.pos[i]+n) > self.limit:
                self.reply(f'ERR {axis} limit pos={int(self.world.pos[i])} lim={self.limit}')
                return
            self.reply(f'MOVE {axis} {n}')
            time.sleep(abs(n)*200e-6+.05)
            self.world.step(axis, n)
            self.reply('DONE')
        elif line[:3] in ('R1 ', 'R2 ', 'FY '):
            self.reply(f'MOVE {line}')
            time.sleep(.05)
            self.reply('DONE')
        elif line.startswith('Z '):
            _, which, d, ms = line.split()
            time.sleep(int(ms)/1000)
            self.reply('DONE Z')
        elif line.startswith('MAG '):
            self.reply(f'DONE {line}')
        elif line == 'DIST':
            self.reply('DIST 1.58')
        elif line.startswith('T'):
            self.reply('ERR T disabled')
        else:
            self.reply(f'ERR unknown {line}')


def main():
    world = World()
    fw = FakeFirmware(world)
    fw.start()
    cv2.VideoCapture = lambda *a, **k: FakeCap(world)

    import calibrate_xy
    print('=== 1. calibrate_xy against the simulated stage')
    rig = calibrate_xy.Rig(0, fw.port)
    results = {axis: calibrate_xy.probe(rig, axis) for axis in ('X', 'Y')}
    rig.ser.close()
    cal, report = calibrate_xy.solve(results)
    print('calibration:', cal)
    print('report:', report)
    SIM_CAL.parent.mkdir(exist_ok=True)
    import json
    SIM_CAL.write_text(json.dumps(cal, indent=2))

    print('\n=== 2. web-app XY loop (skip_theta) on the simulated stage')
    import mask_aligner_optimized_v4 as app_mod
    app_mod.ARDUINO_PORT = fw.port
    app_mod.load_calibration(SIM_CAL)
    app = app_mod.MaskAlignerApp(camera_index=0, detector='opencv', skip_theta=True)
    app.start()
    time.sleep(2.0)
    t0 = time.time()
    app.start_align()
    last = None
    while time.time()-t0 < 120:
        with app.lock:
            s = (app.phase, app.status)
        if s != last:
            true = world.wafer()-world.mask
            print(f'  t={time.time()-t0:5.1f}s  {s[0]:5s} {s[1][:60]:60s} true dx,dy=({true[0]:+.2f}, {true[1]:+.2f})')
            last = s
        if s[0] in ('DONE', 'FAULT') or not app.auto_running and s[0] != 'XY':
            break
        time.sleep(.1)
    true = world.wafer()-world.mask
    moves = [l for l in fw.log if l[:2] in ('X ', 'Y ')]
    xy_result = (app.phase, len(moves), time.time()-t0, true.copy())

    print('\n=== 3. Z mode, magnet, loader, distance, transfer (aligner_mega lines)')
    app_mod.Z_MODES = {'contact': (200, 300, 200), 'proximity': (100, 300, 100)}
    app_mod.TRANSFER_SEQUENCE = ['MAG 1 1', 'R1 -2000', 'Z 3 1 100', 'MAG 1 0']
    start = len(fw.log)

    def run(label, call):
        assert call(), f'{label}: refused (app busy?)'
        while True:
            time.sleep(.05)
            with app.lock:
                if not app.auto_running:
                    print(f'  {label:10s} -> {app.status}')
                    return
    run('contact', lambda: app.z_mode('contact'))
    run('magnet', lambda: app.run_commands('MAG 1', [('MAG 1 1', 'DONE MAG 1 1', 2.0, 0.0)]))
    run('loader', lambda: app.run_commands('R1 move', [('R1 500', 'DONE', 5.0, 0.0)]))
    run('dist', lambda: app.run_commands('DIST', [('DIST', 'DIST *', 2.0, 0.0)]))
    run('transfer', lambda: app.run_commands('transfer', [(l, 'DONE*', 5.0, 0.0) for l in app_mod.TRANSFER_SEQUENCE]))
    print('  sent:', fw.log[start:])
    print(f'  distance shown: {app.distance} mm, z mode: {app.z_last_mode}')
    phase, nmoves, elapsed, true = xy_result
    print(f'\nresult: phase={phase}  moves={nmoves}  time={elapsed:.1f}s')
    print(f'TRUE remaining offset: dx={true[0]:+.3f} px  dy={true[1]:+.3f} px  |e|={np.linalg.norm(true):.3f} px')
    print(f'stage position (steps): X={int(world.pos[0])} Y={int(world.pos[1])}  (limit +-{fw.limit})')
    app.shutdown()
    os._exit(0)


if __name__ == '__main__':
    main()
