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
from tempfile import TemporaryDirectory

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

    def deliver(self, transfer_x):
        # R1 and precision Arduino Y share a direction at a 2:1 step ratio.
        with self.lock:
            error_steps = (transfer_x - 3900) / 2
            self.wafer0 = self.mask + self.A[:, 1] * error_steps - self.A @ self.eff

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
    def __init__(self, world, notch=False):
        self.world = world
        self.notch_frame = None
        if notch:
            frame = np.full((H, W, 3), 30, np.uint8)
            x, y = W // 2, H // 2
            cv2.circle(frame, (x, y), 100, (225,) * 3, -1)
            notch_cut = np.array([(x + 110, y + 15), (x + 85, y), (x + 110, y - 15)], np.int32)
            cv2.fillConvexPoly(frame, notch_cut, (30,) * 3)
            self.notch_frame = frame

    def isOpened(self):
        return True

    def set(self, *a):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: W, cv2.CAP_PROP_FRAME_HEIGHT: H}.get(prop, 30.0)

    def read(self):
        time.sleep(1/30)
        return self.retrieve()

    def grab(self):
        time.sleep(1/30)
        return True

    def retrieve(self):
        if self.notch_frame is not None:
            return True, self.notch_frame
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
        self.transfer_referenced = False
        self.delivery_y = 0
        self.transfer_x = 0

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
            axis, n = line.split()
            if axis == 'R1':
                self.transfer_x += int(n)
            elif axis == 'R2':
                self.delivery_y += int(n)
            self.reply(f'MOVE {line}')
            time.sleep(.05)
            self.reply('DONE')
        elif line == 'PZERO':
            self.transfer_referenced = True
            self.delivery_y = 0
            self.transfer_x = 0
            self.reply('DONE PZERO')
        elif line.startswith('PX '):
            if self.transfer_referenced and self.transfer_x == self.delivery_y == 0:
                self.transfer_x = int(line.split()[1])
                self.reply('DONE PX')
            else:
                self.reply('ERR PX return to transfer datum')
        elif line.startswith('PY '):
            self.delivery_y = int(line.split()[1])
            self.world.deliver(self.transfer_x)
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
    cv2.VideoCapture = lambda index, *a, **k: FakeCap(world, notch=index == 1)

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
    sim_state = TemporaryDirectory(dir=SIM_CAL.parent)
    state_path = Path(sim_state.name) / 'prealign-x.json'
    app = app_mod.MaskAlignerApp(camera_index=0, detector='opencv', skip_theta=True,
                                 prealign_state_path=state_path)
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

    print('\n=== 4. Stationary notch, X + theta, maximum Y, and precision alignment')
    app_mod.PREALIGN = {'notch_camera': 1, 'target_notch_deg': 20,
                        'theta_ms_per_degree': 7, 'x_steps': 4000,
                        'precision_x_axis': 'Y', 'precision_to_transfer_x_ratio': 2,
                        'y_max_steps': 2400, 'handoff': ['Z 1 -1 50', 'MAG 1 0']}

    def wait_for_prealign(calibrate_from_max=False):
        assert app.start_prealign(calibrate_from_max=calibrate_from_max) is None, app.status
        deadline = time.monotonic() + 15
        while app.auto_running and time.monotonic() < deadline:
            time.sleep(.05)
        assert not app.auto_running, app.status

    start = len(fw.log)
    wait_for_prealign()
    assert app.phase == 'FAULT' and fw.log[start] == 'PX 4000 140', app.status
    assert not any(line.startswith('PY ') for line in fw.log[start:])
    assert not state_path.exists()

    run('datum', lambda: app.run_commands('Transfer datum', [('PZERO', 'DONE PZERO', 2.0, 0.0)]))
    start = len(fw.log)
    wait_for_prealign(calibrate_from_max=True)
    assert fw.log[start:start+4] == ['PX 4000 140', 'PY 2400', 'Z 1 -1 50', 'MAG 1 0'], fw.log[start:]
    assert fw.delivery_y == 2400 and app.camera_index == 0 and app.phase == 'DONE', app.status
    first = json.loads(state_path.read_text())
    measured = sum(int(line.split()[1]) for line in fw.log[start:] if line.startswith('Y '))
    assert measured < 0 and first['precision_x_steps'] == measured, first
    expected = 4000 + 2 * measured
    assert first['transfer_x_steps'] == 4000 and expected < 4000, first

    run('return datum', lambda: app.run_commands('Return datum', [
        ('R2 -2400', 'DONE', 5.0, 0.0), ('R1 -4000', 'DONE', 5.0, 0.0)
    ]))
    start = len(fw.log)
    wait_for_prealign()
    assert fw.log[start:start+2] == [f'PX {expected} 140', 'PY 2400'], fw.log[start:]
    assert app.phase == 'DONE', app.status
    second = json.loads(state_path.read_text())
    assert second['transfer_x_steps'] == expected, second
    next_target = expected + 2 * second['precision_x_steps']
    assert app_mod.prealign.load_transfer_x(
        state_path, app_mod.prealign.transfer_plan(app_mod.PREALIGN)
    ) == next_target

    persisted = state_path.read_bytes()
    fw.transfer_referenced = False
    wait_for_prealign(calibrate_from_max=True)
    assert app.phase == 'FAULT' and state_path.read_bytes() == persisted, app.status
    print('  saved precision X steps, applied the next transfer target, and preserved feedback on failure')

    phase, nmoves, elapsed, true = xy_result
    print(f'\nresult: phase={phase}  moves={nmoves}  time={elapsed:.1f}s')
    print(f'TRUE remaining offset: dx={true[0]:+.3f} px  dy={true[1]:+.3f} px  |e|={np.linalg.norm(true):.3f} px')
    print(f'stage position (steps): X={int(world.pos[0])} Y={int(world.pos[1])}  (limit +-{fw.limit})')
    app.shutdown()
    sim_state.cleanup()
    os._exit(0)


if __name__ == '__main__':
    main()
