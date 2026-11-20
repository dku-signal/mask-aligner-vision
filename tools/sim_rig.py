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
        self.notch, self.loader_y = 0.0, 0.0  # loader camera: notch angle, disc offset (px)
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
        elif line[:3] in ('R1 ', 'R2 '):
            self.reply(f'MOVE {line}')
            n = int(line[3:])
            time.sleep(abs(n)/5000+.05)  # rails ~5000 steps/s
            if line[:2] == 'R2':
                self.world.loader_y += n/250  # R2 moves the disc down the loader image, 250 steps/px
            self.reply('DONE')
        elif line.startswith('Z '):
            _, which, d, ms = line.split()
            time.sleep(int(ms)/1000)
            self.reply('DONE Z')
        elif line.startswith('MAG '):
            self.reply(f'DONE {line}')
        elif line == 'DIST':
            self.reply('DIST 1.58')
        elif line.startswith('PT '):
            # Continuous servo, non-blocking like the firmware: dead zone ~40 ms, then ~0.09 deg/ms,
            # turning the notch backwards. Rotation is applied gradually so frames see it turn.
            _, seq, d, ms = line.split()[:4]
            self.reply(f'START PT {seq}')

            def turn():
                for _ in range(10):
                    time.sleep(int(ms)/10000)
                    self.world.notch = (self.world.notch-int(d)*0.09*max(0, int(ms)-40)/10) % 360
                self.reply(f'DONE PT {seq}')
            threading.Thread(target=turn, daemon=True).start()
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
    app.start_align()  # no frame yet: both keys unseen, so START must refuse and move nothing
    with app.lock:
        print(f'  START before any frame -> {app.phase}: {app.status}')
        assert app.phase == 'FAULT' and not app.auto_running
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
    time.sleep(0.6)  # the Z up worker starts 0.3 s after ALIGN FINISH
    while True:  # the mask plate goes back up after ALIGN FINISH
        time.sleep(.1)
        with app.lock:
            if not app.moving:
                break
    print('  Z after align:', [l for l in fw.log if l.startswith('Z ')][-2:])

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
    print('\n=== 4. pre-align: notch on a synthetic loader frame, PT pulses')
    world.notch = 250.0
    rng = np.random.default_rng(1)

    def loader_frame():
        img = rng.normal(30, 6, (480, 640)).clip(0, 255).astype(np.uint8)
        c, r, a = (330, int(round(235+world.loader_y))), 170, np.radians(world.notch)
        cv2.circle(img, c, r, 225, -1, cv2.LINE_AA)
        tip = (c[0]+r*np.cos(a), c[1]+r*np.sin(a))
        side = np.array([-np.sin(a), np.cos(a)])*14
        notch = np.array([tip+side, tip-side, (c[0]+(r-22)*np.cos(a), c[1]+(r-22)*np.sin(a))])
        cv2.fillPoly(img, [np.round(notch*16).astype(np.int32)], 30, cv2.LINE_AA, shift=4)
        return cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_GRAY2BGR)
    app.prealign_grab = loader_frame
    app_mod.PREALIGN_TARGET_DEG = 30.0
    start = len(fw.log)
    run('prealign', app.start_prealign)
    pts = [l for l in fw.log[start:] if l.startswith('PT ')]
    print(f'  pulses: {pts}')
    print(f'  TRUE notch {world.notch:.2f} deg, target 30.00 -> err {(world.notch-30+180)%360-180:+.2f} deg')
    print('\n=== 5. transfer (method 2): unwind, measure, turn while R1 carries, fine theta, Y rail by feedback')
    world.notch = 140.0
    app_mod.Y_RAIL_MAX_STEPS = 20000
    app_mod.TRANSFER_SEQUENCE = ['@UNWIND', '@STAGE_HOME', 'MAG 1 1', '@MEASURE', '@TURN', 'R1 -6000', '@THETA',
                                 '@Y', 'R1 -3000', 'MAG 1 0']
    start = len(fw.log)
    t0 = time.time()
    run('transfer', lambda: app.start_worker('transfer', app.transfer_run))
    print(f'  sent: {fw.log[start:]}  ({time.time()-t0:.1f} s)')
    print(f'  TRUE notch err {(world.notch-30+180)%360-180:+.2f} deg, net PT ms {app_mod.PREALIGN_NET_MS}')
    # Feedback loop: the fine stage needs 0.8 fine steps per rail step the rail is off the ideal 17500.
    print('  method 2 feedback over 5 wafers (ideal rail 17500, fine = -0.8 * (rail - 17500)):')
    for k in range(5):
        rail = app_mod.Y_RAIL_MAX_STEPS-app_mod.Y_RAIL_CORRECTION
        app.y_rail_sent = rail
        fine = int(round(-0.8*(rail-17500)))
        note = app.y_feedback(fine)
        print(f'    wafer {k+1}: rail {rail:6d} -> fine Y needed {fine:+6d}   {note}  ratio {app_mod.Y_RAIL_PER_FINE}')
    run('unwind', lambda: app.start_worker('unwind', lambda g: 'unwound' if app.unwind(g) else 'stopped'))
    print(f'  after unwind: net PT ms {app_mod.PREALIGN_NET_MS}')

    phase, nmoves, elapsed, true = xy_result
    print(f'\nresult: phase={phase}  moves={nmoves}  time={elapsed:.1f}s')
    print(f'TRUE remaining offset: dx={true[0]:+.3f} px  dy={true[1]:+.3f} px  |e|={np.linalg.norm(true):.3f} px')
    print(f'stage position (steps): X={int(world.pos[0])} Y={int(world.pos[1])}  (limit +-{fw.limit})')
    app.shutdown()
    os._exit(0)


if __name__ == '__main__':
    main()
