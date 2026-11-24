"""One-shot check before a rig session: port, firmware, camera, keys, calibration.

  python preflight.py
Prints PASS / FAIL / WARN per item and what to do next. Moves nothing.
"""
import json
import subprocess
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
results = []


def report(status, item, detail=''):
    results.append(status)
    print(f'[{status}] {item}' + (f' - {detail}' if detail else ''))


def main():
    try:
        running = subprocess.run(['pgrep', '-f', 'mask_aligner_optimized_v4.py'], capture_output=True).returncode == 0
    except FileNotFoundError:  # Windows has no pgrep
        running = None
    if running:
        report('FAIL', 'web app is running', 'it holds the port and camera; stop it (Ctrl+C) and rerun')
        return
    if running is None:
        report('WARN', 'web app check skipped', 'close the web app and Arduino serial monitor before this check')
    else:
        report('PASS', 'web app not running')

    from calibrate_xy import NO_PONG_HELP, find_arduino_port
    port = find_arduino_port()
    if port is None:
        report('FAIL', 'Arduino port', 'not found - check the USB cable')
    else:
        report('PASS', 'Arduino port', port)
        import serial
        try:
            ser = serial.Serial(port, 9600, timeout=.2)
            time.sleep(2.5)
            boot = ser.read(ser.in_waiting or 1).decode(errors='replace').strip()
            ser.reset_input_buffer()
            ser.write(b'PING\n')
            reply = ''
            deadline = time.time()+2
            while time.time() < deadline and not reply:
                reply = ser.readline().decode(errors='replace').strip()
            if reply == 'PONG':
                ser.write(b'POS\n')
                pos = ser.readline().decode(errors='replace').strip()
                report('PASS', 'firmware xy_tb6560', pos)
            else:
                report('FAIL', 'no PONG', f'boot: {boot!r}, reply: {reply!r}\n' + NO_PONG_HELP)
            ser.close()
        except Exception as e:
            report('FAIL', 'serial open', f'{e} - another program (serial monitor / web app) may hold the port')

    from subpixel import PreciseKeys
    rig_cam = None
    for index in range(4):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        det = PreciseKeys()
        hits, frames, last = 0, 0, None
        t0 = time.time()
        for _ in range(15):
            ok, f = cap.read()
            if not ok:
                continue
            frames += 1
            md, wd, *_ = det.process(f, 'IDLE', token=index)
            if md and wd:
                hits += 1
                last = (md['center'], wd['center'], f.shape)
        cap.release()
        fps = frames/max(time.time()-t0, 1e-6)
        if hits and rig_cam is None:
            rig_cam = index
            (mx, my), (wx, wy), shape = last
            report('PASS', f'camera {index} sees both keys',
                   f'{hits}/{frames} frames, {shape[1]}x{shape[0]}, dx={wx-mx:+.1f} dy={wy-my:+.1f} px')
            if max(abs(wx-mx), abs(wy-my)) > 60:
                report('WARN', 'keys far apart', 'move the wafer near the mask by hand before ZERO (keeps travel small)')
        else:
            print(f'       camera {index}: {frames} frames, keys {"seen" if hits else "not seen"} ({fps:.0f} fps)')
    if rig_cam is None:
        report('FAIL', 'camera with keys', 'no camera sees both keys - check USB camera, focus, lighting')

    cal = ROOT/'calibration.json'
    if cal.is_file():
        data = json.loads(cal.read_text(encoding='utf-8'))
        report('PASS', 'calibration.json', f'measured {data.get("_measured_at", "?")}')
    else:
        report('WARN', 'calibration.json missing', 'run: python calibrate_xy.py --zero, then python calibrate_xy.py')

    print()
    if 'FAIL' in results:
        print('NOT READY - fix the FAIL items above.')
    else:
        cam = rig_cam if rig_cam is not None else 'auto'
        print('READY. Next:')
        if not cal.is_file():
            print('  python calibrate_xy.py --zero     # stage centred, wafer near mask')
            print('  python calibrate_xy.py')
        print(f'  python mask_aligner_optimized_v4.py --camera {cam} --skip-theta')


if __name__ == '__main__':
    main()
