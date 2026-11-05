"""Send firmware line commands one by one and print every reply (rig bring-up).

  python tools/rig_cmd.py PING POS DIST
  python tools/rig_cmd.py "MAG 1 1" "MAG 1 0" "Z 3 -1 500" "R1 2000"
Stop the web app first (it holds the port). Each command waits for its final reply
(DONE..., STOP, ERR..., PONG, POS..., DIST...) or --wait seconds, then the next is sent.
Ctrl+C sends a stop.
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import time

from calibrate_xy import find_arduino_port

FINAL = ('DONE', 'STOP', 'ERR', 'PONG', 'POS', 'DIST', 'MAG', 'THETA', 'PREALIGN')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('commands', nargs='+')
    ap.add_argument('--port', default=None)
    ap.add_argument('--wait', type=float, default=30.0, help='seconds to wait for each final reply')
    args = ap.parse_args()

    import serial
    port = args.port or find_arduino_port()
    if port is None:
        raise SystemExit('Arduino port not found; pass --port')
    ser = serial.Serial(port, 9600, timeout=.1)
    time.sleep(2.5)  # opening the port resets the Mega
    boot = ser.read_all().decode(errors='replace').strip()
    if boot:
        print(f'[boot] {boot}')
    try:
        for line in args.commands:
            ser.write((line+'\n').encode())
            print(f'-> {line}', flush=True)
            t = time.time()
            while time.time()-t < args.wait:
                r = ser.readline().decode(errors='replace').strip()
                if not r:
                    continue
                print(f'<- {r}', flush=True)
                if r.startswith(FINAL):
                    break
            else:
                print('   (no final reply)')
    except KeyboardInterrupt:
        ser.write(b's\n')
        print('\nstop sent')
    finally:
        ser.close()


if __name__ == '__main__':
    main()
