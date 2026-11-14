"""Mega_All_Control.ino 수동 조작 패널.

펌웨어는 한 글자 명령을 받으면 움직이기 시작하고 'X'를 받을 때까지 계속 움직인다.
그래서 이 패널은 버튼을 누르고 있는 동안만 움직이게 한다:
  - 버튼을 떼면 즉시 'X'
  - 브라우저가 0.15초마다 보내는 신호가 0.5초 끊기면 서버가 'X'
  - 연결 끊김/종료 시에도 'X'
리밋 스위치가 없으므로 이동량은 사람이 화면/장비를 보며 판단해야 한다.

실행:  python tools/motor_panel.py            (포트 자동 선택)
       python tools/motor_panel.py --mock     (보드 없이 시험)
       --cameras 0 1 2  카메라 번호 지정 (허브를 다시 꽂으면 번호가 바뀔 수 있음)
"""
import argparse, json, threading, time, webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

import serial
import serial.tools.list_ports

# 펌웨어 명령표 (Mega_All_Control.ino handleCommand)
MOVES = {  # 누르고 있는 동안만 움직이는 명령
    "F", "B", "f", "b", "Q", "W", "E", "R", "J", "H", "u", "d", "O", "P", "K", "L",
}
ONESHOT = {"T", "G", "t", "M", "X"}
HOLD_TIMEOUT_S = 0.5


class MockSerial:
    """보드 없이 시험할 때 쓰는 가짜 포트. 센서 값처럼 보이는 줄을 낸다."""

    def __init__(self):
        self.sent, self._t = [], time.time()

    def write(self, data):
        self.sent.append(data.decode())
        return len(data)

    def readline(self):
        time.sleep(0.2)
        return f"{12.34 + 0.01 * ((time.time() - self._t) % 10):.2f}\r\n".encode()

    def close(self):
        pass


class Board:
    def __init__(self):
        self.lock = threading.Lock()
        self.port = None
        self.ser = None
        self.lines = deque(maxlen=200)
        self.sensor = None
        self.sensor_at = None
        self.moving = None
        self.last_beat = 0.0
        self.magnet = False
        self.theta = "90°"
        self.prealign = "90°"
        self.log = deque(maxlen=50)

    # ---- 연결 ----
    def connect(self, port):
        self.disconnect()
        if port == "mock":
            ser = MockSerial()
        else:
            ser = serial.Serial(port, 9600, timeout=0.3, write_timeout=0.5)
            time.sleep(2.0)  # Mega는 포트를 열면 리셋됨
        with self.lock:
            self.ser, self.port = ser, port
            self.magnet, self.theta, self.prealign = False, "90°", "90°"  # 리셋 직후: 서보 정지(90), 전자석 OFF
        threading.Thread(target=self._reader, args=(ser,), daemon=True).start()
        self._note(f"연결: {port}")

    def disconnect(self):
        with self.lock:
            ser, self.ser, self.port = self.ser, None, None
        if ser:
            try:
                ser.write(b"X")
            except Exception:
                pass
            ser.close()
            self._note("연결 해제 (정지 전송)")

    def _reader(self, ser):
        while self.ser is ser:
            try:
                raw = ser.readline()
            except Exception as e:
                self._note(f"읽기 오류: {e}")
                break
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            self.lines.append(line)
            try:
                self.sensor, self.sensor_at = float(line), time.time()
            except ValueError:
                if line in ("ERR", "ADS_ERROR"):
                    self.sensor, self.sensor_at = line, time.time()

    # ---- 명령 ----
    def send(self, c):
        with self.lock:
            if not self.ser:
                raise RuntimeError("보드가 연결되지 않았습니다")
            self.ser.write(c.encode("ascii"))
        if c in ("X", "x"):
            self.moving = None
        elif c in MOVES:
            self.moving, self.last_beat = c, time.time()
        elif c == "M":
            self.magnet = not self.magnet
        elif c == "T":
            self.theta = "90°"
        elif c == "G":
            self.theta = "0°"
        elif c == "t":
            self.prealign = "90°" if self.prealign == "0°" else "0°"
        self._note(f"보냄 {c}")

    def beat(self):
        self.last_beat = time.time()

    def watchdog(self):
        while True:
            time.sleep(0.05)
            if self.moving and time.time() - self.last_beat > HOLD_TIMEOUT_S:
                try:
                    self.send("X")
                    self._note("신호 끊김 → 자동 정지")
                except Exception:
                    self.moving = None

    def _note(self, s):
        self.log.appendleft(time.strftime("%H:%M:%S ") + s)

    def status(self):
        age = None if self.sensor_at is None else round(time.time() - self.sensor_at, 1)
        return {"port": self.port, "moving": self.moving, "sensor": self.sensor, "sensor_age": age,
                "magnet": self.magnet, "theta": self.theta, "prealign": self.prealign,
                "log": list(self.log)[:12], "raw": list(self.lines)[-8:]}


def ports():
    out = [{"device": p.device, "desc": f"{p.description} {p.manufacturer or ''}".strip(),
            "arduino": bool(p.vid) and ("arduino" in (p.manufacturer or "").lower() or p.vid in (0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4))}
           for p in serial.tools.list_ports.comports() if p.vid]  # USB 장치만 (블루투스·내장 포트 제외)
    return out + [{"device": "mock", "desc": "가상 보드 (시험용)", "arduino": False}]


class CameraCapture:
    """한 카메라를 백그라운드 스레드로 계속 읽고 최신 프레임만 보관한다."""

    def __init__(self, index, size):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {index} did not open")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.latest, self.latest_at, self.error = None, None, None
        self.closed = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while not self.closed.is_set():
            ok, bgr = self.cap.read()
            if ok:
                self.latest, self.latest_at, self.error = bgr, time.monotonic(), None
            else:
                self.error = "Camera read failed"
                self.closed.wait(0.1)

    def close(self):
        self.closed.set()
        self.cap.release()


class Cameras:
    """카메라 3대 관찰용. 0.3초마다 화면을 JPEG로 내보낸다. 모터 제어와 무관."""

    def __init__(self):
        self.views = {}  # index -> dict(capture, error)

    def open(self, indices, size=(640, 480)):
        for i in indices:
            try:
                self.views[i] = dict(capture=CameraCapture(i, size), error=None)
            except Exception as e:
                self.views[i] = dict(capture=None, error=str(e))

    def state(self):
        out = []
        for i, v in self.views.items():
            cap = v["capture"]
            frame = getattr(cap, "latest", None)
            err = None if frame is not None else v["error"] or getattr(cap, "error", None)
            age = None if frame is None else round(time.monotonic() - cap.latest_at, 1)
            out.append({"index": i, "error": err, "age": age,
                        "size": None if frame is None else [frame.shape[1], frame.shape[0]]})
        return out

    def jpeg(self, i):
        cap = self.views.get(i, {}).get("capture")
        frame = getattr(cap, "latest", None)
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else None

    def close(self):
        for v in self.views.values():
            if v["capture"] is not None:
                v["capture"].close()


BOARD = Board()
CAMS = Cameras()
PAGE = Path(__file__).with_name("motor_panel.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = PAGE.read_bytes()  # 매 요청마다 새로 읽음
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            self._json(BOARD.status())
        elif self.path == "/ports":
            self._json(ports())
        elif self.path == "/cameras":
            self._json(CAMS.state())
        elif self.path.startswith("/cam/"):
            jpg = CAMS.jpeg(int(self.path[5:].split(".")[0].split("?")[0]))
            if jpg is None:
                return self._json({"error": "no frame"}, 503)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/connect":
                BOARD.connect(data["port"])
            elif self.path == "/disconnect":
                BOARD.disconnect()
            elif self.path == "/cmd":
                c = data["c"]
                if c not in MOVES | ONESHOT:
                    raise ValueError(f"알 수 없는 명령 {c!r}")
                BOARD.send(c)
            elif self.path == "/beat":
                BOARD.beat()
            else:
                return self._json({"error": "not found"}, 404)
            self._json({"ok": True, **BOARD.status()})
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="시리얼 포트 (기본: 자동)")
    ap.add_argument("--mock", action="store_true", help="보드 없이 가상 포트로 실행")
    ap.add_argument("--http", type=int, default=8790)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cameras", type=int, nargs="*", default=[0, 1, 2],
                    help="열 카메라 번호 (기본 0 1 2, 빈 값이면 카메라 없이)")
    # 카메라 3대가 USB 2.0 허브 하나를 같이 쓰면 640×480 무압축은 대역폭을 넘어 1대가 영상을 못 받는다.
    ap.add_argument("--camera-size", type=int, nargs=2, default=[320, 240], metavar=("W", "H"))
    a = ap.parse_args()
    threading.Thread(target=BOARD.watchdog, daemon=True).start()
    target = "mock" if a.mock else a.port or next((p["device"] for p in ports() if p["arduino"]), None)
    if target:
        try:
            BOARD.connect(target)
        except Exception as e:
            print("연결 실패:", e)
    if a.cameras:
        print("카메라 여는 중:", a.cameras)
        CAMS.open(a.cameras, tuple(a.camera_size))  # macOS 카메라 권한 요청은 메인 스레드에서만 동작
        for c in CAMS.state():
            print(f"  카메라 {c['index']}:", c["error"] or "열림")
    url = f"http://127.0.0.1:{a.http}/"
    srv = ThreadingHTTPServer(("127.0.0.1", a.http), Handler)
    print("모터 조작 패널:", url, "| 연결:", BOARD.port or "없음 (화면에서 선택)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        BOARD.disconnect()
        CAMS.close()


if __name__ == "__main__":
    main()
