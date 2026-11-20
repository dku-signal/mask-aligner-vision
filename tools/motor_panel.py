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
import argparse, json, queue, sys, threading, time, webbrowser
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
    """보드 없이 시험할 때 쓰는 가짜 포트. 센서 값처럼 보이는 줄을 내고,
    '#<c> <n>' 줄 명령에는 펌웨어처럼 걸리는 시간 뒤 DONE을 돌려준다 (SPEED배 빠르게)."""

    SPEED = 20.0

    def __init__(self):
        self.sent, self._t = [], time.time()
        self.out = queue.Queue()
        self.buf = ""

    def write(self, data):
        text = data.decode()
        self.sent.append(text)
        for ch in text:
            if self.buf or ch == "#":
                if ch in "\r\n":
                    self._line(self.buf[1:])
                    self.buf = ""
                else:
                    self.buf += ch
            elif ch in "Xx":
                self.out.put("STOP")
        return len(data)

    def _line(self, line):
        p = line.split()
        if p[0] == "PING":
            return self.out.put("PONG")
        if p[0] == "M":
            return self.out.put(f"DONE M {p[1]}")
        n = int(p[1])
        secs = n / 10000 if p[0] in "FBfbQWERJH" else n / 1000
        threading.Timer(secs / self.SPEED, lambda: self.out.put(f"DONE {p[0]} {n}")).start()

    def readline(self):
        try:
            return (self.out.get(timeout=0.2) + "\r\n").encode()
        except queue.Empty:
            return f"{12.34 + 0.01 * ((time.time() - self._t) % 10):.2f}\r\n".encode()

    def close(self):
        pass


LEDGER_FILE = Path(__file__).resolve().parents[1] / "out" / "position_ledger.json"   # out/은 git 제외
SIGNED = {  # 명령 글자 -> (축, 부호). 줄 명령(#)으로 끝까지 간 이동만 기록된다.
    "W": ("x_rail", 1), "Q": ("x_rail", -1), "E": ("y_rail", 1), "R": ("y_rail", -1),
    "F": ("align_x", 1), "B": ("align_x", -1), "f": ("align_y", 1), "b": ("align_y", -1),
}


class Ledger:
    """원상 복구용 장부: 레일 위치(스텝)와 회전 서보 누적 회전(ms, 부호=방향). 파일에 저장된다."""

    def __init__(self):
        self.lock = threading.Lock()
        self.d = {"pos": {"x_rail": 0, "y_rail": 0, "align_x": 0, "align_y": 0},
                  "servo_ms": {"t": 0, "A": 0}, "home": None, "box_x": None, "align_left": None}
        try:
            self.d.update(json.loads(LEDGER_FILE.read_text()))
        except Exception:
            pass

    def save(self):
        LEDGER_FILE.parent.mkdir(exist_ok=True)
        LEDGER_FILE.write_text(json.dumps(self.d, indent=1))

    def apply(self, c, n, servo_dir=1):
        with self.lock:
            if c in SIGNED:
                ax, sign = SIGNED[c]
                self.d["pos"][ax] += sign * n
            elif c in ("t", "A"):
                self.d["servo_ms"][c] += servo_dir * n
            else:
                return
            self.save()

    def reset(self):
        with self.lock:
            self.d.update(pos={k: 0 for k in self.d["pos"]}, servo_ms={"t": 0, "A": 0}, home=None, box_x=None, align_left=None)
            self.save()

    def snapshot(self, key, value):
        with self.lock:
            self.d[key] = value
            self.save()


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
        self.pulse_cancel = threading.Event()
        self.replies = queue.Queue()   # 줄 명령 응답(DONE/ERR/STOP/PONG)
        self.ledger = Ledger()
        self.unmatched = []
        self.servo_dir = {"t": 1, "A": 1}   # 회전 명령의 방향 (DONE에는 방향이 없어서 보낼 때 기억)
        self.pulses = deque(maxlen=100)  # 시간 지정 구동 기록 (값 측정용)

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
            if line.split()[0] in ("DONE", "ERR", "STOP", "PONG"):
                p = line.split()
                if p[0] in ("DONE", "STOP") and len(p) == 3 and p[1] != "M":
                    # DONE = 끝까지 감, STOP <c> <n> = 중간 정지 시 실제로 간 양 → 둘 다 장부에 반영
                    self.ledger.apply(p[1], int(p[2]), self.servo_dir.get(p[1], 1))
                    if p[0] == "STOP":
                        continue   # 이어서 오는 단독 "STOP"이 대기 중인 명령을 끝낸다
                self.replies.put(line)
                continue
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
            self.pulse_cancel.set()
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

    def pulse(self, c, ms):
        """c를 보내고 정확히 ms 뒤에 X를 보낸다 (DC 축 시간 측정용). 정지 버튼으로 중단된다."""
        if c not in MOVES:
            raise ValueError(f"시간 구동은 이동 명령만 가능: {c!r}")
        if not 20 <= ms <= 15000:
            raise ValueError("시간은 20~15000 ms")
        if self.moving:
            raise RuntimeError("다른 축이 움직이는 중")
        self.pulse_cancel.clear()
        self.send(c)
        started = time.perf_counter()
        end = started + ms / 1000
        while not self.pulse_cancel.is_set():
            self.last_beat = time.time()  # 감시견이 끊지 않게
            left = end - time.perf_counter()
            if left <= 0:
                break
            time.sleep(min(left, 0.005))
        cancelled = self.pulse_cancel.is_set()
        if not cancelled:
            self.send("X")
        actual = round((time.perf_counter() - started) * 1000)
        rec = {"at": time.strftime("%H:%M:%S"), "c": c, "ms": ms, "actual_ms": actual, "cancelled": cancelled}
        self.pulses.appendleft(rec)
        self._note(f"시간 구동 {c} {ms}ms → 실제 {actual}ms" + (" (중단)" if cancelled else ""))
        return rec

    # ---- 줄 명령 (펌웨어가 스텝/ms를 세어 스스로 멈추고 DONE을 보냄) ----
    def line_nowait(self, cmd):
        with self.lock:
            if not self.ser:
                raise RuntimeError("보드가 연결되지 않았습니다")
            self.ser.write(f"#{cmd}\n".encode("ascii"))
        c, *rest = cmd.split()
        if c == "M":
            self.magnet = rest[0] == "1"
        if c in self.servo_dir:
            self.servo_dir[c] = 1 if int(rest[1]) >= 0 else -1
        self._note(f"보냄 #{cmd}")

    def wait_done(self, c, timeout, stop_evt=None):
        end = time.time() + timeout
        while time.time() < end:
            if stop_evt is not None and stop_evt.is_set():
                raise InterruptedError("정지됨")
            early = next((r for r in self.unmatched if r.startswith(f"DONE {c}")), None)
            if early:
                self.unmatched.remove(early)
                return early
            try:
                r = self.replies.get(timeout=0.1)
            except queue.Empty:
                continue
            if r.startswith(f"DONE {c}"):
                return r
            if r.startswith("DONE"):
                self.unmatched.append(r)   # 동시에 돌던 다른 명령의 완료 (예: 레일 이동 중 서보 회전)
                continue
            if r.startswith("ERR"):
                raise RuntimeError(f"보드 오류: {r}")
            if r == "STOP":
                raise InterruptedError("보드가 정지(STOP)")
        self.send("X")
        raise TimeoutError(f"{c}: {timeout:.0f}초 안에 DONE 없음 → 정지")

    def line(self, cmd, timeout, stop_evt=None):
        # 이전 응답 찌꺼기는 버린다 (같은 글자의 옛 DONE을 새 완료로 착각하지 않게)
        self.unmatched.clear()
        while not self.replies.empty():
            self.replies.get_nowait()
        self.line_nowait(cmd)
        return self.wait_done(cmd.split()[0], timeout, stop_evt)

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
                "log": list(self.log)[:12], "pulses": list(self.pulses)[:30], "ledger": self.ledger.d, "raw": list(self.lines)[-8:]}


def ports():
    out = [{"device": p.device, "desc": f"{p.description} {p.manufacturer or ''}".strip(),
            "arduino": bool(p.vid) and ("arduino" in (p.manufacturer or "").lower() or p.vid in (0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4))}
           for p in serial.tools.list_ports.comports() if p.vid]  # USB 장치만 (블루투스·내장 포트 제외)
    return out + [{"device": "mock", "desc": "가상 보드 (시험용)", "arduino": False}]


class CameraCapture:
    """한 카메라를 백그라운드 스레드로 계속 읽고 최신 프레임(BGR)만 보관한다."""

    def __init__(self, index, size):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {index} did not open")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
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

    def frame_bgr(self, i):
        cap = self.views.get(i, {}).get("capture")
        return getattr(cap, "latest", None)

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
SEQ = None


def seq():
    global SEQ
    if SEQ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from prealign_sequence import Sequence
        SEQ = Sequence(BOARD, CAMS)
    return SEQ
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
        elif self.path == "/seq":
            self._json({"plan": seq().plan(), **seq().state})
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
            elif self.path == "/pulse":
                return self._json({"ok": True, **BOARD.pulse(data["c"], int(data["ms"]))})
            elif self.path == "/seq/start":
                seq().start(data.get("only"))
                return self._json({"ok": True})
            elif self.path == "/seq/restore":
                seq().restore()
                return self._json({"ok": True})
            elif self.path == "/ledger/reset":
                BOARD.ledger.reset()
            elif self.path == "/ledger/mark":
                # 지금 위치를 기준점으로 기억: align_left(정밀 X 왼쪽 끝) 또는 box_x(X 레일 박스 끝)
                key = data["key"]
                axis = {"align_left": "align_x", "box_x": "x_rail"}[key]
                BOARD.ledger.snapshot(key, BOARD.ledger.d["pos"][axis])
            elif self.path == "/turn":
                c, ms, d = data["c"], int(data["ms"]), int(data["dir"])
                if c not in ("t", "A") or not 20 <= ms <= 20000:
                    raise ValueError("회전: t(프리얼라인) 또는 A(정밀 θ), 20~20000ms")
                return self._json({"ok": True, "reply": BOARD.line(f"{c} {ms} {1 if d >= 0 else -1}", ms / 1000 + 5)})
            elif self.path == "/seq/stop":
                seq().stop()
                return self._json({"ok": True})
            elif self.path == "/line":
                c, n = data["c"], int(data["n"])
                if c == "M":   # 전자석 절대값 켜기/끄기 (토글 아님)
                    return self._json({"ok": True, "reply": BOARD.line(f"M {1 if n else 0}", 5)})
                if c not in MOVES:
                    raise ValueError(f"알 수 없는 축 {c!r}")
                limit = 2_000_000 if c in "FBfbQWERJH" else 30_000
                if not 0 < n <= limit:
                    raise ValueError(f"값 범위 1~{limit}")
                secs = n / 10000 if c in "FBfbQWERJH" else n / 1000
                return self._json({"ok": True, "reply": BOARD.line(f"{c} {n}", secs * 1.5 + 5)})
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
