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
if __package__:
    from .fine_stage import DIRECTIONS, FineStage
else:
    from fine_stage import DIRECTIONS, FineStage

# 펌웨어 명령표 (Mega_All_Control.ino handleCommand)
MOVES = {  # 누르고 있는 동안만 움직이는 명령
    "F", "B", "f", "b", "Q", "W", "E", "R", "J", "H", "u", "d", "O", "P", "K", "L",
}
ONESHOT = {"T", "G", "t", "M", "X"}
# 이송·프리얼라인 쪽 글자 (이 축이 움직이면 8단계 측정 위치가 바뀐다) / 정밀 얼라인 쪽 글자
TRANSFER_LETTERS = "QWERJHKLOPtM"
FINE_LETTERS = "FBfbAud"
HOLD_TIMEOUT_S = 0.5
FINE_JOG_STEPS = 2500


class MockSerial:
    """보드 없이 시험할 때 쓰는 가짜 포트. 센서 값처럼 보이는 줄을 내고,
    '#<c> <n>' 줄 명령에는 펌웨어처럼 걸리는 시간 뒤 DONE을 돌려준다 (SPEED배 빠르게)."""

    SPEED = 20.0

    def __init__(self):
        self.sent, self._t = [], time.time()
        self.out = queue.Queue()
        self.buf = ""
        self.pending = {}   # 축 그룹 -> (글자, 값, 시작 시각, 타이머)  펌웨어 v2처럼 축마다 따로 돈다
        self.lock = threading.Lock()

    GROUP = {c: g for g, letters in (("X", "FB"), ("Y", "fb"), ("R1", "QW"), ("R2", "ER"), ("Z1", "ud"),
                                     ("Z2", "OP"), ("Z3", "KL"), ("t", "t"), ("A", "A")) for c in letters}

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
                self._stop(self.GROUP.values(), bare=True)
        return len(data)

    def _stop(self, groups, bare=False):
        with self.lock:
            hit = [self.pending.pop(g) for g in set(groups) if g in self.pending]
        for c, n, started, timer in hit:
            timer.cancel()
            rate = 10000 if c in "FBfbQWER" else 1000
            self.out.put(f"STOP {c} {min(n, int((time.time() - started) * self.SPEED * rate))}")
        if bare and hit:
            self.out.put("STOP")

    def _line(self, line):
        p = line.split()
        if p[0] == "PING":
            return self.out.put("PONG")
        if p[0] == "X":
            return self._stop([self.GROUP[c] for c in (p[1] if len(p) > 1 else "") if c in self.GROUP])
        if p[0] == "M":
            return self.out.put(f"DONE M {p[1]}")
        n = int(p[1])
        secs = n / 10000 if p[0] in "FBfbQWERJH" else n / 1000
        group = self.GROUP.get(p[0], p[0])
        self._stop([group])

        def done():
            with self.lock:
                if self.pending.get(group, (None,))[0] != p[0]:
                    return
                self.pending.pop(group)
            self.out.put(f"DONE {p[0]} {n}")
        timer = threading.Timer(secs / self.SPEED, done)
        with self.lock:
            self.pending[group] = (p[0], n, time.time(), timer)
        timer.start()

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
            self.d.update(json.loads(LEDGER_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass

    def save(self):
        LEDGER_FILE.parent.mkdir(exist_ok=True)
        LEDGER_FILE.write_text(json.dumps(self.d, indent=1), encoding="utf-8")

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
        self.lock = threading.RLock()
        self.port = None
        self.ser = None
        self.revision = 0
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
        self.fine_stage = None
        self.unmatched = []
        self.mail = threading.Condition()
        self.mailbox = {}          # 글자 -> 아직 안 읽은 "DONE/STOP <c> <n>" 응답 (축마다 따로)
        self.err_epoch = 0         # ERR/연결 끊김이 올 때마다 증가 → 기다리던 쪽이 모두 실패
        self.stop_epoch = 0        # 전체 정지(단독 "STOP")가 올 때마다 증가
        self.servo_dir = {"t": 1, "A": 1}   # 회전 명령의 방향 (DONE에는 방향이 없어서 보낼 때 기억)
        self.pulses = deque(maxlen=100)  # 시간 지정 구동 기록 (값 측정용)

    # ---- 연결 ----
    def connect(self, port):
        self.disconnect()
        if port == "mock":
            ser = MockSerial()
        else:
            ser = serial.Serial(port, 9600, timeout=0.3, write_timeout=0.5, exclusive=True)
            time.sleep(2.0)  # Mega는 포트를 열면 리셋됨
        with self.lock:
            self.fine_stage = FineStage.for_port(port)
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
                with self.lock:
                    if self.ser is ser:
                        self.ser, self.port, self.moving = None, None, None
                        ser.close()
                        self.replies.put("ERR disconnected")
                with self.mail:
                    self.err_epoch += 1
                    self.last_err = "ERR disconnected"
                    self.mail.notify_all()
                break
            line = raw.decode(errors="replace").strip()
            if not line:
                continue
            self.lines.append(line)
            if line.split()[0] in ("DONE", "ERR", "STOP", "PONG") and line != "ERR":   # 단독 "ERR"은 레이저 범위 밖
                p = line.split()
                if p[0] in ("DONE", "STOP") and len(p) == 3 and p[1] != "M":
                    # DONE = 끝까지 감, STOP <c> <n> = 중간 정지 시 실제로 간 양 → 둘 다 장부에 반영
                    self.ledger.apply(p[1], int(p[2]), self.servo_dir.get(p[1], 1))
                    if p[1] in DIRECTIONS:
                        with self.lock:
                            if self.ser is not ser:
                                break
                            self.fine_stage.ack(p[1], int(p[2]), partial=p[0] == "STOP")
                            if p[0] == "DONE" and self.moving == p[1]:
                                if time.time() - self.last_beat < HOLD_TIMEOUT_S:
                                    self.line_nowait(f"{p[1]} {FINE_JOG_STEPS}")
                                continue
                if p[0] in ("DONE", "STOP") and len(p) == 3:
                    with self.mail:   # 그 글자를 기다리는 쪽에만 전달 (정밀·이송이 동시에 기다려도 섞이지 않게)
                        self.mailbox.setdefault(p[1], []).append(line)
                        self.mail.notify_all()
                    continue
                if p[0] in ("ERR", "STOP"):
                    with self.mail:
                        if p[0] == "ERR":
                            self.err_epoch += 1
                            self.last_err = line
                        else:
                            self.stop_epoch += 1
                        self.mail.notify_all()
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
            if c in DIRECTIONS:
                self.moving, self.last_beat = c, time.time()
                self.line_nowait(f"{c} {FINE_JOG_STEPS}")
                return
            if c in ("X", "x"):
                self.moving = None
            self.ser.write(c.encode("ascii"))
            if c not in ("X", "x"):
                self.revision += 1
        if c in ("X", "x"):
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
        c, *rest = cmd.split()
        with self.lock:
            if not self.ser:
                raise RuntimeError("보드가 연결되지 않았습니다")
            if c in DIRECTIONS:
                self.fine_stage.begin(c, int(rest[0]))
            if c in self.servo_dir:
                self.servo_dir[c] = 1 if int(rest[1]) >= 0 else -1
            self.ser.write(f"#{cmd}\n".encode("ascii"))
            if c in TRANSFER_LETTERS:   # 정밀 스테이지 이동은 이송 팔의 측정 위치를 바꾸지 않는다
                self.revision += 1
            if c == "M":
                self.magnet = rest[0] == "1"
        self._note(f"보냄 #{cmd}")

    def epochs(self):
        with self.mail:
            return self.err_epoch, self.stop_epoch

    def wait_done(self, c, timeout, stop_evt=None, amount=None, since=None):
        """글자 c의 응답만 기다린다. 다른 축의 DONE은 그 축을 기다리는 쪽 몫이라 건드리지 않는다."""
        err0, stop0 = since if since is not None else self.epochs()
        end = time.time() + timeout
        with self.mail:
            while True:
                if stop_evt is not None and stop_evt.is_set():
                    raise InterruptedError("정지됨")
                box = self.mailbox.get(c, [])
                for r in list(box):
                    kind, _, n = r.split()
                    if kind == "STOP":
                        box.remove(r)
                        raise InterruptedError(f"보드가 정지(STOP {c} {n})")
                    if amount is None or n == str(amount):
                        box.remove(r)
                        return r
                if self.err_epoch != err0:
                    raise RuntimeError(f"보드 오류: {getattr(self, 'last_err', 'ERR')}")
                if self.stop_epoch != stop0:
                    raise InterruptedError("보드가 정지(STOP)")
                left = end - time.time()
                if left <= 0:
                    break
                self.mail.wait(min(left, 0.1))
        self.stop_letters(c)
        raise TimeoutError(f"{c}: {timeout:.0f}초 안에 DONE 없음 → 그 축 정지")

    def clear_replies(self, letters=None):
        # 그 글자의 옛 응답만 버린다 (다른 쪽이 기다리는 응답은 남긴다)
        with self.mail:
            for c in (letters if letters is not None else list(self.mailbox)):
                self.mailbox.pop(c, None)

    def stop_letters(self, letters):
        """그 글자들의 축만 정지 (펌웨어 v2 "#X <글자들>"). 다른 쪽 동작은 계속된다."""
        with self.lock:
            if self.ser:
                self.ser.write(f"#X {letters}\n".encode("ascii"))
                if self.moving and self.moving in letters:
                    self.moving = None
        self._note(f"부분 정지 {letters}")

    def line(self, cmd, timeout, stop_evt=None):
        c, amount, *_ = cmd.split()
        with self.lock:
            if stop_evt is not None and stop_evt.is_set():
                raise InterruptedError("정지됨")
            self.clear_replies([c])
            since = self.epochs()
            self.line_nowait(cmd)
        return self.wait_done(c, timeout, stop_evt, amount, since)

    def transfer_turn(self, rail, steps, ms, direction, stop_evt):
        # 레일 이동과 서보 회전을 동시에 보내고 둘 다 기다린다. 짧은 쪽이 먼저 끝나도 우편함에 남는다.
        with self.lock:
            if stop_evt.is_set():
                raise InterruptedError("정지됨")
            self.clear_replies([rail, "t"])
            since = self.epochs()
            self.line_nowait(f"{rail} {steps}")
            self.line_nowait(f"t {ms} {direction}")
        self.wait_done("t", ms / 1000 + 5, stop_evt, ms, since)
        self.wait_done(rail, steps / 10000 * 1.5 + 5, stop_evt, steps, since)

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
        # 정밀 앱과 같은 백엔드로 연다. 윈도우는 MSMF와 DirectShow의 카메라 번호가 다를 수 있다.
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {index} did not open")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.sample, self.error = None, None
        self.closed = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        while not self.closed.is_set():
            ok, bgr = self.cap.read()
            if ok:
                self.sample, self.error = (time.monotonic(), bgr), None
            else:
                self.error = "Camera read failed"
                self.closed.wait(0.1)

        self.cap.release()

    def close(self):
        self.closed.set()
        self.reader.join(timeout=1)


class Cameras:
    """카메라 3대 관찰용. 0.3초마다 화면을 JPEG로 내보낸다. 모터 제어와 무관."""

    def __init__(self):
        self.indices, self.size = [], (320, 240)
        self.views = {}  # index -> dict(capture, error)

    def open(self, indices, size=(640, 480)):
        size_unchanged = size == self.size
        self.indices, self.size = indices, size
        for i in indices:
            previous = self.views.get(i, {}).get("capture")
            if size_unchanged and previous and previous.sample and time.monotonic() - previous.sample[0] < 1:
                continue
            if previous:
                previous.close()
            try:
                self.views[i] = dict(capture=CameraCapture(i, size), error=None)
            except Exception as e:
                self.views[i] = dict(capture=None, error=str(e))

    def state(self):
        out = []
        for i, v in list(self.views.items()):
            cap = v["capture"]
            sample = cap.sample if cap else None
            frame = sample[1] if sample else None
            err = v["error"] or getattr(cap, "error", None)
            age = None if sample is None else round(time.monotonic() - sample[0], 1)
            out.append({"index": i, "error": err, "age": age,
                        "size": None if frame is None else [frame.shape[1], frame.shape[0]]})
        return out

    def frame_bgr(self, i):
        cap = self.views.get(i, {}).get("capture")
        return cap.sample[1] if cap and cap.sample else None

    def fresh_frames(self, indices):
        # Both frames must arrive after pickup/the request. Never use the last image of a lost USB device.
        started = time.monotonic()
        while time.monotonic() - started < 1.5:
            samples = [getattr(self.views.get(i, {}).get("capture"), "sample", None) for i in indices]
            stamps = [s[0] for s in samples if s and s[0] >= started]
            if len(stamps) == len(indices) and max(stamps) - min(stamps) <= 0.25:
                return [s[1] for s in samples]
            time.sleep(0.02)
        raise ValueError(f"카메라 {indices}: 새 프레임이 필요합니다")

    def jpeg(self, i):
        cap = self.views.get(i, {}).get("capture")
        frame = self.frame_bgr(i)
        if frame is None or time.monotonic() - cap.sample[0] > 1:
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
OPERATION = threading.Lock()


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
            self._json({"plan": seq().plan(), **seq().state, "prepared": seq().prepared})
        elif self.path == "/prealign/timing":
            from quarter_turn import quarter_turn_ms, read_timing
            from prealign_sequence import load
            cfg = load()["prealign"]
            ms = quarter_turn_ms(cfg)
            measured = read_timing(cfg)
            self._json({"quarter_turn_ms": ms, "directions": {d: {"ms": ms} for d in ("1", "-1")},
                        "measured_directions": measured["directions"] if measured else None})
        elif self.path == "/prealign/settings":
            from prealign_sequence import load
            self._json(load()["prealign"])
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
        emergency = self.path in ("/seq/stop", "/beat") or (self.path == "/cmd" and data.get("c") == "X")
        acquired = False
        try:
            if not emergency:
                acquired = OPERATION.acquire(blocking=False)
                if not acquired or seq().state["running"] or BOARD.moving:
                    raise RuntimeError("진행 중인 동작이 있습니다")
            if self.path == "/prealign/measure":
                from prealign_sequence import load
                return self._json({"ok": True, **seq().measure_angle(load()["prealign"], preview=True)})
            elif self.path == "/prealign/wafer":
                from prealign_sequence import load
                return self._json({"ok": True, **seq().vision.measure_wafer(CAMS, load()["prealign"])})
            elif self.path == "/prealign/calibration":
                from prealign_sequence import save_calibration
                save_calibration(data)
                seq().measured_settings = None
            elif self.path == "/cameras/open":
                width = data.get("width", CAMS.size[0])
                sizes = {320: (320, 240), 640: (640, 480), 1280: (1280, 720)}
                if width not in sizes:
                    raise ValueError("지원하지 않는 카메라 해상도")
                seq().measured_settings = None
                CAMS.open(CAMS.indices, sizes[width])
            elif self.path == "/connect":
                BOARD.connect(data["port"])
            elif self.path == "/disconnect":
                BOARD.disconnect()
            elif self.path == "/cmd":
                c = data["c"]
                if c not in MOVES | ONESHOT:
                    raise ValueError(f"알 수 없는 명령 {c!r}")
                if c == "X" and seq().state["running"]:
                    seq().stop()
                else:
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
            elif self.path == "/fine/teach":
                if not BOARD.port:
                    raise RuntimeError("보드를 연결하세요")
                BOARD.fine_stage.teach()
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
        finally:
            if acquired:
                OPERATION.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="시리얼 포트 (기본: 자동)")
    ap.add_argument("--standby", action="store_true", help="포트와 카메라는 패널에서 연결")
    ap.add_argument("--mock", action="store_true", help="보드 없이 가상 포트로 실행")
    ap.add_argument("--http", type=int, default=8790)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--cameras", type=int, nargs="*", default=[0, 1, 2],
                    help="열 카메라 번호 (기본 0 1 2, 빈 값이면 카메라 없이)")
    # 카메라 3대가 USB 2.0 허브 하나를 같이 쓰면 640×480 무압축은 대역폭을 넘어 1대가 영상을 못 받는다.
    ap.add_argument("--camera-size", type=int, nargs=2, default=[320, 240], metavar=("W", "H"))
    a = ap.parse_args()
    seq()
    threading.Thread(target=BOARD.watchdog, daemon=True).start()
    target = "mock" if a.mock else a.port or next((p["device"] for p in ports() if p["arduino"]), None)
    if target and not a.standby:
        try:
            BOARD.connect(target)
        except Exception as e:
            print("연결 실패:", e)
    CAMS.indices, CAMS.size = a.cameras, tuple(a.camera_size)
    if a.cameras and not a.standby:
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
