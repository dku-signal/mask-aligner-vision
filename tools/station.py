"""통합 스테이션: 정밀 얼라인 + 이송·프리얼라인을 한 프로그램에서 연속 운전한다.

    python tools/station.py                      # 아두이노 자동, 카메라 0(정밀·마스크) 1(웨이퍼)
    python tools/station.py --mock               # 보드 없이 (가상 보드, 20배 빠름)
    python tools/station.py --detector yolo-2pair --frame 1920x1080   (기본값)

왜 한 프로그램인가
  - 보드(시리얼 포트)는 한 프로그램만 열 수 있다. 예전에는 정밀 앱(8765)과 이송 패널(8790)이
    각자 포트를 열어서 번갈아 넘겨줘야 했다. 여기서는 이송 패널의 Board 하나가 포트를 갖고,
    정밀 앱은 BusLink로 같은 보드에 명령을 보낸다. 응답은 축 글자별 우편함으로 나뉘어 섞이지 않는다.
  - 카메라 0은 정밀 앱이 열고, 프리얼라인 8단계 측정은 같은 프레임을 빌려 쓴다.
  - 펌웨어 v2(firmware/mega_previous)는 레일 4개·DC 3쌍·서보 2개가 동시에 움직이고, "#X <글자들>"로
    한쪽만 멈출 수 있다. 그래서 정밀 얼라인이 웨이퍼 N을 맞추는 동안 이송 팔이 N+1을 가져온다.

연속 운전 (Pipeline)
  가져오기(0~8단계, 8단계에서 두 키 각도 측정) → 정밀 스테이지가 비면 넘겨주기(9~14단계, 이송 중 θ 보정)
  → 정밀 얼라인 START → 정렬하는 동안 다음 웨이퍼 가져오기 → … 반복
  정렬이 끝난 웨이퍼를 스테이지에서 치우는 장치는 없으므로, 기본값은 "스테이지 비움 확인" 버튼을 누른 뒤
  다음 웨이퍼를 넘겨준다 (설정에서 자동 진행으로 바꿀 수 있음).

알고리즘·수치는 손대지 않는다: 정밀 얼라인은 mask_aligner_optimized_v4.MaskAlignerApp 그대로,
이송은 config/prealign_sequence.json의 단계·값 그대로다.
"""
import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import motor_panel as panel                       # noqa: E402  이송 패널 (Board·CAMS·Handler)
from prealign_sequence import DELIVER_STEPS, FETCH_STEPS, MEASURE_STEPS, TRANSFER_LETTERS   # noqa: E402

FINE_LETTERS = "FBfbAud"     # 정밀 얼라인이 쓰는 축: 정밀 X·Y, θ 서보(D5), 정밀 Z(Z1)
PIPELINE_FILE = ROOT / "out" / "pipeline.json"


# ---------------------------------------------------------------------------
# 정밀 앱 ↔ 공용 보드
# ---------------------------------------------------------------------------
class _FineStageView:
    """정밀 앱에 보여 줄 정밀 스테이지 장부. 시작·완료 기록은 Board가 하므로 여기선 읽기만 한다."""

    def __init__(self, board):
        self.board = board

    def __getattr__(self, name):
        return getattr(self.board.fine_stage, name)

    @property
    def data(self):
        return self.board.fine_stage.data if self.board.fine_stage else {}

    def begin(self, letter, steps):
        pass   # Board.line_nowait가 기록한다 (두 번 기록하면 장부가 무효가 된다)

    def ack(self, letter, steps, partial=False):
        pass   # Board 읽기 스레드가 DONE/STOP을 받을 때 기록한다


class BusLink:
    """mask_aligner_optimized_v4.ArduinoLink와 같은 모양으로 공용 Board에 명령을 보낸다.

    정밀 얼라인은 정밀 축(FINE_LETTERS)만 움직일 수 있다. 정지는 그 축만 멈춘다(이송은 계속).
    """

    dialect = "previous"

    def __init__(self, board, to_previous):
        self.board, self.to_previous = board, to_previous
        self.last_error = ""
        self.last_rx = ""
        self.pt_done = set()
        self.pending = None      # (글자, 값, 보낸 시점의 오류·정지 번호)
        self.fine_stage = _FineStageView(board)

    @property
    def port(self):
        return self.board.port

    def connected(self):
        return self.board.ser is not None

    def send(self, cmd):
        cmd = cmd.strip()
        if cmd.lower() in ("s", "stop"):
            self.pending = None
            try:
                self.board.stop_letters(FINE_LETTERS)
            except Exception:
                pass
            return True
        line, reply = self.to_previous(cmd)
        if line is None:
            self.last_error = reply
            return False
        letter, amount = line[1], line[1:].split()[1]
        if letter not in FINE_LETTERS:
            self.last_error = f"통합 스테이션: 정밀 앱은 정밀 축만 움직입니다 ({cmd} → 이송 패널에서)"
            return False
        try:
            with self.board.lock:
                self.board.clear_replies([letter])
                since = self.board.epochs()
                self.board.line_nowait(line[1:])
        except Exception as exc:
            self.last_error = f"SERIAL WRITE FAILED: {exc}"
            return False
        self.last_error = ""
        self.pending = (letter, amount, since)
        return True

    def wait_done(self, stop_event, timeout=25.0, expected="DONE"):
        if self.pending is None:
            return True     # 기다릴 응답 없음 ('' 명령 등)
        letter, amount, since = self.pending
        self.pending = None

        class _Either:     # 정밀 앱의 STOP 이벤트로도 기다림을 끝낸다
            def is_set(self_inner):
                return stop_event.is_set()
        try:
            reply = self.board.wait_done(letter, timeout, _Either(), amount, since)
            self.last_rx = reply
            return True
        except InterruptedError as exc:
            if stop_event.is_set():
                self.board.stop_letters(letter)
                self.last_error = "Cancelled by STOP"
            else:
                self.last_error = f"Arduino replied: {exc}"
            return False
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def read_line(self):
        return ""

    def stop(self):
        self.send("s")

    def close(self):
        self.stop()


class _SharedCapture:
    """CAMS(이송 패널 카메라)에 정밀 앱 카메라 프레임을 빌려주는 보기 전용 캡처."""

    def __init__(self, app, size=None):
        self.app, self.size = app, size

    @property
    def sample(self):
        latest = self.app.latest_frame
        if not latest:
            return None
        frame = latest[1]
        # 정밀 앱은 1920x1080으로 잡지만 패널 화면·8단계 측정은 원래대로 패널 크기(640x480)로 본다
        if self.size and (frame.shape[1], frame.shape[0]) != tuple(self.size):
            frame = cv2.resize(frame, tuple(self.size), interpolation=cv2.INTER_AREA)
        return (self.app.capture_time, frame)

    @property
    def error(self):
        return None if self.app.latest_frame else "정밀 카메라 대기 중"

    def close(self):
        pass


def share_camera(cams, app, index):
    """카메라 index는 정밀 앱이 연다. CAMS가 다시 열려고 해도 같은 프레임을 빌려 쓰게 만든다."""
    original_open = cams.open

    def open_without_shared(indices, size=(640, 480)):
        original_open([i for i in indices if i != index], size)
        cams.indices = list(indices)
        cams.views[index] = dict(capture=_SharedCapture(app, size), error=None)

    cams.open = open_without_shared
    cams.views[index] = dict(capture=_SharedCapture(app, getattr(cams, "size", None)), error=None)


# ---------------------------------------------------------------------------
# 연속 운전
# ---------------------------------------------------------------------------
class AlignFailed(RuntimeError):
    """정밀 얼라인 실패 — 박스가 빈 것(8단계 미검출)과 구별한다."""


class Pipeline:
    # 정밀 얼라인의 마스크 판(정밀 Z) 내림: hard = 끝까지(정밀 앱 기본 ALIGN_Z_DOWN_MS),
    # soft = hard보다 덜 내림(cfg soft_down_ms). 올림은 내린 만큼.
    MODES = {"hard": "하드 컨택", "soft": "소프트 컨택"}

    def __init__(self, seq, app, board, z_setter=None, hard_ms=(5000, 5000)):
        self.seq, self.app, self.board = seq, app, board
        self.z_setter, self.hard_ms = z_setter, tuple(hard_ms)
        self.mode = "hard"
        self.thread = None
        self.stop_evt = threading.Event()
        self.cleared = threading.Event()
        self.cfg = dict(max_wafers=0, auto_clear=False, start_timeout_s=20.0, soft_down_ms=3500)
        try:
            self.cfg.update(json.loads(PIPELINE_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
        self.state = dict(running=False, phase="대기", error=None, log=[],
                          fetched=0, delivered=0, aligned=0, holding=False, stage="비어 있음", results=[])

    def save_cfg(self, values):
        new = dict(self.cfg)
        for k in ("max_wafers", "auto_clear", "soft_down_ms"):
            if k in values:
                new[k] = bool(values[k]) if k == "auto_clear" else int(values[k])
        if not 0 < new["soft_down_ms"] < self.hard_ms[0]:
            raise ValueError(f"소프트 컨택 내림 시간은 0보다 크고 하드 컨택({self.hard_ms[0]}ms)보다 작아야 합니다")
        self.cfg = new
        PIPELINE_FILE.parent.mkdir(exist_ok=True)
        PIPELINE_FILE.write_text(json.dumps(self.cfg), encoding="utf-8")

    def _log(self, s):
        self.state["log"].insert(0, time.strftime("%H:%M:%S ") + s)
        del self.state["log"][60:]

    def status(self):
        app = self.app
        down, up = self.z_ms()
        return dict(self.state, cfg=self.cfg, mode=self.mode, mode_name=self.MODES[self.mode],
                    z_down_ms=down, z_up_ms=up, hard_down_ms=self.hard_ms[0],
                    fine=dict(phase=app.phase, status=app.status, auto_running=app.auto_running,
                              mask=app.mask_detected, wafer=app.wafer_detected),
                    transfer=dict(running=self.seq.state["running"], current=self.seq.state["current"],
                                  error=self.seq.state["error"]))

    def z_ms(self):
        if self.mode == "soft":
            ms = int(self.cfg["soft_down_ms"])
            return ms, ms
        return self.hard_ms

    # ---- 시작·정지 ----
    def start(self, mode="hard"):
        if mode not in self.MODES:
            raise ValueError(f"알 수 없는 모드 {mode!r}")
        if self.thread and self.thread.is_alive():
            raise RuntimeError("연속 운전이 이미 실행 중")
        if self.seq.state["running"]:
            raise RuntimeError("이송 시퀀스가 실행 중입니다")
        if not self.board.port:
            raise RuntimeError("먼저 아두이노 포트를 연결하세요")
        self.mode = mode
        self.stop_evt.clear()
        self.state.update(running=True, error=None, phase="시작")
        down, up = self.z_ms()
        self._log(f"연속 운전 시작: {self.MODES[mode]} (마스크 판 내림 {down}ms · 올림 {up}ms)")
        self.thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self.thread.start()

    def stop(self, reason="정지"):
        self.stop_evt.set()
        if self.seq.state["running"]:
            self.seq.stop()
        self.state["phase"] = reason

    def mark_cleared(self):
        self.cleared.set()
        self.state["stage"] = "비어 있음"
        self._log("작업자 확인: 정밀 스테이지 비움")

    # ---- 내부 ----
    def _wait(self, cond, what, timeout=None, poll=0.2):
        end = None if timeout is None else time.time() + timeout
        while not cond():
            if self.stop_evt.is_set():
                raise InterruptedError("연속 운전 정지")
            if end is not None and time.time() > end:
                raise TimeoutError(f"{what}: 시간 초과")
            time.sleep(poll)

    def _transfer(self, steps, label):
        self.state["phase"] = label
        self._log(f"{label} 시작 ({steps[0]}~{steps[-1]}단계)")
        self.seq.start(steps)
        self._wait(lambda: not self.seq.state["running"], label)
        if self.seq.state["error"]:
            raise RuntimeError(f"{label} 실패: {self.seq.state['error']}")
        self._log(f"{label} 완료")

    def _fine_idle(self):
        return not self.app.auto_running and not self.app.moving

    def _note_align_result(self):
        app = self.app
        if self.state["stage"] == "정렬 중" and self._fine_idle():
            ok = app.phase == "DONE"
            self.state["aligned"] += ok
            self.state["stage"] = "정렬 완료 (치워야 함)" if ok else f"정렬 실패: {app.status}"
            self.state["results"].insert(0, dict(wafer=self.state["delivered"], ok=ok, status=app.status,
                                                 at=time.strftime("%H:%M:%S")))
            self._log(f"정밀 얼라인 {'완료' if ok else '실패'}: {app.status}")
            if not ok:
                raise AlignFailed(f"정밀 얼라인 실패 — 연속 운전을 멈춥니다: {app.status}")

    def align_only(self, mode="hard"):
        """스테이지에 이미 놓인 웨이퍼 하나만 정밀 얼라인 (컨택 모드 적용, 이송 없음)."""
        if mode not in self.MODES:
            raise ValueError(f"알 수 없는 모드 {mode!r}")
        if self.thread and self.thread.is_alive():
            raise RuntimeError("연속 운전이 실행 중")
        self.mode = mode
        self.stop_evt.clear()
        down, up = self.z_ms()
        self._log(f"정밀 얼라인만: {self.MODES[mode]} (마스크 판 내림 {down}ms · 올림 {up}ms)")
        self._start_fine()

    def _start_fine(self):
        app = self.app
        if self.z_setter:
            self.z_setter(*self.z_ms())
        self.state["phase"] = "정밀 얼라인 START"
        deadline = time.time() + self.cfg["start_timeout_s"]
        while time.time() < deadline:
            if self.stop_evt.is_set():
                raise InterruptedError("연속 운전 정지")
            app.start_align()
            if app.auto_running:
                self.state["stage"] = "정렬 중"
                self._log("정밀 얼라인 START")
                return
            time.sleep(1.0)   # 키가 아직 안 보이면 (Z·전자석 직후) 잠시 뒤 다시
        raise RuntimeError(f"정밀 얼라인을 시작하지 못함: {app.status}")

    def _run(self):
        holding = False
        try:
            while not self.stop_evt.is_set():
                limit = self.cfg["max_wafers"]
                if not holding:
                    if limit and self.state["fetched"] >= limit:
                        break
                    try:
                        self._transfer(FETCH_STEPS, "웨이퍼 가져오기")
                        # 8단계는 카메라 0에서 마스크를 재므로 정밀 얼라인이 끝나 마스크 Z가 올라간 뒤에 한다
                        self.state["phase"] = "정밀 얼라인 종료 대기 (끝나면 8단계 측정)"
                        self._wait(lambda: self._fine_idle(), "정밀 얼라인 종료 대기")
                        self._note_align_result()
                        self._transfer(MEASURE_STEPS, "각도 측정 (8단계)")
                    except AlignFailed:
                        raise
                    except RuntimeError as exc:
                        if "웨이퍼" in str(exc) and ("검출" in str(exc) or "못" in str(exc)):
                            self._log(f"웨이퍼를 찾지 못함 → 박스가 빈 것으로 보고 종료: {exc}")
                            break
                        raise
                    holding = True
                    self.state.update(holding=True, fetched=self.state["fetched"] + 1)

                self.state["phase"] = "정밀 스테이지 대기"
                self._wait(lambda: self._fine_idle(), "정밀 얼라인 종료 대기")
                self._note_align_result()
                if self.state["stage"].startswith("정렬 완료") and not self.cfg["auto_clear"]:
                    self.cleared.clear()
                    self.state["phase"] = "작업자: 정렬된 웨이퍼를 치우고 '스테이지 비움 확인'"
                    self._wait(self.cleared.is_set, "스테이지 비움 확인")
                self.state["stage"] = "비어 있음"

                self._transfer(DELIVER_STEPS, "웨이퍼 넘겨주기")
                holding = False
                self.state.update(holding=False, delivered=self.state["delivered"] + 1, stage="놓임")
                self._start_fine()
                # 정렬하는 동안 다음 웨이퍼를 가져온다 (루프 처음으로)

            self.state["phase"] = "마지막 정렬 종료 대기"
            self._wait(lambda: self._fine_idle(), "정밀 얼라인 종료 대기")
            self._note_align_result()
            self.state["phase"] = "완료"
            self._log(f"연속 운전 종료: 가져옴 {self.state['fetched']} · 넘김 {self.state['delivered']} · 정렬 {self.state['aligned']}")
        except InterruptedError as exc:
            self.state["phase"] = "정지"
            self._log(str(exc))
        except Exception as exc:
            self.state.update(error=str(exc), phase="멈춤 (오류)")
            self._log(f"멈춤: {exc}")
            if self.seq.state["running"]:
                self.seq.stop()
        finally:
            self.state["running"] = False


# ---------------------------------------------------------------------------
# HTTP: 8790 = 통합 화면(이송 패널 + 연속 운전 + 정밀 화면), 8765 = 정밀 앱 원래 화면
# ---------------------------------------------------------------------------
PIPE = None
FINE_APP = None
FINE_PORT = 8765


class StationHandler(panel.Handler):
    def do_GET(self):
        if self.path == "/pipeline":
            return self._json(dict(PIPE.status(), station=True, fine_url=f"http://127.0.0.1:{FINE_PORT}/"))
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/pipeline") or self.path == "/station/stop":
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path == "/pipeline/start":
                    if "soft_down_ms" in data:
                        PIPE.save_cfg({"soft_down_ms": data["soft_down_ms"]})
                    PIPE.start(data.get("mode", "hard"))
                elif self.path == "/pipeline/align":
                    if "soft_down_ms" in data:
                        PIPE.save_cfg({"soft_down_ms": data["soft_down_ms"]})
                    PIPE.align_only(data.get("mode", "hard"))
                elif self.path == "/pipeline/stop":
                    PIPE.stop()
                elif self.path == "/pipeline/cleared":
                    PIPE.mark_cleared()
                elif self.path == "/pipeline/config":
                    PIPE.save_cfg(data)
                elif self.path == "/station/stop":
                    emergency_stop()
                else:
                    return self._json({"error": "not found"}, 404)
                return self._json({"ok": True})
            except Exception as exc:
                return self._json({"ok": False, "error": str(exc)}, 400)
        if self.path == "/cmd" and self._peek_x():
            emergency_stop()
            return self._json({"ok": True, **panel.BOARD.status()})
        return super().do_POST()

    def _peek_x(self):
        # 빨간 전체 정지(X)는 연속 운전·정밀 얼라인·이송을 모두 멈춘다
        import io
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        self.rfile = io.BytesIO(raw)   # 부모 핸들러가 같은 본문을 다시 읽는다
        try:
            return json.loads(raw or b"{}").get("c") in ("X", "x")
        except Exception:
            return False


def emergency_stop():
    if PIPE:
        PIPE.stop("전체 정지")
    if FINE_APP:
        FINE_APP.stop_align()
    if panel.SEQ and panel.SEQ.state["running"]:
        panel.SEQ.stop()
    try:
        panel.BOARD.send("X")
    except Exception:
        pass


def main():
    global PIPE, FINE_APP, FINE_PORT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", help="시리얼 포트 (기본: 자동)")
    ap.add_argument("--mock", action="store_true", help="보드 없이 가상 보드")
    ap.add_argument("--http", type=int, default=8790, help="통합 화면 포트")
    ap.add_argument("--fine-http", type=int, default=8765, help="정밀 얼라인 화면 포트")
    ap.add_argument("--fine-camera", type=int, default=0, help="정밀 얼라인 카메라 (프리얼라인 마스크 측정도 이 영상)")
    ap.add_argument("--cameras", type=int, nargs="*", default=[0, 1], help="이송 패널에 보일 카메라")
    ap.add_argument("--camera-size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    ap.add_argument("--finish-frames", type=int, default=3,
                    help="정밀 얼라인 완료: 중심 일치를 연속 몇 번 확인할지 (정밀 앱 단독 기본 8)")
    ap.add_argument("--finish-px", type=float, default=1.0,
                    help="정밀 얼라인 완료: 마스크·웨이퍼 중심 차이 허용 px (calibration.json 기본 0.65)")
    ap.add_argument("--detector", default="yolo-2pair", help="정밀 얼라인 검출기 (mask_aligner_optimized_v4 --detector)")
    ap.add_argument("--frame", default="1920x1080",
                    help="정밀 카메라 해상도 WxH. calibration.json(스텝/px)과 YOLO 키 크기가 1920x1080 기준")
    ap.add_argument("--skip-theta", action="store_true")
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    import mask_aligner_optimized_v4 as fine
    fine.load_calibration(str(ROOT / "calibration.json"))
    # 연속 운전용 정밀 얼라인 완료 기준 (다음 웨이퍼의 8단계가 정밀 종료를 기다리므로 빨리 끝낸다)
    fine.WAFER_LOST_DONE_FRAMES = a.finish_frames
    fine.DEAD_BAND_PX = a.finish_px
    fine.HTML_PAGE = fine.HTML_PAGE.replace("${d.wafer_lost_count}/8", "${d.wafer_lost_count}/%d" % a.finish_frames)
    print(f"정밀 얼라인 완료 기준: 중심 {a.finish_px:g}px 이내 {a.finish_frames}회 연속")
    fine.load_rig_config(str(ROOT / "rig_config.json"))
    if a.frame:
        fine.FRAME_WIDTH, fine.FRAME_HEIGHT = (int(v) for v in a.frame.lower().split("x"))

    # 1) 보드는 이송 패널의 Board 하나만 연다
    seq = panel.seq()
    threading.Thread(target=panel.BOARD.watchdog, daemon=True).start()
    target = "mock" if a.mock else a.port or next((p["device"] for p in panel.ports() if p["arduino"]), None)
    if target:
        panel.BOARD.connect(target)
    print("보드:", panel.BOARD.port or "없음")

    # 2) 정밀 앱은 공용 보드로 (ArduinoLink 대신 BusLink)
    to_previous = fine.ArduinoLink.to_previous
    fine.ArduinoLink = lambda *args, **kwargs: BusLink(panel.BOARD, to_previous)
    app = fine.MaskAlignerApp(camera_index=a.fine_camera, detector=a.detector, skip_theta=a.skip_theta)
    FINE_APP = app
    if sys.platform == "darwin":
        app.ensure_camera()       # macOS 카메라 권한 요청은 메인 스레드에서
    app.start()

    # 3) 카메라: 정밀 카메라는 정밀 앱이 열고 CAMS는 빌려 쓴다, 나머지는 CAMS가 연다
    panel.CAMS.indices, panel.CAMS.size = a.cameras, tuple(a.camera_size)
    share_camera(panel.CAMS, app, a.fine_camera)
    panel.CAMS.open(a.cameras, tuple(a.camera_size))
    for c in panel.CAMS.state():
        print(f"  카메라 {c['index']}:", c["error"] or "열림")

    def set_align_z(down, up):
        fine.ALIGN_Z_DOWN_MS, fine.ALIGN_Z_UP_MS = int(down), int(up)
    PIPE = Pipeline(seq, app, panel.BOARD, z_setter=set_align_z,
                    hard_ms=(fine.ALIGN_Z_DOWN_MS, fine.ALIGN_Z_UP_MS))
    FINE_PORT = a.fine_http

    fine_srv = ThreadingHTTPServer(("127.0.0.1", a.fine_http), fine.RequestHandler)
    fine_srv.app = app
    threading.Thread(target=fine_srv.serve_forever, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.http), StationHandler)
    url = f"http://127.0.0.1:{a.http}/"
    print(f"통합 화면: {url}  (정밀 얼라인 원래 화면: http://127.0.0.1:{a.fine_http}/)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        emergency_stop()
        app.shutdown()
        panel.BOARD.disconnect()
        panel.CAMS.close()


if __name__ == "__main__":
    main()
