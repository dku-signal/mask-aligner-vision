"""웨이퍼 이송·프리얼라인 자동 시퀀스 (config/prealign_sequence.json).

보드(firmware/mega_previous)의 줄 명령 '#<c> <n>'을 하나씩 보내고 'DONE'을 기다린다.
스텝(레일)·ms(DC)는 펌웨어가 스스로 세어 멈추므로 PC 타이밍 오차가 없다.
"""
import json
import queue
import threading
import time
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config" / "prealign_sequence.json"
STEP_RATE = 10_000          # 펌웨어 레일 펄스: 50us 토글 = 10,000 스텝/초
AXIS_OF = {}                # 명령 글자 -> 축 이름


def load(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    AXIS_OF.clear()
    for name, ax in cfg["axes"].items():
        AXIS_OF[ax["plus"]] = AXIS_OF[ax["minus"]] = name
    return cfg


def resolve(cfg, c, n, ledger=None):
    """('Q','full') -> ('Q', 280000). 'full'은 축의 실측 최대값.
    ('X_RAIL_TO','box'|'stage'): 장부의 박스 위치를 기준으로 남은 거리만큼만 (끝에 부딪히지 않게)."""
    if c == "X_RAIL_TO":
        ax = cfg["axes"]["x_rail"]
        box = ledger.d.get("box_x") if ledger else None
        if box is None:   # 기준 모름: 박스 쪽 끝까지 밀어 기준을 잡는다 (이때만 끝에 닿음)
            return ("Q", int(ax["full"])) if n == "box" else ("W", int(ax["full"]))
        target = box + int(ax.get("box_margin", 0)) if n == "box" else box + int(ax["stage_from_box"])
        delta = target - ledger.d["pos"]["x_rail"]
        return ("W" if delta >= 0 else "Q"), abs(delta)
    if c == "ALIGN_X_TO":
        ax = cfg["axes"]["align_x"]
        left = ledger.d.get("align_left") if ledger else None
        if left is None:   # 왼쪽 끝을 모름: 먼저 왼쪽 끝까지 밀어 기준을 잡는다 (_run이 이어서 F를 보냄)
            return "B", int(ax["left_home_steps"])
        target = left if n == "left" else left + int(ax["receive_from_left"])
        delta = target - ledger.d["pos"]["align_x"]
        return ("F" if delta >= 0 else "B"), abs(delta)
    if c == "M":
        return c, int(n)
    if n == "full":
        full = cfg["axes"][AXIS_OF[c]]["full"]
        if not full:
            raise ValueError(f"{c}: 축 최대값(full)이 설정되지 않음")
        return c, int(full)
    return c, int(n)


def expected_s(cfg, c, n):
    if c == "M":
        return 0.0
    unit = cfg["axes"][AXIS_OF[c]]["unit"]
    return n / STEP_RATE if unit == "steps" else n / 1000


class Sequence:
    def __init__(self, board, cameras=None):
        self.board, self.cameras = board, cameras
        self.cfg = load()
        self.thread = None
        self.stop_evt = threading.Event()
        self.state = {"running": False, "current": None, "done": [], "error": None, "log": [], "rotation_deg": None}

    def _log(self, s):
        self.state["log"].insert(0, time.strftime("%H:%M:%S ") + s)
        del self.state["log"][40:]

    def plan(self):
        """화면 표시용: 단계별 실제 명령과 예상 시간."""
        out = []
        for st in self.cfg["steps"]:
            cmds = [resolve(self.cfg, c, n, self.board.ledger) for c, n in st["cmds"]]
            out.append({"no": st["no"], "title": st["title"],
                        "cmds": [f"{c} {n}" for c, n in cmds],
                        "seconds": round(sum(expected_s(self.cfg, c, n) for c, n in cmds), 1),
                        "prealign": bool(st.get("prealign"))})
        return out

    def start(self, only=None):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("시퀀스가 이미 실행 중")
        self.cfg = load()
        self.stop_evt.clear()
        self.state.update(running=True, current=None, done=[], error=None)
        self.thread = threading.Thread(target=self._run, args=(only,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        try:
            self.board.send("X")
        except RuntimeError:
            pass   # 보드 미연결이면 보낼 곳이 없다

    def _cmd(self, c, n):
        timeout = expected_s(self.cfg, c, n) * 1.5 + 5
        return self.board.line(f"{c} {n}", timeout, self.stop_evt)

    def _prealign_turn(self):
        p = self.cfg["prealign"]
        if not p.get("enabled"):
            self._log("프리얼라인 회전 건너뜀: " + p.get("_why_disabled", "비활성"))
            return None
        if self.cameras is None or p.get("camera") is None or not p.get("deg_per_s"):
            self._log("프리얼라인 회전 건너뜀: 카메라 또는 회전 속도 미설정")
            return None
        import sys
        sys.path.insert(0, str(CONFIG.parents[1]))
        from notch import measure_notch, rotation_to
        frame = self.cameras.frame_bgr(p["camera"])
        n = measure_notch(frame) if frame is not None else None
        if n is None:
            self._log("노치를 찾지 못해 회전 안 함")
            return None
        rot = rotation_to(p["target_deg"], n.angle_deg)
        ms = int(abs(rot) / p["deg_per_s"] * 1000)
        self._log(f"노치 {n.angle_deg:.1f}° -> 목표 {p['target_deg']}°: 회전 {rot:+.1f}° ({ms}ms)")
        self.state["rotation_deg"] = rot
        return ms, (1 if rot > 0 else -1)

    def restore(self):
        """원상 복구: Y 빼기 -> 회전 두 개 반대로 -> 정밀 스테이지 처음 위치 -> X 레일 박스 위치 -> Z 맨 아래."""
        if self.thread and self.thread.is_alive():
            raise RuntimeError("시퀀스가 실행 중")
        self.cfg = load()
        self.stop_evt.clear()
        self.state.update(running=True, current="복구", done=[], error=None)
        self.thread = threading.Thread(target=self._restore, daemon=True)
        self.thread.start()

    def _move_to(self, axis, target):
        ax = self.cfg["axes"][axis] if axis in self.cfg["axes"] else {"align_y": {"plus": "f", "minus": "b"}}[axis]
        delta = target - self.board.ledger.d["pos"][axis]
        if delta:
            c = ax["plus"] if delta > 0 else ax["minus"]
            self._log(f"  {axis}: {self.board.ledger.d['pos'][axis]} -> {target} ({c} {abs(delta)})")
            self._cmd(c, abs(delta))

    def _restore(self):
        led = self.board.ledger.d
        home = led.get("home") or {k: 0 for k in led["pos"]}
        try:
            self._log("원상 복구 시작")
            self._cmd("K", self.cfg["axes"]["y_linear"]["full"])
            self._move_to("y_rail", home["y_rail"])
            for c, name in (("t", "프리얼라인"), ("A", "정밀 θ")):
                net = led["servo_ms"][c]
                if net:
                    self.board.line(f"{c} {abs(net)} {-1 if net > 0 else 1}", abs(net) / 1000 + 5, self.stop_evt)
                    self._log(f"  {name} 서보 반대로 {abs(net)}ms")
            # 정밀 X는 왼쪽 끝으로 (다음 실행이 왼쪽 끝 + 32,000스텝 = 3.2초로 시작하도록)
            left = led.get("align_left")
            self._move_to("align_x", left if left is not None else home["align_x"])
            self._move_to("align_y", home["align_y"])
            box = led.get("box_x")
            margin = int(self.cfg["axes"]["x_rail"].get("box_margin", 0))
            self._move_to("x_rail", box + margin if box is not None else home["x_rail"])
            self._cmd("O", self.cfg["axes"]["z"]["full"])
            self._log("원상 복구 완료")
        except Exception as e:
            self.state["error"] = str(e)
            self._log(f"복구 중단: {e}")
            try:
                self.board.send("X")
            except Exception:
                pass
        finally:
            self.state.update(running=False, current=None)

    def _expand(self, cmds):
        """명령을 하나씩 계산해 내보낸다. 정밀 X 왼쪽 끝을 모르면 기준을 잡은 뒤 받는 위치로 간다."""
        for c, n in cmds:
            if c == "ALIGN_X_TO" and self.board.ledger.d.get("align_left") is None:
                ax = self.cfg["axes"]["align_x"]
                yield "B", int(ax["left_home_steps"])
                self.board.ledger.snapshot("align_left", self.board.ledger.d["pos"]["align_x"])
                self._log("  정밀 X 왼쪽 끝을 기준으로 기억")
                if n != "left":
                    yield "F", int(ax["receive_from_left"])
            else:
                yield resolve(self.cfg, c, n, self.board.ledger)

    def _run(self, only):
        try:
            if only is None:
                self.board.ledger.snapshot("home", dict(self.board.ledger.d["pos"]))
            for st in self.cfg["steps"]:
                if only is not None and st["no"] != only:
                    continue
                if self.stop_evt.is_set():
                    raise InterruptedError("정지됨")
                self.state["current"] = st["no"]
                self._log(f"{st['no']}단계 시작: {st['title']}")
                turn = self._prealign_turn() if st.get("prealign") else None
                for i, (c, n) in enumerate(self._expand(st["cmds"])):
                    if n == 0 and c != "M":   # 전자석 M 0(끄기)은 건너뛰면 안 된다
                        self._log(f"  {c} 0 (이미 목표 위치)")
                        continue
                    if turn and i == 0:
                        # 레일 이동(타이머)과 서보 회전은 펌웨어에서 동시에 진행된다
                        self.board.line_nowait(f"{c} {n}")
                        self.board.line(f"t {turn[0]} {turn[1]}", turn[0] / 1000 + 5, self.stop_evt)
                        self.board.wait_done(c, expected_s(self.cfg, c, n) * 1.5 + 5, self.stop_evt)
                    else:
                        self._cmd(c, n)
                    self._log(f"  {c} {n} 완료")
                self.state["done"].append(st["no"])
                if st["no"] == 1 and self.board.ledger.d.get("box_x") is None:
                    # 처음 한 번만: 끝까지 밀어 닿은 곳을 박스 끝으로 기억 (이후엔 그 앞에 멈춘다)
                    self.board.ledger.snapshot("box_x", self.board.ledger.d["pos"]["x_rail"])
            self._log("시퀀스 완료")
        except Exception as e:
            self.state["error"] = str(e)
            self._log(f"중단: {e}")
            try:
                self.board.send("X")
            except Exception:
                pass
        finally:
            self.state.update(running=False, current=None)
