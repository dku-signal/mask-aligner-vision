"""웨이퍼 이송·프리얼라인 자동 시퀀스 (config/prealign_sequence.json).

보드(firmware/mega_previous)의 줄 명령 '#<c> <n>'을 하나씩 보내고 'DONE'을 기다린다.
스텝(레일)·ms(DC)는 펌웨어가 스스로 세어 멈추므로 PC 타이밍 오차가 없다.
"""
import json
import math
from tempfile import NamedTemporaryFile
import threading
import time
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config" / "prealign_sequence.json"
CALIBRATION = CONFIG.parents[1] / "out" / "prealign_camera_calibration.json"
STEP_RATE = 10_000          # 펌웨어 레일 펄스: 50us 토글 = 10,000 스텝/초
AXIS_OF = {}                # 명령 글자 -> 축 이름


def load(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    if CALIBRATION.is_file():
        cfg["prealign"].update(calibration(json.loads(CALIBRATION.read_text())))
    for name, ax in cfg["axes"].items():
        AXIS_OF[ax["plus"]] = AXIS_OF[ax["minus"]] = name
    return cfg


def calibration(data):
    # Older panel files supplied only a direction for an unsigned quarter turn.
    # Preserve their camera mapping; the signed D8 polarity comes from this rig.
    offset, sign = data.get("camera_offset_deg"), data.get("camera_sign")
    if type(sign) is not int or sign not in (-1, 1) or type(offset) not in (int, float) or not math.isfinite(offset) or abs(offset) > 180:
        raise ValueError("카메라 방향 부호와 설치 각도(-180~180°)를 확인하세요")
    values = dict(camera_offset_deg=offset, camera_sign=sign)
    if "wafer_increasing_direction" in data:
        direction = data["wafer_increasing_direction"]
        if type(direction) is not int or direction not in (-1, 1):
            raise ValueError("웨이퍼 영상 각도가 증가하는 D8 방향을 선택하세요")
        values["wafer_increasing_direction"] = direction
    return values


def save_calibration(data):
    values = calibration(data)
    CALIBRATION.parent.mkdir(exist_ok=True)
    with NamedTemporaryFile("w", dir=CALIBRATION.parent, delete=False) as f:
        json.dump(values, f)
    Path(f.name).replace(CALIBRATION)


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
    unit = "steps" if c in ("f", "b") else cfg["axes"][AXIS_OF[c]]["unit"]
    return n / STEP_RATE if unit == "steps" else n / 1000


class Sequence:
    def __init__(self, board, cameras=None):
        self.board, self.cameras = board, cameras
        from key_prealign import KeyPrealign
        self.vision = KeyPrealign()
        self.cfg = load()
        self.station = None
        self.measured_settings = None
        self.measured_pair = None
        self.thread = None
        self.stop_evt = threading.Event()
        self.state = {"running": False, "current": None, "done": [], "error": None, "log": [], "rotation_deg": None, "theta_feedback": None}

    def _log(self, s):
        self.state["log"].insert(0, time.strftime("%H:%M:%S ") + s)
        del self.state["log"][40:]

    def plan(self):
        """화면 표시용: 단계별 실제 명령과 예상 시간."""
        out = []
        for st in self.cfg["steps"]:
            cmds, labels = [], []
            for c, n in st["cmds"]:
                if c == "FINE_STAGE":
                    try:
                        moves = self.fine_moves(n)
                    except ValueError:
                        labels.append("정밀 X·Y 오른쪽 위 기준 필요")
                        continue
                else:
                    moves = [resolve(self.cfg, c, n, self.board.ledger)]
                cmds.extend(moves)
                labels.extend(f"{c} {n}" for c, n in moves)
            out.append({"no": st["no"], "title": st["title"],
                        "cmds": labels,
                        "seconds": round(sum(expected_s(self.cfg, c, n) for c, n in cmds), 1),
                        "prealign": bool(st.get("prealign"))})
        return out

    def fine_moves(self, target):
        if not self.board.port or not self.board.fine_stage:
            raise ValueError("정밀 기준을 사용하려면 보드를 연결하세요")
        offset = self.cfg["axes"]["align_x"]["receive_from_right_top"] if target == "receive" else 0
        return self.board.fine_stage.moves(offset)

    @property
    def at_station(self):
        return self.board.ser is not None and self.station == (self.board.ser, self.board.revision)

    @property
    def prepared(self):
        return self.at_station and self.measured_settings is not None

    def invalidate(self):
        self.station = self.measured_settings = self.measured_pair = None

    def start(self, only=None):
        if self.thread and self.thread.is_alive():
            raise RuntimeError("시퀀스가 이미 실행 중")
        self.cfg = load()
        if only in (None, "fine-home") and "receive_from_right_top" in self.cfg["axes"]["align_x"]:
            self.fine_moves("home")
        if only in (None, 8, 9, "8-9", "theta", "theta-calibrate") and not self.board.port:
            raise ValueError("먼저 아두이노 포트를 연결하세요")
        if only in (None, 8, 9, "8-9", "theta", "theta-calibrate"):
            calibration(self.cfg["prealign"])
            if self.board.moving or self.board.prealign != "90°" or self.board.theta != "90°":
                raise ValueError("수동 구동을 정지한 뒤 자동 정렬을 시작하세요")
        if only in (8, "8-9") and not self.board.magnet:
            raise ValueError("8단계는 웨이퍼를 수취해 전자석을 켠 상태에서 시작하세요")
        if only == 9:
            if not self.prepared or self.measured_settings != self.cfg["prealign"]:
                raise ValueError("8단계 인출 완료 위치에서 측정한 각도가 필요합니다")
        else:
            self.invalidate()
            self.state["rotation_deg"] = None
        self.stop_evt.clear()
        self.state.update(running=True, current=None, done=[], error=None, theta_feedback=None)
        self.thread = threading.Thread(target=self._run, args=(only,), daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_evt.set()
        self.invalidate()
        try:
            self.board.send("X")
        except RuntimeError:
            pass   # 보드 미연결이면 보낼 곳이 없다

    def _cmd(self, c, n):
        timeout = expected_s(self.cfg, c, n) * 1.5 + 5
        return self.board.line(f"{c} {n}", timeout, self.stop_evt)

    def measure_angle(self, settings, preview=False):
        # A preview can retry a failed step-8 measurement while the wafer is still here.
        station = self.station if self.at_station else None
        self.measured_settings = None
        if self.board.moving or self.board.prealign != "90°" or self.board.theta != "90°":
            raise ValueError("레일과 수동 회전 서보를 정지한 뒤 각도를 측정하세요")
        if station is not None and not self.board.magnet:
            raise ValueError("수취한 웨이퍼의 전자석 ON 상태를 확인하세요")
        if station is None and preview:
            return self.vision.measure(self.cameras, settings, preview=True)
        result = self.vision.stable(self.cameras, settings, self.stop_evt, preview=preview)
        if station is not None and station == self.station and self.at_station and not result["error"]:
            self.state["rotation_deg"] = result["turn_deg"]
            self.measured_pair = result
            self.measured_settings = settings.copy()
            self._log(f"8단계 완료 후 측정: 마스크 {result['mask']['angle']}° · 웨이퍼 {result['wafer']['angle']}° → 9단계 θ {result['turn_deg']}°")
        return result

    def _align_theta(self, first=None, transfer_steps=None):
        from key_prealign import axis_difference
        from quarter_turn import quarter_turn_ms, VERIFY_TOLERANCE
        cfg = self.cfg["prealign"]
        first = first or self.vision.stable(self.cameras, cfg, self.stop_evt)
        error = first["turn_deg"]
        turn = error if abs(error) > cfg["parallel_tolerance_deg"] else 0.0
        feedback = dict(mask=first["mask"]["angle"], wafer=first["wafer"]["angle"],
                        error_deg=error, planned_deg=turn, pulses=0, verified=False, transfer=transfer_steps is not None)
        self.state["theta_feedback"] = feedback
        if self.stop_evt.is_set():
            raise InterruptedError("정지됨")
        remaining = transfer_steps
        if turn:
            direction = (1 if turn > 0 else -1) * cfg["camera_sign"] * cfg["wafer_increasing_direction"]
            quarter_ms = quarter_turn_ms(cfg)
            target = abs(turn)
            ms = max(1, math.floor(quarter_ms * target / 90 + 0.5))
            feedback.update(ms=ms, direction=direction)
            self._log(f"YOLO 보정각 {turn:+.2f}° · D8 {direction:+d} {ms}ms 1회")
            if remaining is None:
                self.board.line(f"t {ms} {direction}", ms / 1000 + 5, self.stop_evt)
            else:
                segment = math.ceil((ms + 20) * STEP_RATE / 1000)
                if remaining < segment:
                    raise ValueError(f"θ {turn:+.2f}° 보정에 필요한 X 이송 구간이 부족합니다")
                self.board.transfer_turn("W", segment, ms, direction, self.stop_evt)
                remaining -= segment
            feedback["pulses"] = 1
        if remaining is not None:
            if self.stop_evt.is_set():
                raise InterruptedError("정지됨")
            if remaining:
                self._log(f"9단계 {'D8 구동 완료' if turn else 'θ 생략'} · W {remaining} 이송")
                self._cmd("W", remaining)
            return
        if turn:
            if self.stop_evt.wait(0.6):
                raise InterruptedError("정지됨")
            final = self.vision.stable(self.cameras, cfg, self.stop_evt)
            # Unwrap the measured axis displacement around the requested turn.
            # This keeps a 91° response to a 90° command on the same branch.
            image_sign = direction * cfg["wafer_increasing_direction"]
            observed = image_sign * (final["wafer"]["angle"] - first["wafer"]["angle"])
            miss = axis_difference(observed, target)
            travel = target + miss
            feedback.update(mask=final["mask"]["angle"], wafer=final["wafer"]["angle"],
                            error_deg=final["turn_deg"], measured_deg=round(travel, 2))
            if abs(miss) > VERIFY_TOLERANCE or abs(final["turn_deg"]) > VERIFY_TOLERANCE:
                raise ValueError(f"θ 검증 실패 · 목표 회전량 {target:.2f}° · 구동 방향 실측 {travel:.2f}° · 잔차 {final['turn_deg']:+.2f}°")
        if self.stop_evt.is_set():
            raise InterruptedError("정지됨")
        self.state["theta_feedback"] = dict(feedback, verified=True)
        self._log(f"θ {turn:+.2f}° 보정 완료 · 잔차 {feedback['error_deg']:+.2f}°")

    def restore(self):
        """원상 복구: Y 빼기 -> 회전 두 개 반대로 -> 정밀 스테이지 처음 위치 -> X 레일 박스 위치 -> Z 맨 아래."""
        if self.thread and self.thread.is_alive():
            raise RuntimeError("시퀀스가 실행 중")
        self.cfg = load()
        self.stop_evt.clear()
        self.invalidate()
        self.state.update(running=True, current="복구", done=[], error=None, rotation_deg=None, theta_feedback=None)
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
            if "receive_from_right_top" in self.cfg["axes"]["align_x"]:
                for c, n in self.fine_moves("home"):
                    self._cmd(c, n)
            else:
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
            if c == "FINE_STAGE":
                yield from self.fine_moves(n)
                continue
            if c == "ALIGN_X_TO" and n == "left" and self.board.ledger.d.get("align_left") is not None:
                # 왼쪽 끝까지 확실히 밀고(끝 + 여유) 그 자리를 다시 기준으로 잡는다
                ax = self.cfg["axes"]["align_x"]
                led = self.board.ledger.d
                yield "B", max(0, led["pos"]["align_x"] - led["align_left"]) + int(ax.get("left_overshoot", 2000))
                self.board.ledger.snapshot("align_left", led["pos"]["align_x"])
                continue
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
            if only == "fine-home":
                self.state["current"] = "정밀 X·Y 초기화"
                for c, n in self.fine_moves("home"):
                    self._cmd(c, n)
                self._log("정밀 X·Y 오른쪽 위 복귀 완료")
                return
            if only == "theta-calibrate":
                from quarter_turn import calibrate
                self.state["current"] = "90° 시간 측정"
                result = calibrate(self.board, self.cameras, self.vision, self.cfg["prealign"],
                                   self.stop_evt, self._log)
                self._log("90° 실측 기록 · " + " / ".join(
                    f"{d:+d} {trial['ms']}ms" for d, trial in result["directions"].items()))
                return
            if only == "theta":
                self.state["current"] = "자동 θ"
                self._align_theta()
                return
            if only is None:
                self.board.ledger.snapshot("home", dict(self.board.ledger.d["pos"]))
            selected = (8, 9) if only == "8-9" else (only,)
            for st in self.cfg["steps"]:
                if only is not None and st["no"] not in selected:
                    continue
                if self.stop_evt.is_set():
                    raise InterruptedError("정지됨")
                self.state["current"] = st["no"]
                self._log(f"{st['no']}단계 시작: {st['title']}")
                for i, (c, n) in enumerate(self._expand(st["cmds"])):
                    if st.get("prealign") and i == 0:
                        if not self.prepared or self.measured_settings != self.cfg["prealign"]:
                            raise ValueError("8단계 완료 후 측정된 각도가 없어 X 이송을 시작할 수 없습니다")
                        pair = self.measured_pair
                        self.invalidate()
                        if c != "W":
                            raise ValueError("θ 자동 보정에는 전진 X 이송 구간이 필요합니다")
                        self._align_theta(first=pair, transfer_steps=n)
                    elif n == 0 and c != "M":
                        self._log(f"  {c} 0 (이미 목표 위치)")
                        continue
                    else:
                        self._cmd(c, n)
                    self._log(f"  {c} {n} 완료")
                self.state["done"].append(st["no"])
                if st["no"] == 8:
                    if self.stop_evt.is_set():
                        raise InterruptedError("정지됨")
                    self.station = (self.board.ser, self.board.revision)
                    self.state["current"] = "각도 측정"
                    self._log("8단계 Y 후퇴 완료 · 정지 상태에서 두 키 측정")
                    result = self.measure_angle(self.cfg["prealign"])
                    if self.stop_evt.is_set():
                        self.invalidate()
                        raise InterruptedError("정지됨")
                    if result["error"]:
                        raise ValueError(result["error"])
                if st["no"] == 1 and self.board.ledger.d.get("box_x") is None:
                    # 원상 복구에서 사용할 박스 쪽 기준을 기억한다.
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
