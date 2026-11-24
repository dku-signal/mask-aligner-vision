"""통합 스테이션: 공용 보드에서 정밀·이송이 동시에 돌고, 연속 운전이 웨이퍼를 겹쳐 처리하는지."""
import threading
import time
from pathlib import Path

import pytest

from tools import motor_panel, prealign_sequence


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(motor_panel.__file__).parent))
    monkeypatch.setattr(motor_panel, "LEDGER_FILE", tmp_path / "ledger.json")
    stage = motor_panel.FineStage("mock", tmp_path / "fine.json")
    stage.teach()
    monkeypatch.setattr(motor_panel.FineStage, "for_port", lambda port: stage)
    monkeypatch.setattr(motor_panel.MockSerial, "SPEED", 200)
    b = motor_panel.Board()
    b.connect("mock")
    yield b
    b.disconnect()


def test_fine_and_transfer_wait_for_their_own_replies(board):
    """정밀(F)과 이송(Q)을 동시에 보내도 각자 자기 DONE만 받는다."""
    results = {}

    def run(cmd):
        results[cmd] = board.line(cmd, 5)
    threads = [threading.Thread(target=run, args=(c,)) for c in ("Q 40000", "F 2000")]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    assert results == {"Q 40000": "DONE Q 40000", "F 2000": "DONE F 2000"}
    assert board.ledger.d["pos"]["x_rail"] == -40000 and board.ledger.d["pos"]["align_x"] == 2000


def test_partial_stop_leaves_other_side_running(board):
    """정밀 쪽만 멈추면(#X FBfbAud) 이송 레일은 끝까지 간다."""
    errors, done = [], []

    def transfer():
        done.append(board.line("W 60000", 5))

    def fine():
        try:
            board.line("F 60000", 5)
        except InterruptedError as exc:
            errors.append(str(exc))
    threads = [threading.Thread(target=transfer), threading.Thread(target=fine)]
    [t.start() for t in threads]
    time.sleep(0.01)
    board.stop_letters("FBfbAud")
    [t.join(5) for t in threads]
    assert done == ["DONE W 60000"]
    assert errors and "STOP F" in errors[0]
    assert "#X FBfbAud\n" in board.ser.sent


def test_buslink_maps_aligner_commands_and_rejects_transfer_axes(board):
    from tools.station import BusLink
    import mask_aligner_optimized_v4 as fine
    link = BusLink(board, fine.ArduinoLink.to_previous)
    stop = threading.Event()
    assert link.send("X -1500") and link.wait_done(stop, 5, "DONE")
    assert board.ser.sent[-1] == "#B 1500\n"
    assert link.send("T 7 1 120") and link.wait_done(stop, 5, "DONE T 7")
    assert board.ser.sent[-1] == "#A 120 1\n"
    assert not link.send("R1 1000")          # 이송 축은 정밀 앱이 움직이지 않는다
    assert "정밀 축만" in link.last_error
    link.stop()
    assert board.ser.sent[-1] == "#X FBfbAud\n"


class FakeFine:
    """정밀 앱 흉내: START 후 align_s초 뒤 DONE."""

    def __init__(self, align_s=0.4):
        self.align_s, self.auto_running, self.moving = align_s, False, False
        self.phase, self.status = "IDLE", "대기"
        self.mask_detected = self.wafer_detected = True
        self.starts, self.finishes = [], []

    def start_align(self):
        if self.auto_running:
            return
        self.starts.append(time.time())
        self.auto_running, self.phase = True, "XY"

        def finish():
            time.sleep(self.align_s)
            self.phase, self.status, self.auto_running = "DONE", "정밀 얼라인 완료", False
            self.finishes.append(time.time())
        threading.Thread(target=finish, daemon=True).start()

    def stop_align(self):
        self.auto_running = False


def test_pipeline_fetches_next_wafer_while_aligning(board, tmp_path, monkeypatch):
    from tools import station
    monkeypatch.setattr(prealign_sequence, "CALIBRATION", tmp_path / "cal.json")
    monkeypatch.setattr(station, "PIPELINE_FILE", tmp_path / "pipe.json")
    seq = prealign_sequence.Sequence(board)
    measured = []

    def stable(*a, **k):
        measured.append(time.time())
        return dict(mask=dict(angle=90), wafer=dict(angle=90), turn_deg=0.0, error=None)
    seq.vision.stable = stable
    monkeypatch.setattr(seq.stop_evt, "wait", lambda _: seq.stop_evt.is_set())
    app = FakeFine(align_s=0.6)
    pipe = station.Pipeline(seq, app, board)
    pipe.cfg.update(max_wafers=3, auto_clear=True)
    sent_log = []
    orig_line_nowait = board.line_nowait

    def spy(cmd):
        sent_log.append((time.time(), cmd))
        return orig_line_nowait(cmd)
    monkeypatch.setattr(board, "line_nowait", spy)
    pipe.start()
    pipe.thread.join(30)
    st = pipe.state
    assert st["error"] is None, st["log"]
    assert (st["fetched"], st["delivered"], st["aligned"]) == (3, 3, 3)
    assert len(app.starts) == 3
    # 겹침: 첫 정렬 시작 후, 그 정렬이 끝나기 전에 다음 웨이퍼의 가져오기(Q)가 이미 보내졌다
    first_start = app.starts[0]
    q_after = [t for t, cmd in sent_log if cmd.startswith("Q ") and t > first_start]
    assert q_after and q_after[0] < first_start + app.align_s
    # 8단계(마스크 측정)는 앞 웨이퍼의 정밀 얼라인이 끝난(마스크 Z 상승) 뒤에만
    for n, finish in enumerate(app.finishes[:2]):
        later = [t for t in measured if t > app.starts[n]]
        assert later and later[0] >= finish


def test_pipeline_waits_for_operator_to_clear_stage(board, tmp_path, monkeypatch):
    from tools import station
    monkeypatch.setattr(prealign_sequence, "CALIBRATION", tmp_path / "cal.json")
    monkeypatch.setattr(station, "PIPELINE_FILE", tmp_path / "pipe.json")
    seq = prealign_sequence.Sequence(board)
    seq.vision.stable = lambda *a, **k: dict(mask=dict(angle=90), wafer=dict(angle=90), turn_deg=0.0, error=None)
    monkeypatch.setattr(seq.stop_evt, "wait", lambda _: seq.stop_evt.is_set())
    app = FakeFine(align_s=0.1)
    pipe = station.Pipeline(seq, app, board)
    pipe.cfg.update(max_wafers=2, auto_clear=False)
    pipe.start()
    deadline = time.time() + 20
    while "비움" not in pipe.state["phase"] and time.time() < deadline:
        time.sleep(0.05)
    assert "비움" in pipe.state["phase"] and pipe.state["delivered"] == 1 and pipe.state["holding"]
    time.sleep(0.3)
    assert pipe.state["delivered"] == 1          # 확인 전에는 넘기지 않는다
    pipe.mark_cleared()
    pipe.thread.join(20)
    assert pipe.state["error"] is None and pipe.state["delivered"] == 2


def test_align_failure_is_not_taken_as_empty_box(board, tmp_path, monkeypatch):
    """정밀 얼라인 실패(키 미검출)는 '박스 빔'으로 조용히 끝내지 않고 오류로 멈춘다."""
    from tools import station
    monkeypatch.setattr(prealign_sequence, "CALIBRATION", tmp_path / "cal.json")
    monkeypatch.setattr(station, "PIPELINE_FILE", tmp_path / "pipe.json")
    seq = prealign_sequence.Sequence(board)
    seq.vision.stable = lambda *a, **k: dict(mask=dict(angle=90), wafer=dict(angle=90), turn_deg=0.0, error=None)
    monkeypatch.setattr(seq.stop_evt, "wait", lambda _: seq.stop_evt.is_set())
    app = FakeFine(align_s=30)      # 정상 완료 전에 FAULT가 난다
    orig = app.start_align

    def failing():
        orig()

        def fault():
            time.sleep(0.15)
            app.phase, app.status = "FAULT", "정렬 중단: 마스크 키 · 웨이퍼 키 5초 이상 미검출"
            app.auto_running = False
        threading.Thread(target=fault, daemon=True).start()
    app.start_align = failing
    pipe = station.Pipeline(seq, app, board)
    pipe.cfg.update(max_wafers=2, auto_clear=True)
    pipe.start()
    pipe.thread.join(20)
    assert pipe.state["error"] and "정밀 얼라인 실패" in pipe.state["error"]
    assert not any("박스가 빈" in line for line in pipe.state["log"])


@pytest.mark.parametrize("mode,expected", [("hard", (5000, 5000)), ("soft", (3500, 3500))])
def test_contact_mode_sets_mask_plate_travel(board, tmp_path, monkeypatch, mode, expected):
    """하드 컨택은 마스크 판을 끝까지(정밀 앱 기본), 소프트 컨택은 설정한 만큼만 내리고 같은 만큼 올린다."""
    from tools import station
    monkeypatch.setattr(prealign_sequence, "CALIBRATION", tmp_path / "cal.json")
    monkeypatch.setattr(station, "PIPELINE_FILE", tmp_path / "pipe.json")
    seq = prealign_sequence.Sequence(board)
    seq.vision.stable = lambda *a, **k: dict(mask=dict(angle=90), wafer=dict(angle=90), turn_deg=0.0, error=None)
    monkeypatch.setattr(seq.stop_evt, "wait", lambda _: seq.stop_evt.is_set())
    app = FakeFine(align_s=0.1)
    seen = []
    pipe = station.Pipeline(seq, app, board, z_setter=lambda d, u: seen.append((d, u)), hard_ms=(5000, 5000))
    pipe.save_cfg(dict(max_wafers=1, auto_clear=True, soft_down_ms=3500))
    pipe.start(mode)
    pipe.thread.join(20)
    assert pipe.state["error"] is None and seen == [expected]
    assert pipe.status()["mode"] == mode


def test_soft_contact_must_be_less_than_hard(board, tmp_path, monkeypatch):
    from tools import station
    monkeypatch.setattr(station, "PIPELINE_FILE", tmp_path / "pipe.json")
    pipe = station.Pipeline(prealign_sequence.Sequence(board), FakeFine(), board, hard_ms=(5000, 5000))
    with pytest.raises(ValueError):
        pipe.save_cfg(dict(soft_down_ms=6000))
    assert pipe.cfg["soft_down_ms"] == 4000      # 잘못된 값은 저장하지 않는다
