import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import key_prealign, motor_panel, prealign_sequence


def boxes(points):
    return np.array([[x-2, y-2, x+2, y+2] for x, y in points], float)


def test_pair_axis_and_camera_mapping():
    vertical = boxes([(30, 10), (40, 10), (30, 23), (40, 23), (30, 36), (40, 36)])
    horizontal = boxes([(50, 42), (76, 44)])
    _, mask = key_prealign.key_axis(vertical, "마스크", (240, 320))
    _, wafer = key_prealign.key_axis(horizontal, "웨이퍼", (240, 320))
    assert 84 < key_prealign.axis_difference(mask, wafer) < 87
    assert key_prealign.axis_difference(175, 4) == -9
    assert key_prealign.axis_difference(4, 175) == 9
    assert key_prealign.axis_difference(0, 47) == -47
    with pytest.raises(ValueError):
        key_prealign.key_axis(boxes([(10,10),(40,10),(10,40),(40,40)]), "마스크", (240,320))


def test_preview_uses_yolo_classes_and_restores_crop_coordinates():
    model = key_prealign.KeyPrealign()
    calls = []
    def predict(image, classes, **options):
        calls.append((image.shape[:2], classes))
        if classes == [0]:
            coords = boxes([(x,y) for x in (12,18) for y in (4,14,24,34)])
        else:
            points = np.array([[60,50],[88,50]])
            if len(calls) == 3:
                import cv2
                transform = cv2.getRotationMatrix2D((160,120), -15, 1)
                transform[:,2] += (np.array(image.shape[1::-1]) - [320,240]) / 2
                points = points @ transform[:,:2].T + transform[:,2]
            coords = boxes(points)
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: coords))))]
    model.model = SimpleNamespace(predict=predict)
    cfg = prealign_sequence.load()["prealign"]
    frame = np.zeros((240, 320, 3), np.uint8)
    result = model.measure(SimpleNamespace(fresh_frames=lambda ids: [frame, frame]), cfg, True)
    assert [role for _,role in calls] == [[0],[1],[1]]
    assert calls[0][0] == (58,80)
    assert abs(result["turn_deg"]) == 90 and result["wafer"]["jpeg"] and result["mask"]["jpeg"]


def test_dark_mask_geometry_with_no_yolo_seeds():
    detector = key_prealign.KeyPrealign()
    empty = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.empty((0, 4))))
    detector.model = SimpleNamespace(predict=lambda *args, **kwargs: [SimpleNamespace(boxes=SimpleNamespace(xyxy=empty))])
    camera = dict(roi=[0.2, 0.2, 0.8, 0.8], guide=[[0.5, 0.4], [0.5, 0.5]])
    frame = np.full((480, 640, 3), 40, np.uint8)
    for top in (180, 226):
        for offset in (0, 18):
            for left in (300, 318):
                frame[top+offset:top+offset+8, left:left+8] = 21
    found = detector._detect(frame, camera, "마스크", 0, 0.45, False)
    assert found["method"] == "two-pair" and found["count"] == 8
    assert abs(found["angle"] - 90) < 1
    frame[220:] = 40
    assert detector._detect(frame, camera, "마스크", 0, 0.45, False)["angle"] is None


def test_dark_wafer_requires_two_current_crosses(monkeypatch):
    detector = key_prealign.KeyPrealign()
    frame = np.full((100, 150, 3), 40, np.uint8)
    for centre in (45, 105):
        frame[42:59, centre-2:centre+3] = 25
        frame[48:53, centre-8:centre+9] = 23
        frame[49:52, centre-1:centre+2] = 12
    pair = np.array([[36, 41, 54, 59], [96, 41, 114, 59]], float)
    def detect_pair(image, confidence):
        return (pair, None) if image.max() > 100 else (np.empty((0, 4)), "미검출")
    monkeypatch.setattr(detector, "_wafer_boxes", detect_pair)
    camera = dict(roi=[0, 0, 1, 1], guide=[[0.3, 0.5], [0.7, 0.5]])
    assert detector._detect(frame, camera, "웨이퍼", 1, 0.45, False)["angle"] == 0
    frame[:, 90:] = 40
    assert detector._detect(frame, camera, "웨이퍼", 1, 0.45, False)["angle"] is None


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(motor_panel.__file__).parent))
    monkeypatch.setattr(motor_panel, "LEDGER_FILE", tmp_path / "ledger.json")
    stage = motor_panel.FineStage("mock", tmp_path / "fine.json")
    stage.teach()
    monkeypatch.setattr(motor_panel.FineStage, "for_port", lambda port: stage)
    monkeypatch.setattr(prealign_sequence, "CALIBRATION", tmp_path / "cal.json")
    monkeypatch.setattr(motor_panel.MockSerial, "SPEED", 500)
    import quarter_turn
    timing = tmp_path / "timing.json"
    monkeypatch.setattr(quarter_turn, "TIMING", timing)
    timing.write_text(json.dumps(dict(wafer_increasing_direction=-1, directions={
        "1": dict(ms=574, samples=[89,91]), "-1": dict(ms=543, samples=[90,90])})), encoding="utf-8")
    board = motor_panel.Board()
    seq = prealign_sequence.Sequence(board)
    board.connect("mock")
    monkeypatch.setattr(seq.stop_evt, "wait", lambda _: seq.stop_evt.is_set())
    yield seq, board
    board.disconnect()


def measured(error):
    return dict(mask=dict(angle=90), wafer=dict(angle=90-error), turn_deg=error, error=None)


def test_shared_timing_works_without_local_calibration(rig):
    import quarter_turn
    seq, board = rig
    quarter_turn.TIMING.unlink()
    observations = iter([45, 0])
    seq.vision.stable = lambda *args: measured(next(observations))
    seq.start("theta")
    seq.thread.join(3)
    assert seq.state["error"] is None and board.ser.sent == ["#t 275 -1\n"]


def test_trial_needs_held_wafer_before_retracting(rig):
    seq, board = rig
    with pytest.raises(ValueError, match="전자석"):
        seq.start("8-9")
    assert not board.ser.sent


@pytest.mark.parametrize("angle", [0, 90])
def test_transfer_cycle(rig, angle):
    seq, board = rig
    # A previously saved box and rail position must not shorten Q or W.
    board.ledger.snapshot("box_x", 0)
    board.ledger.snapshot("pos", dict(board.ledger.d["pos"], x_rail=272000))
    recipe = {
        0: ["K 6500", "O 5000", "M 0"], 1: ["Q 250000"], 2: ["E 90000"],
        3: ["L 6500"], 4: ["M 1"], 5: ["P 1100"], 7: ["K 6500"],
        8: ["R 90000"], 9: ["W 250000", "F 32000"], 9.5: ["P 5000"],
        10: ["E 90000"], 11: ["L 2700"], 12: ["O 3000"], 13: ["M 0"],
        14: ["R 90000", "K 6500"],
    }
    assert [(s["no"], s["cmds"]) for s in seq.plan()] == list(recipe.items())
    prealign_sequence.save_calibration(dict(quarter_turn_ms=560, servo_direction=-1, camera_sign=1, camera_offset_deg=0))
    seen = []
    def measure(*args, **kwargs):
        if seen:
            return measured(0)
        seen.extend(board.ser.sent)
        return dict(mask=dict(angle=90), wafer=dict(angle=90-angle), turn_deg=angle, error=None)
    seq.vision.stable = measure
    seq.start()
    seq.thread.join(5)
    assert seq.state["error"] is None and seq.state["done"][-1] == 14
    assert "#M 1\n" in seen and seen[-1] == "#R 90000\n"
    sent = board.ser.sent
    assert board.fine_stage.moves() == [("B", 32000)]
    if angle:
        recipe[9] = ["W 5700", "t 550 -1", "W 244300", "F 32000"]
    assert sent == [f"#{cmd}\n" for step in recipe.values() for cmd in step]


def test_missing_calibration_or_missing_key_blocks_transfer(rig):
    seq, board = rig
    with pytest.raises(ValueError):
        seq.start(9)
    assert board.ser.sent == []
    prealign_sequence.save_calibration(dict(quarter_turn_ms=600, servo_direction=1, camera_sign=1, camera_offset_deg=0))
    board.magnet = True
    seq.vision.stable = lambda *a, **k: dict(error="마스크 미검출")
    seq.start(8)
    seq.thread.join(2)
    assert seq.state["error"] == "마스크 미검출"
    assert seq.state["done"] == [8] and not seq.prepared
    with pytest.raises(ValueError, match="8단계"):
        seq.start(9)
    assert not any(cmd.startswith(("#W", "#t")) for cmd in board.ser.sent)


@pytest.mark.parametrize("angle, segment", [(0, []),
    (90, ["#W 5700\n", "#t 550 -1\n", "#W 244300\n"]),
    (45, ["#W 2950\n", "#t 275 -1\n", "#W 247050\n"]),
    (-45, ["#W 2950\n", "#t 275 1\n", "#W 247050\n"])])
@pytest.mark.parametrize("separate", [False, True])
def test_step8_measures_once_at_rest_then_step9_consumes(rig, angle, segment, separate):
    seq, board = rig
    board.magnet = True
    board.ledger.snapshot("box_x", 0)
    prealign_sequence.save_calibration(dict(quarter_turn_ms=600, servo_direction=-1, camera_sign=1, camera_offset_deg=0))
    measuring, release = threading.Event(), threading.Event()
    calls = []
    def measure(*args, **kwargs):
        if calls:
            return measured(0)
        calls.append(True)
        measuring.set()
        assert release.wait(2)
        return dict(mask=dict(angle=90), wafer=dict(angle=90-angle), turn_deg=angle, error=None)
    seq.vision.stable = measure
    seq.start(8 if separate else "8-9")
    assert measuring.wait(2)
    assert seq.state["done"] == [8] and seq.state["current"] == "각도 측정"
    assert board.ledger.d["pos"]["y_rail"] == -90000 and board.ser.sent == ["#R 90000\n"]
    release.set()
    if separate:
        seq.thread.join(2)
        assert seq.prepared and board.ser.sent == ["#R 90000\n"]
        seq.start(9)
    seq.thread.join(2)
    assert seq.state["error"] is None and calls == [True]
    expected = segment or ["#W 250000\n"]
    assert board.ser.sent == ["#R 90000\n"] + expected + ["#F 32000\n"]
    assert not seq.prepared
    with pytest.raises(ValueError):
        seq.start(9)


def test_cannot_use_angle_after_other_motor_movement(rig):
    seq, board = rig
    board.magnet = True
    prealign_sequence.save_calibration(dict(quarter_turn_ms=600, servo_direction=1, camera_sign=1, camera_offset_deg=0))
    seq.vision.stable = lambda *a, **k: dict(mask=dict(angle=90), wafer=dict(angle=0), turn_deg=90, error=None)
    seq.start(8)
    seq.thread.join(2)
    # 정밀 스테이지(F)는 이송 팔 위 웨이퍼를 건드리지 않으므로 측정이 유지된다 (연속 운전의 전제)
    board.line("F 100", 2)
    assert seq.prepared
    # 이송 축이 움직이면 측정 위치가 바뀌므로 무효
    board.line("Q 100", 2)
    assert not seq.prepared
    with pytest.raises(ValueError, match="8단계"):
        seq.start(9)


def test_remeasure_at_step8_without_retracting_again(rig):
    seq, board = rig
    board.magnet = True
    cfg = dict(quarter_turn_ms=600, servo_direction=1, camera_sign=1, camera_offset_deg=0)
    prealign_sequence.save_calibration(cfg)
    seq.vision.stable = lambda *a, **k: dict(error="마스크 미검출")
    seq.start(8)
    seq.thread.join(2)
    assert not seq.prepared
    seq.vision.stable = lambda *a, **k: measured(0)
    seq.measure_angle(seq.cfg["prealign"], preview=True)
    assert seq.prepared
    seq.start(9)
    seq.thread.join(2)
    assert seq.state["done"] == [9] and board.ser.sent.count("#R 90000\n") == 1


def test_stale_pair_is_rejected(monkeypatch):
    cams = motor_panel.Cameras()
    cams.views = {i: {"capture": SimpleNamespace(sample=(1, object()))} for i in (0, 1)}
    times = iter([5, 5.1, 7])
    monkeypatch.setattr(motor_panel.time, "monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="새 프레임"):
        cams.fresh_frames([0, 1])


@pytest.mark.parametrize("responses", [("DONE W 10000", "DONE t 500"), ("DONE t 500", "DONE W 10000")])
def test_overlapping_completions_in_either_order(rig, responses):
    _, board = rig
    def send(data):
        if data.startswith(b"#t"):
            for r in responses:
                board.ser.out.put(r)   # 실제 읽기 스레드를 거쳐 축별 우편함으로
    board.ser.write = send
    board.transfer_turn("W", 10000, 500, 1, threading.Event())
    assert not board.mailbox.get("W") and not board.mailbox.get("t")

@pytest.mark.parametrize("start,direction,ms", [
    (90, -1, 550), (-90, 1, 550), (45, -1, 275), (-45, 1, 275),
    (44.7, -1, 273), (-22.5, 1, 138), (2.0, -1, 12),
])
def test_theta_only_sends_one_proportional_pulse(rig, start, direction, ms):
    seq, board = rig
    observations = iter([start, 0.0])
    seq.vision.stable = lambda *args: measured(next(observations))
    seq.start("theta")
    seq.thread.join(3)
    assert seq.state["error"] is None and seq.state["theta_feedback"]["verified"]
    assert seq.state["theta_feedback"]["planned_deg"] == start
    assert seq.state["theta_feedback"]["measured_deg"] == abs(start)
    assert board.ser.sent == [f"#t {ms} {direction}\n"]
    assert not board.magnet


def test_step9_finishes_transfer_without_post_turn_image(rig):
    seq, board = rig
    board.magnet = True
    def view(*args, **kwargs):
        if any(cmd.startswith("#t") for cmd in board.ser.sent):
            raise ValueError("새 영상에서 키 미검출")
        return measured(45)
    seq.vision.stable = view
    seq.start("8-9")
    seq.thread.join(3)
    assert seq.state["error"] is None and seq.state["done"] == [8, 9]
    assert not seq.state["theta_feedback"]["verified"]
    assert board.ser.sent == ["#R 90000\n", "#W 2950\n", "#t 275 -1\n", "#W 247050\n", "#F 32000\n"]


def test_stop_after_turn_blocks_remaining_transfer(rig, monkeypatch):
    seq, board = rig
    board.magnet = True
    seq.vision.stable = lambda *args, **kwargs: measured(45)
    transfer_turn = board.transfer_turn
    def finish_then_stop(*args):
        transfer_turn(*args)
        seq.stop_evt.set()
    monkeypatch.setattr(board, "transfer_turn", finish_then_stop)
    seq.start("8-9")
    seq.thread.join(3)
    assert seq.state["error"] == "정지됨"
    assert board.ser.sent == ["#R 90000\n", "#W 2950\n", "#t 275 -1\n", "#X QWERJHKLOPtM\n"]


def test_stop_while_measuring_prevents_pulse(rig):
    seq, board = rig
    measuring, release = threading.Event(), threading.Event()
    def view(*args):
        measuring.set()
        assert release.wait(2)
        return measured(32)
    seq.vision.stable = view
    seq.start("theta")
    assert measuring.wait(2)
    seq.stop()
    release.set()
    seq.thread.join(2)
    assert seq.state["error"] and not any(cmd.startswith("#") and not cmd.startswith("#X ") for cmd in board.ser.sent)

def test_out_of_tolerance_pulse_stops_without_retry(rig):
    seq, board = rig
    pairs = iter([measured(45), measured(15)])
    seq.vision.stable = lambda *args: next(pairs)
    seq.start("theta")
    seq.thread.join(2)
    assert "검증 실패" in seq.state["error"] and board.ser.sent == ["#t 275 -1\n", "#X QWERJHKLOPtM\n"]


@pytest.mark.parametrize("error", [-1.5, 0, 1.5])
def test_parallel_keys_skip_d8(rig, error):
    seq, board = rig
    seq.vision.stable = lambda *args: measured(error)
    seq.start("theta")
    seq.thread.join(2)
    assert seq.state["theta_feedback"]["verified"]
    assert seq.state["theta_feedback"]["planned_deg"] == 0 and not board.ser.sent


def test_reversed_camera_sign_and_axis_wrap(rig):
    seq, board = rig
    prealign_sequence.save_calibration(dict(camera_sign=-1, camera_offset_deg=0))
    # Mask at 150°. Raw wafer goes from 165° through 0° to 30° (+45°).
    observations = iter([(-45, 165), (0, 30)])
    def view(*args):
        error, wafer = next(observations)
        return dict(mask=dict(angle=150), wafer=dict(angle=wafer), turn_deg=error, error=None)
    seq.vision.stable = view
    seq.start("theta")
    seq.thread.join(2)
    assert seq.state["theta_feedback"]["verified"]
    assert seq.state["theta_feedback"]["measured_deg"] == 45
    assert board.ser.sent == ["#t 275 -1\n"]


def test_timing_requires_repeated_real_travel_in_both_directions(rig, monkeypatch):
    import quarter_turn
    seq, board = rig
    cfg = seq.cfg["prealign"]
    monkeypatch.setattr(quarter_turn, "_stroke", lambda b,c,v,s,e,ms,d: (ms-30)*90/(500 if d==1 else 600))
    result = quarter_turn.calibrate(board, None, None, cfg, seq.stop_evt, lambda msg: None)
    for d, expected in [(1,530),(-1,630)]:
        record = result["directions"][d]
        assert abs(record["ms"]-expected)<=2
        assert len(record["samples"])==2
    assert quarter_turn.read_timing(cfg)["directions"]["-1"]["ms"] == 630
