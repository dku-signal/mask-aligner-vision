"""Measure one continuous D8 quarter-turn in each direction using the wafer camera."""
import json
import math
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

TIMING = Path(__file__).resolve().parents[1] / "out" / "prealign_d8_timing.json"
CAL_TOLERANCE = 3.0
VERIFY_TOLERANCE = 5.0
MAX_MS = 1500


def quarter_turn_ms(settings):
    ms = settings.get("quarter_turn_ms")
    if type(ms) is not int or not 20 <= ms <= MAX_MS:
        raise ValueError(f"공통 90° 구동 시간은 20~{MAX_MS}ms 정수로 설정하세요")
    return ms


def read_timing(settings):
    try:
        data = json.loads(TIMING.read_text())
        if data["wafer_increasing_direction"] != settings["wafer_increasing_direction"]:
            raise ValueError
        for direction in ("1", "-1"):
            record = data["directions"][direction]
            ms = record["ms"]
            samples = record["samples"]
            if len(samples) != 2 or any(not math.isfinite(a) or abs(a - 90) > CAL_TOLERANCE for a in samples):
                raise ValueError
            if type(ms) is not int or not 20 <= ms <= MAX_MS:
                raise ValueError
        return data
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None


def _stopped(event):
    if event.is_set():
        raise InterruptedError("정지됨")


def _stroke(board, cameras, vision, cfg, stop, ms, direction):
    from key_prealign import axis_difference
    before = vision.stable_wafer(cameras, cfg, stop)
    # Capture during the uninterrupted firmware-timed pulse. Use these frames
    # after DONE to unwrap the undirected axis through (and beyond) 90 degrees.
    index = cfg["wafer"]["camera"]
    frames = []
    with board.lock:
        _stopped(stop)
        board.clear_replies()
        board.line_nowait(f"t {ms} {direction}")
    started = time.monotonic()
    while time.monotonic() - started < ms / 1000:
        _stopped(stop)
        frame, = cameras.fresh_frames([index])
        frames.append((time.monotonic() - started, frame))
        stop.wait(0.035)
    board.wait_done("t", ms / 1000 + 5, stop, ms)
    if stop.wait(0.6):
        raise InterruptedError("정지됨")
    after = vision.stable_wafer(cameras, cfg, stop)
    track = [(0.0, before)]
    for stamp, frame in frames:
        _stopped(stop)
        angle = vision.wafer_frame(frame, cfg)["angle"]
        if angle is not None:
            track.append((stamp, angle))
    track.append((ms / 1000 + 0.10, after))
    sign = direction * cfg["wafer_increasing_direction"]
    travel = 0.0
    for (t0, a0), (t1, a1) in zip(track, track[1:]):
        delta = sign * axis_difference(a1, a0)
        if t1 - t0 > 0.38 or abs(delta) > 65 or delta < -6:
            raise ValueError("회전 중 웨이퍼 각도를 연속 추적할 수 없습니다. 두 십자의 시야와 D8 방향을 확인하세요")
        travel += delta
    if not 4 <= travel <= 150:
        raise ValueError(f"시간 측정 범위를 벗어난 회전량 {travel:.1f}°")
    return round(travel, 2)


def calibrate(board, cameras, vision, cfg, stop, report):
    trials = {d: {"ms": 220, "samples": []} for d in (1, -1)}
    history = []
    for _ in range(12):
        # Alternate to avoid accumulating cable twist. Each direction has its own response.
        for d, trial in trials.items():
            ms = trial["ms"]
            report(f"D8 {d:+d} · {ms}ms 연속 구동 측정")
            degrees = _stroke(board, cameras, vision, cfg, stop, ms, d)
            history.append(dict(direction=d, ms=ms, degrees=degrees))
            report(f"D8 {d:+d} · {ms}ms → YOLO 실측 {degrees:.2f}°")
            error = 90 - degrees
            if abs(error) <= CAL_TOLERANCE:
                trial["samples"] = (trial["samples"] + [degrees])[-2:]
            else:
                trial["samples"] = []
                estimate = ms + error * (ms - 30) / degrees
                trial["ms"] = round(min(MAX_MS, ms * 1.65, max(40, ms * 0.6, estimate)))
        if all(len(t["samples"]) == 2 for t in trials.values()):
            result = dict(wafer_increasing_direction=cfg["wafer_increasing_direction"],
                          directions=trials, history=history)
            _stopped(stop)
            TIMING.parent.mkdir(exist_ok=True)
            with NamedTemporaryFile("w", dir=TIMING.parent, delete=False) as f:
                json.dump(result, f, indent=2)
            Path(f.name).replace(TIMING)
            return result
    raise ValueError("양방향 모두 90° ±3°를 같은 시간으로 두 번 재현하지 못했습니다")
