import time

import pytest

import mask_aligner_optimized_v4 as fine
from tools.fine_stage import FineStage


def move(stage, letter, steps):
    stage.begin(letter, steps)
    stage.ack(letter, steps)


def test_reference_survives_restart_and_partial_stop(tmp_path):
    path = tmp_path / 'fine.json'
    stage = FineStage('board', path)
    stage.teach()
    move(stage, 'F', 32000)
    stage.begin('b', 1000)
    with pytest.raises(ValueError):
        FineStage('board', path).moves()
    stage.ack('b', 325, partial=True)
    assert FineStage('board', path).moves() == [('B', 32000), ('f', 325)]
    with pytest.raises(ValueError):
        FineStage('other board', path).moves()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    stage = FineStage('board', tmp_path / 'stage.json')
    stage.teach()
    move(stage, 'F', 32000)
    move(stage, 'f', 120)
    commands = []

    class Link:
        fine_stage = stage
        last_error = ''

        def connected(self):
            return True

        def send(self, command):
            commands.append((time.monotonic(), command))
            return True

        def wait_done(self, *args):
            return True

        def stop(self):
            pass

    monkeypatch.setattr(fine, 'ArduinoLink', lambda *args: Link())
    app = fine.MaskAlignerApp(detector='opencv-v4')
    monkeypatch.setattr(fine, 'ALIGN_Z_DOWN_MS', 0)
    monkeypatch.setattr(app, 'missing_keys', lambda *args: [])
    return app, stage, commands


def test_finish_returns_relative_to_start_after_lift_and_ten_seconds(rig):
    app, stage, commands = rig
    assert fine.FINE_RETURN_DELAY_S == 10
    move(stage, 'F', 1000)
    app.start_align()
    move(stage, 'B', 2931)
    move(stage, 'b', 66938)
    started = time.monotonic()
    app.finish_align('ALIGN FINISH')
    app.workers[-1].join(13)
    xy = [(t, c) for t, c in commands if c[0] in 'XY']
    assert xy and xy[0][0] - started >= 10
    assert commands[0][1] == 'Z 1 1 5000'
    assert [c for _, c in xy] == ['X -20000', 'X -9069', 'Y 20000', 'Y 20000', 'Y 20000', 'Y 6938']
    assert stage.position() == (1000, 120) and not app.auto_running
    assert app.status == '정밀 X·Y 자동 복귀 완료'
    assert app.return_fine_stage(app.generation) == '정밀 X·Y 오른쪽 위 복귀 완료'
    assert stage.position() == (0, 0)


def test_stop_cancels_pending_return(rig, monkeypatch):
    app, stage, commands = rig
    monkeypatch.setattr(fine, 'ALIGN_Z_UP_MS', 0)
    app.start_align()
    app.finish_align('ALIGN FINISH')
    app.stop_align()
    app.workers[-1].join(1)
    assert not app.workers[-1].is_alive() and not commands
    assert stage.moves() == [('B', 32000), ('b', 120)]


def test_xy_checks_each_axis_before_larger_corrections(rig, monkeypatch):
    app, _, _ = rig
    monkeypatch.setattr(fine, 'STEPS_PER_PIXEL_X', 100)
    monkeypatch.setattr(fine, 'STEPS_PER_PIXEL_Y', 80)
    assert app.checked_xy_steps((None, None, 200, 200, -8000, -8000)) == [-1200, -960]
    with pytest.raises(ValueError, match='Y축'):
        app.checked_xy_steps((None, None, 188, 200, -8000, -8000))


def test_theta_return_reverses_confirmed_pulses_before_xy(rig, monkeypatch):
    app, _, commands = rig
    monkeypatch.setattr(fine, 'FINE_RETURN_DELAY_S', 0)
    monkeypatch.setattr(fine, 'THETA_SETTLE_SEC', 0)
    app.start_align()
    assert app.pulse_fine_theta(1, 100, app.generation)
    assert app.pulse_fine_theta(-1, 80, app.generation)
    app.finish_align('ALIGN FINISH')
    app.workers[-1].join(3)
    assert [cmd for _, cmd in commands] == [
        'T 1 1 100', 'T 2 -1 80', 'Z 1 1 5000',
        'T 3 1 80', 'T 4 -1 100', 'X -20000', 'X -12000',
    ]
    assert app.align_theta_pulses == [100, -80]
    assert not app.auto_running
    app.start_align()
    assert app.align_theta_pulses == []


def test_theta_return_stops_on_missing_ack(rig, monkeypatch):
    app, _, commands = rig
    monkeypatch.setattr(fine, 'FINE_RETURN_DELAY_S', 0)
    monkeypatch.setattr(fine, 'THETA_SETTLE_SEC', 0)
    monkeypatch.setattr(fine, 'ALIGN_Z_UP_MS', 0)
    app.start_align()
    assert app.pulse_fine_theta(1, 100, app.generation)
    assert app.pulse_fine_theta(1, 120, app.generation)
    monkeypatch.setattr(app.arduino, 'wait_done', lambda *args: False)
    app.finish_align('ALIGN FINISH')
    app.workers[-1].join(2)
    assert [cmd for _, cmd in commands] == ['T 1 1 100', 'T 2 1 120', 'T 3 -1 120']
    assert not app.auto_running and app.phase == 'FAULT'
