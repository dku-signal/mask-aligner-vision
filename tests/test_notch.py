from pathlib import Path

import cv2
import numpy as np
import pytest

from notch import measure_notch, rotation_to

PHOTO = Path(__file__).parent / "fixtures" / "wafer_notch_photo.jpg"


def disc(angle_deg, size=(960, 1280), center=(640, 480), r=360, notch_px=40, noise=6, seed=0):
    """어두운 배경 위 흰 원판, angle_deg 방향에 V자 노치."""
    h, w = size
    img = np.full((h, w), 30, np.uint8)
    cv2.circle(img, center, r, 235, -1)
    t = np.radians(angle_deg)
    u = np.array([np.cos(t), -np.sin(t)])          # 영상 좌표에서 노치 방향
    v = np.array([-u[1], u[0]])
    c = np.array(center, float)
    tip = c + (r - notch_px) * u
    edge = c + (r + 5) * u
    tri = np.array([tip, edge + v * notch_px * 0.7, edge - v * notch_px * 0.7]).astype(np.int32)
    cv2.fillPoly(img, [tri], 30)
    rng = np.random.default_rng(seed)
    img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def ang_err(a, b):
    return abs((a - b + 180) % 360 - 180)


@pytest.mark.parametrize("angle", [0, 1, 45, 90, 154.5, 180, 270, 359])
def test_finds_notch_angle(angle):
    n = measure_notch(disc(angle))
    assert n is not None
    assert ang_err(n.angle_deg, angle) <= 1.5
    assert abs(n.radius - 360) < 15
    assert n.contrast > 0.4


def test_off_center_disc():
    n = measure_notch(disc(30, center=(500, 520), r=300))
    assert n is not None and ang_err(n.angle_deg, 30) <= 1.5


def test_no_notch_returns_none():
    img = np.full((960, 1280), 30, np.uint8)
    cv2.circle(img, (640, 480), 360, 235, -1)
    assert measure_notch(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) is None


def test_no_disc_returns_none():
    assert measure_notch(np.full((480, 640, 3), 40, np.uint8)) is None


def test_rotation_to_takes_short_way():
    assert rotation_to(0, 154.5) == pytest.approx(-154.5)
    assert rotation_to(0, 200) == pytest.approx(160)
    assert rotation_to(0, 359) == pytest.approx(1)
    assert rotation_to(0, 0) == 0
    assert rotation_to(90, 270) == pytest.approx(180)


@pytest.mark.skipif(not PHOTO.exists(), reason="실사진 없음")
def test_real_photo():
    # 2026-11-18 팀이 보낸 사진: 노치는 원판 왼쪽 위
    n = measure_notch(cv2.imread(str(PHOTO)))
    assert n is not None
    assert ang_err(n.angle_deg, 154.5) <= 3
