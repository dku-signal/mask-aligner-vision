"""웨이퍼 노치(세모 홈) 각도 측정. 학습·기준 그림 없이 기하 계산만 쓴다.

1. Hough 원 검출로 원판의 중심·반지름을 찾는다 (축소 영상에서 찾고 원본 좌표로 되돌림).
2. 원 안쪽 가장자리(0.93r~0.97r)를 0.5° 간격으로 한 바퀴 돌며 밝기를 잰다.
3. 원판 내부보다 확실히 어두운 가장 긴 연속 구간 = 노치. 그 가운데 각도를 돌려준다.

각도 규약: 0° = 영상 오른쪽, 반시계(영상 위쪽)로 증가, 0~360.
프리얼라인 목표: 노치를 항상 원판의 정 오른쪽(TARGET_DEG = 0°)에 둔다 (팀 결정 2026-11-18).
조건: 원판이 배경보다 밝아야 한다 (흰 원판 + 어두운 배경).
"""
from dataclasses import dataclass

import cv2
import numpy as np

STEP_DEG = 0.5
RING = (0.93, 0.95, 0.97)   # 노치 깊이보다 얕게: 홈 안쪽만 어둡게 잡힌다
DARK_RATIO = 0.6            # 원판 밝기 대비 이 비율보다 어두우면 '원판이 아님'
MAX_NOTCH_DEG = 30.0
TARGET_DEG = 0.0            # 프리얼라인 목표: 노치를 정 오른쪽에        # 이보다 넓은 어두운 구간은 노치가 아니라 원 검출 실패로 본다


@dataclass
class Notch:
    center: tuple          # (x, y) 원본 픽셀
    radius: float          # 원본 픽셀
    angle_deg: float       # 노치 방향, 0=오른쪽, 반시계
    width_deg: float       # 어두운 구간 폭
    contrast: float        # 1 - (노치 밝기 / 원판 밝기), 클수록 확실


def _find_circle(gray, work=800):
    s = work / max(gray.shape)
    small = cv2.medianBlur(cv2.resize(gray, None, fx=s, fy=s), 7)
    m = min(small.shape)
    found = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, 1.5, m, param1=120, param2=60,
                             minRadius=int(0.15 * m), maxRadius=int(0.5 * m))
    if found is None:
        return None
    x, y, r = found[0][0] / s
    return float(x), float(y), float(r)


def _ring(gray, x, y, r, frac, rad):
    h, w = gray.shape
    xs = np.clip((x + frac * r * np.cos(rad)).astype(int), 0, w - 1)
    ys = np.clip((y - frac * r * np.sin(rad)).astype(int), 0, h - 1)
    return gray[ys, xs]


def measure_notch(bgr):
    """노치를 찾으면 Notch, 원판이나 노치를 확실히 못 찾으면 None."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    circle = _find_circle(gray)
    if circle is None:
        return None
    x, y, r = circle
    gray = gray.astype(float)
    angles = np.arange(0, 360, STEP_DEG)
    rad = np.radians(angles)
    rim = np.mean([_ring(gray, x, y, r, f, rad) for f in RING], axis=0)
    disk = float(np.median(_ring(gray, x, y, r, 0.8, rad)))
    if disk <= 0:
        return None
    dark = rim < DARK_RATIO * disk
    if not dark.any() or dark.all():
        return None

    # 0°를 가로지르는 구간도 이어지도록 어둡지 않은 칸에서 시작해 한 바퀴 돈다
    n = len(dark)
    start = int(np.argmin(dark))
    best, run = (0, 0), None
    for k in range(n + 1):
        i = (start + k) % n
        if k < n and dark[i]:
            run = k if run is None else run
        elif run is not None:
            if k - run > best[1] - best[0]:
                best = (run, k)
            run = None
    width = (best[1] - best[0]) * STEP_DEG
    if width > MAX_NOTCH_DEG:
        return None
    idx = [(start + k) % n for k in range(best[0], best[1])]
    mid = (start + (best[0] + best[1] - 1) / 2) * STEP_DEG % 360
    contrast = 1 - float(np.mean(rim[idx])) / disk
    return Notch((x, y), r, round(mid, 2), width, round(contrast, 3))


def rotation_to(target_deg=TARGET_DEG, current_deg=0.0):
    """current를 target으로 돌리는 최소 회전량 (-180, 180], 반시계가 +."""
    d = (target_deg - current_deg) % 360
    return d - 360 if d > 180 else d


def draw(bgr, notch):
    """확인용 그림: 초록 원, 빨간 노치 방향, 파란 0° 기준."""
    out = bgr.copy()
    x, y = notch.center
    c = (int(x), int(y))
    t = np.radians(notch.angle_deg)
    tip = (int(x + notch.radius * np.cos(t)), int(y - notch.radius * np.sin(t)))
    lw = max(2, int(notch.radius / 100))
    cv2.circle(out, c, int(notch.radius), (0, 200, 0), lw)
    cv2.line(out, c, (int(x + notch.radius), int(y)), (255, 120, 0), lw)
    cv2.line(out, c, tip, (0, 0, 255), lw + 1)
    cv2.circle(out, tip, lw * 4, (0, 0, 255), lw)
    cv2.putText(out, f"notch {notch.angle_deg:.1f} deg", (20, 20 + 20 * lw),
                cv2.FONT_HERSHEY_SIMPLEX, lw * 0.5, (0, 0, 255), lw)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="사진 한 장에서 노치 각도 재기")
    ap.add_argument("image")
    ap.add_argument("--target", type=float, default=TARGET_DEG, help="목표 각도(°), 기본 0 = 정 오른쪽")
    ap.add_argument("--out", help="확인용 그림 저장 경로")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"이미지를 읽을 수 없음: {a.image}")
    n = measure_notch(img)
    if n is None:
        raise SystemExit("노치를 찾지 못함 (원판이 배경보다 밝은지, 화면에 원판 전체가 보이는지 확인)")
    print(f"노치 {n.angle_deg:.1f}° · 폭 {n.width_deg:.1f}° · 대비 {n.contrast:.2f} · "
          f"중심 ({n.center[0]:.0f}, {n.center[1]:.0f}) · 반지름 {n.radius:.0f}px")
    print(f"목표 {a.target:.1f}° 까지 회전 {rotation_to(a.target, n.angle_deg):+.1f}° (반시계 +)")
    if a.out:
        cv2.imwrite(a.out, draw(img, n))
