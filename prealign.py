"""Measure a wafer notch and plan the loader's coarse alignment."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class TransferPlan:
    x_max_steps: int
    y_max_steps: int
    target_notch_deg: float
    theta_ms_per_degree: float
    notch_camera: Optional[int]
    notch_roi: Optional[list]
    handoff: list
    precision_x_axis: str
    precision_to_transfer_x_ratio: Optional[float]


@dataclass
class TransferXRun:
    plan: TransferPlan
    transfer_x_steps: int
    precision_x_steps: int = 0


def angle_difference(target, current):
    """Shortest signed rotation, positive clockwise in image coordinates."""
    return (target - current + 180.0) % 360.0 - 180.0


def notch_angle(frame, roi=None):
    """Return the angle of one rim notch on a fully visible, near-circular wafer."""
    if roi is not None:
        x, y, w, h = roi
        if x + w > frame.shape[1] or y + h > frame.shape[0]:
            raise ValueError('PREALIGN.notch_roi is outside the camera frame')
        frame = frame[y:y + h, x:x + w]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    angles = []
    for binary in (mask, cv2.bitwise_not(mask)):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if min(w, h) < 70 or min(w, h) / max(w, h) < .9:
                continue
            if x <= 1 or y <= 1 or x + w >= gray.shape[1] - 1 or y + h >= gray.shape[0] - 1:
                continue
            angle = rim_notch(contour)
            if angle is not None:
                angles.append(angle)
    return angles[0] if len(angles) == 1 else None


def rim_notch(contour):
    # Fit the convex hull, which spans the notch instead of following it inward.
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(float)
    if len(hull) < 12:
        return None
    origin = hull.mean(axis=0)
    xy = hull - origin
    cx, cy, offset = np.linalg.lstsq(
        np.column_stack((2 * xy, np.ones(len(xy)))), (xy ** 2).sum(axis=1), rcond=None
    )[0]
    radius = math.sqrt(offset + cx * cx + cy * cy)
    center = origin + [cx, cy]
    if not .85 <= cv2.contourArea(contour) / (math.pi * radius ** 2) <= 1.03:
        return None
    if np.percentile(np.abs(np.linalg.norm(hull - center, axis=1) - radius), 90) > .025 * radius:
        return None

    rim = contour.reshape(-1, 2) - center
    depth = radius - np.linalg.norm(rim, axis=1)
    inset = depth > max(2.5, .025 * radius)
    if inset.all() or not inset.any():
        return None
    # Start on the circular rim so a notch across the contour seam stays one run.
    start = int(np.flatnonzero(~inset)[0])
    indices = np.roll(np.arange(len(rim)), -start)
    runs = np.split(indices, np.flatnonzero(np.diff(inset[indices])) + 1)
    pockets = []
    for run in runs:
        if not inset[run[0]] or len(run) < 4 or depth[run].max() < .04 * radius:
            continue
        unit = rim[run] / np.linalg.norm(rim[run], axis=1)[:, None]
        x, y = np.average(unit, axis=0, weights=depth[run])
        angle = math.degrees(math.atan2(y, x))
        offsets = angle_difference(np.degrees(np.arctan2(unit[:, 1], unit[:, 0])), angle)
        if np.ptp(offsets) > 35 or depth[run].max() > .25 * radius:
            return None
        pockets.append(angle)
    return pockets[0] if len(pockets) == 1 else None


def transfer_plan(settings):
    """Validate the rig-specific settings before any movement is sent."""
    if not isinstance(settings, dict):
        raise ValueError('Set PREALIGN in rig_config.json')
    x_steps = settings.get('x_steps')
    y_max = settings.get('y_max_steps')
    for name, steps in (('x_steps', x_steps), ('y_max_steps', y_max)):
        if type(steps) is not int or not 0 < abs(steps) <= 400000:
            raise ValueError(f'Set PREALIGN.{name} to the signed, calibrated step count')
    target = settings.get('target_notch_deg')
    scale = settings.get('theta_ms_per_degree')
    if type(target) not in (int, float) or not -180 <= target <= 180:
        raise ValueError('Set PREALIGN.target_notch_deg within -180..180')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale == 0:
        raise ValueError('Set PREALIGN.theta_ms_per_degree to a signed calibration')
    camera = settings.get('notch_camera')
    if camera is not None and (type(camera) is not int or camera < 0):
        raise ValueError('PREALIGN.notch_camera must be a nonnegative camera index')
    roi = settings.get('notch_roi')
    if roi is not None and (
        not isinstance(roi, list) or len(roi) != 4 or any(type(n) is not int for n in roi)
        or min(roi[:2]) < 0 or min(roi[2:]) < 70
    ):
        raise ValueError('PREALIGN.notch_roi must be [x, y, width, height] in pixels')
    axis = settings.get('precision_x_axis', 'X')
    if axis not in ('X', 'Y'):
        raise ValueError('PREALIGN.precision_x_axis must be the Arduino axis X or Y')
    ratio = settings.get('precision_to_transfer_x_ratio')
    if ratio is not None and (
        type(ratio) not in (int, float) or ratio == 0
        or (type(ratio) is float and not math.isfinite(ratio))
    ):
        raise ValueError('PREALIGN.precision_to_transfer_x_ratio must be a signed, nonzero step ratio')
    handoff = settings.get('handoff', [])
    if not isinstance(handoff, list):
        raise ValueError('PREALIGN.handoff must be a list of commands')
    commands = []
    for line in handoff:
        if not isinstance(line, str):
            raise ValueError('PREALIGN.handoff commands must be strings')
        if re.fullmatch(r'MAG [12] [01]', line):
            commands.append((line, f'DONE {line}', 2.0, 0.3))
            continue
        z_move = re.fullmatch(r'Z [123] [+-]?1 ([0-9]{1,5})', line)
        if z_move and 0 < int(z_move[1]) <= 15000:
            commands.append((line, 'DONE Z', int(z_move[1]) / 1000 + 3, 0.3))
            continue
        match = re.fullmatch(r'(R[12]|FY) ([+-]?[0-9]{1,6})', line)
        if match and 0 < abs(int(match[2])) <= 400000:
            timeout = abs(int(match[2])) / (125 if match[1] == 'FY' else 4000) + 5
            commands.append((line, 'DONE', timeout, 0.3))
            continue
        raise ValueError('PREALIGN.handoff supports MAG, timed Z, R1, R2 and FY')
    return TransferPlan(x_steps, y_max, target, scale, camera, roi, commands, axis, ratio)


def within_x_travel(steps, maximum):
    return (0 < steps <= maximum) if maximum > 0 else (maximum <= steps < 0)


def next_x_steps(plan, transfer_steps, precision_steps):
    """Apply the residual to the actual transfer target used on the last wafer."""
    ratio = plan.precision_to_transfer_x_ratio
    if ratio is None:
        return plan.x_max_steps
    try:
        target = transfer_steps + round(precision_steps * ratio)
    except (OverflowError, ValueError):
        raise ValueError('X correction exceeds the calibrated transfer range') from None
    if not within_x_travel(target, plan.x_max_steps):
        raise ValueError('X correction exceeds the calibrated transfer range')
    return target


def load_transfer_x(path, plan):
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return plan.x_max_steps
    except (ValueError, OSError) as exc:
        raise ValueError(f'Cannot read saved X calibration; run from the right end: {exc}') from exc
    if not isinstance(data, dict):
        raise ValueError('Saved X calibration is invalid; run from the right end')
    if (data.get('x_max_steps') != plan.x_max_steps or
            data.get('precision_x_axis') != plan.precision_x_axis):
        return plan.x_max_steps
    transfer_steps = data.get('transfer_x_steps')
    precision_steps = data.get('precision_x_steps')
    if (type(transfer_steps) is not int or type(precision_steps) is not int
            or not within_x_travel(transfer_steps, plan.x_max_steps)):
        raise ValueError('Saved X calibration is invalid; run from the right end')
    return next_x_steps(plan, transfer_steps, precision_steps)


def save_transfer_x(path, run):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', dir=path.parent, prefix=f'.{path.name}.',
                                delete=False, encoding='utf-8') as stream:
            temporary = Path(stream.name)
            json.dump({'x_max_steps': run.plan.x_max_steps,
                       'precision_x_axis': run.plan.precision_x_axis,
                       'transfer_x_steps': run.transfer_x_steps,
                       'precision_x_steps': run.precision_x_steps}, stream, indent=2)
            stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def turn_duration(observed, target, ms_per_degree):
    error = angle_difference(target, observed)
    if abs(error) <= 3.0:
        return 0
    duration = int(round(error * ms_per_degree))
    if not 20 <= abs(duration) <= 2000:
        raise ValueError('Pre-align theta correction is outside the calibrated 20..2000 ms range')
    return duration
