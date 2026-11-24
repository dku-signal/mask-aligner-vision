"""Coarse orientation from two cameras and the released YOLO square/cross weights."""
import base64
import math
from collections import deque
import statistics
import time
import threading
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .key_geometry import MaskPairGeometry, SEED_SIZE, blob_center
else:
    from key_geometry import MaskPairGeometry, SEED_SIZE, blob_center

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "runs/keys_real/weights/best.pt"


def wafer_disc(frame):
    """웨이퍼 원 (x, y, r) 또는 None. 크기(화면 높이의 0.32~0.5)와 밝은 안쪽으로 가짜 원을 거른다."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    h, w = gray.shape
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w, param1=80, param2=40,
                             minRadius=int(0.32 * h), maxRadius=int(0.5 * h))
    if found is None:
        return None
    x, y, r = found[0][0]
    inside = np.zeros_like(gray)
    cv2.circle(inside, (int(x), int(y)), int(r * 0.8), 255, -1)
    if cv2.mean(gray, mask=inside)[0] < 150:
        return None
    return np.array([x, y, r], float)


def cross_side_turn(turn, side, camera_sign, want_y):
    """축 정렬 회전량 turn(°)을 돌렸을 때 십자(웨이퍼 중심 기준 side=(dx, dy))가 want_y 쪽
    (-1 = 카메라 1 화면 위)에 오도록, 아니면 180° 반대로 돌린다. 영상 각도는 camera_sign × turn만큼 바뀐다."""
    a = math.radians(camera_sign * turn)
    dy = math.sin(a) * side[0] + math.cos(a) * side[1]
    if dy * want_y >= 0:
        return turn
    return round(turn - 180 if turn > 0 else turn + 180, 2)


def key_axis(boxes, role, shape, min_span=0):
    """An individual square or cross has 90-degree symmetry. Use their centres."""
    if len(boxes) < 2:
        raise ValueError(f"{role} 마크가 {len(boxes)}개 검출됨 (최소 2개 필요)")
    if role == "웨이퍼" and len(boxes) != 2:
        raise ValueError("웨이퍼 십자가 3개 이상 검출됨. 검출 영역을 확인하세요")
    centres = (boxes[:, :2] + boxes[:, 2:4]) / 2
    centre = centres.mean(axis=0)
    _, singular, axes = np.linalg.svd(centres - centre, full_matrices=False)
    if singular[0] <= 2 * singular[1]:
        raise ValueError(f"{role} 마크 배열의 긴 방향이 불분명함")
    direction = axes[0]
    offsets = (centres - centre) @ direction
    if np.ptp(offsets) < max(min_span, min(shape[:2]) * 0.05):
        raise ValueError(f"{role} 마크 사이 거리가 너무 짧음")
    ends = centre + np.array([offsets.min(), offsets.max()])[:, None] * direction
    return ends, math.degrees(math.atan2(direction[1], direction[0])) % 180


def axis_difference(target, actual):
    """Shortest signed difference for an undirected axis in image coordinates."""
    return (target - actual + 90) % 180 - 90


class KeyPrealign:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def _wafer_boxes(self, crop, confidence):
        # Rotated YOLO views recover inclined crosses. Map detections back before
        # comparing centres; two agreeing views must identify the same pair.
        h, w = crop.shape[:2]
        candidates = []
        boxes = np.empty((0, 4))
        for angle in (0, -15, 45, -45, 30, -30, 75, -75):
            transform = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            size = np.ceil(np.abs(transform[:, :2]) @ [w, h]).astype(int)
            transform[:, 2] += (size - [w, h]) / 2
            view = crop if angle == 0 else cv2.warpAffine(crop, transform, tuple(size), borderValue=(114, 114, 114))
            raw = self.model.predict(view, classes=[1], conf=max(confidence, 0.55),
                                     imgsz=640, device="cpu", verbose=False)[0].boxes.xyxy.cpu().numpy()
            if len(raw) > 2:
                raw = self._outer_pair(raw)
            if len(raw) != 2:
                continue
            quads = np.array([[(l,t), (r,t), (r,b), (l,b)] for l,t,r,b in raw])
            quads = (quads - transform[:, 2]) @ transform[:, :2]
            boxes = np.column_stack((quads.min(axis=1), quads.max(axis=1)))
            centres = quads.mean(axis=1)
            sizes = raw[:, 2:] - raw[:, :2]
            margins = sizes.min(axis=1)[:, None] * 0.4
            if np.any(centres < margins) or np.any(centres > [w, h] - margins):
                continue
            for previous in candidates:
                if np.linalg.norm(centres[0] - previous[1]) < np.linalg.norm(centres[0] - previous[0]):
                    previous = previous[::-1]
                span = np.linalg.norm(centres[1] - centres[0])
                if np.max(np.linalg.norm(centres - previous, axis=1)) <= max(2, span * 0.045):
                    shift = (previous - centres) / 2
                    return boxes + np.tile(shift, (1, 2)), None
            candidates.append(centres)
        return boxes, "웨이퍼 십자 2개가 화면 안에서 두 번 일치 검출되어야 합니다"

    def _wafer_contrast_pair(self, crop, confidence):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        low, high = np.percentile(gray, (0.5, 99.5))
        if high - low < 6:
            return None
        contrast = np.clip((gray.astype(np.float32) - low) * (255 / (high - low)), 0, 255).astype(np.uint8)
        boxes, error = self._wafer_boxes(cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR), confidence)
        if error:
            return None
        refined = []
        for box in boxes:
            centre = blob_center(contrast, box, pad=0)
            half = (box[2:] - box[:2]) / 2
            if centre is None or np.any(abs(centre - (box[:2] + half)) > half * 0.5):
                return None
            refined.append(np.r_[centre - half, centre + half])
        return np.array(refined)

    @staticmethod
    def _outer_pair(raw):
        """세로로 붙은 두 십자 사이에 YOLO가 상자를 더 그릴 때: 한 쌍으로 볼 수 있는 거리(상자 2.5개 이내)
        안에서 가장 멀리 떨어진 두 상자 = 양 끝 십자."""
        centres = (raw[:, :2] + raw[:, 2:]) / 2
        reach = 2.5 * np.median(raw[:, 2:] - raw[:, :2])
        best, pair = 0.0, raw[:0]
        for i in range(len(raw)):
            for j in range(i + 1, len(raw)):
                d = float(np.linalg.norm(centres[i] - centres[j]))
                if best < d <= reach:
                    best, pair = d, raw[[i, j]]
        return pair

    def _detect(self, frame, camera, role, class_id, confidence, preview):
        h, w = frame.shape[:2]
        guide_span = np.linalg.norm(np.diff(camera["guide"], axis=0) * [w, h])
        # Dark mask squares near the stage need a different scale as lighting varies.
        views = [(camera["roi"], 640, None)]
        if class_id == 0 and "contrast_roi" in camera:
            views += [(camera["contrast_roi"], 320, (180, 8)),
                      (camera["contrast_roi"], 160, (170, 6)),
                      (camera["contrast_roi"], 160, (165, 7))]
        line = None
        mask_regions = {}
        for region, image_size, levels in views:
            x, y, right, bottom = np.rint(np.array(region) * [w, h, w, h]).astype(int)
            if not (0 <= x < right <= w and 0 <= y < bottom <= h):
                raise ValueError(f"{role} 검출 영역 설정 오류")
            crop = frame[y:bottom, x:right]
            if levels:
                # Remove uneven illumination on the dark stage; boxes still come exclusively from YOLO.
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
                background = cv2.GaussianBlur(gray, (0, 0), max(1, min(gray.shape) / 16))
                bias, gain = levels
                contrast = np.clip(bias + gain * (gray - background), 0, 255).astype(np.uint8)
                crop = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
            detection_error = None
            if class_id == 1:
                boxes, detection_error = self._wafer_boxes(crop, confidence)
                if detection_error:
                    refined = self._wafer_contrast_pair(crop, confidence)
                    if refined is not None:
                        boxes, detection_error = refined, None
            else:
                prediction = self.model.predict(crop, classes=[0], conf=confidence,
                                                imgsz=image_size, device="cpu", verbose=False)[0]
                boxes = prediction.boxes.xyxy.cpu().numpy()
                bounds = (x, y, right, bottom)
                if bounds not in mask_regions or len(mask_regions[bounds]) < len(boxes):
                    mask_regions[bounds] = boxes
                if len(boxes) != 8:
                    detection_error = f"마스크 사각점 {len(boxes)}/8개 검출"
            boxes = boxes + [x, y, x, y]
            result = {"count": len(boxes), "angle": None, "error": None}
            try:
                if detection_error:
                    raise ValueError(detection_error)
                line, angle = key_axis(boxes, role, frame.shape, min_span=0.6 * guide_span)
                result["angle"] = round(angle, 2)
                if class_id == 1:
                    # 각도는 0~180°(선의 기울기)라 십자가 웨이퍼의 어느 쪽에 있는지는 웨이퍼 원 중심으로 따로 본다
                    disc = wafer_disc(frame)
                    if disc is not None:
                        centre = (boxes[:, :2] + boxes[:, 2:4]).mean(axis=0) / 2
                        result["side"] = [round(float(v), 1) for v in centre - disc[:2]]
                break
            except ValueError as exc:
                result["error"] = str(exc)
        if class_id == 0 and result["error"]:
            for bounds, seeds in mask_regions.items():
                left, top, end_x, end_y = bounds
                gray = cv2.cvtColor(frame[top:end_y, left:end_x], cv2.COLOR_BGR2GRAY)
                if int(gray.max()) - int(gray.min()) < 8:
                    continue
                gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
                geometry = MaskPairGeometry(max(3, round(SEED_SIZE * w / 1920)))
                fit = geometry.measure_mask(gray, np.empty((0, 4)))
                if fit is None and len(seeds) >= 2:
                    fit = geometry.measure_mask(gray, seeds)
                if fit is None:
                    continue
                groups = np.array(fit["groups"]) + [left, top]
                delta = groups[1] - groups[0]
                if np.linalg.norm(delta) < max(0.6 * guide_span, min(h, w) * 0.05):
                    continue
                centres = fit["squares"][:fit["observed"]] + [left, top]
                boxes = np.c_[centres - fit["size"]/2, centres + fit["size"]/2]
                x, y, right, bottom = bounds
                line = groups
                result.update(count=fit["observed"], angle=round(math.degrees(math.atan2(*delta[::-1])) % 180, 2),
                              error=None, method="two-pair")
                break
        if preview:
            out = frame.copy()
            cv2.rectangle(out, (x, y), (right-1, bottom-1), (130, 130, 130), 1)
            for l, t, r, b in boxes.round().astype(int):
                cv2.rectangle(out, (l, t), (r, b), (0, 220, 230), 1)
            if line is not None:
                a, b = line.round().astype(int)
                cv2.line(out, a, b, (40, 210, 40), 2, cv2.LINE_AA)
            ok, jpg = cv2.imencode(".jpg", out)
            if ok:
                result["jpeg"] = base64.b64encode(jpg).decode("ascii")
        return result

    def _load_model(self):
        if self.model is None:
            weights = WEIGHTS if WEIGHTS.is_file() else ROOT / "runs/keys/weights/best.pt"
            if not weights.is_file():
                raise ValueError(f"YOLO 가중치 필요: {WEIGHTS}")
            from ultralytics import YOLO
            self.model = YOLO(str(weights))

    def wafer_frame(self, frame, settings):
        with self.lock:
            self._load_model()
            return self._detect(frame, settings["wafer"], "웨이퍼", 1, settings["confidence"], False)

    def stable_wafer(self, cameras, settings, stop_evt):
        angles = deque(maxlen=5)
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            if stop_evt.is_set():
                raise InterruptedError("정지됨")
            frame, = cameras.fresh_frames([settings["wafer"]["camera"]])
            angle = self.wafer_frame(frame, settings)["angle"]
            if angle is None:
                angles.clear()
                continue
            angles.append(angle)
            offsets = [axis_difference(a, angle) for a in angles]
            if len(angles) == 5 and max(offsets) - min(offsets) <= 2:
                return (angle + statistics.median(offsets)) % 180
        raise ValueError("웨이퍼 십자 2개가 같은 위치에서 안정적으로 검출되어야 합니다")

    def measure_wafer(self, cameras, settings):
        with self.lock:
            self._load_model()
            camera = settings["wafer"]
            for _ in range(3):
                frame, = cameras.fresh_frames([camera["camera"]])
                result = self._detect(frame, camera, "웨이퍼", 1, settings["confidence"], False)
                if not result["error"]:
                    break
            return result

    def measure(self, cameras, settings, preview=False):
        with self.lock:
            self._load_model()
            source = [settings["mask"], settings["wafer"]]
            indices = [s["camera"] for s in source]
            if indices[0] == indices[1]:
                raise ValueError("마스크와 웨이퍼는 서로 다른 카메라를 지정하세요")
            frames = cameras.fresh_frames(indices)
            result = {}
            for frame, camera, key, role, cls in zip(frames, source, ("mask", "wafer"), ("마스크", "웨이퍼"), (0, 1)):
                result[key] = self._detect(frame, camera, role, cls, settings["confidence"], preview)
            errors = [view["error"] for view in result.values() if view["error"]]
            result.update(turn_deg=None, error=" / ".join(errors) or None)
            if not errors:
                mapped = settings["camera_sign"] * result["wafer"]["angle"] + settings["camera_offset_deg"]
                turn = round(axis_difference(result["mask"]["angle"], mapped), 2)
                want = settings.get("wafer_cross_side_y")
                if want and result["wafer"].get("side"):
                    flipped = cross_side_turn(turn, result["wafer"]["side"], settings["camera_sign"], want)
                    result["side_flip"] = flipped != turn
                    turn = flipped
                result["turn_deg"] = turn
            return result

    def stable(self, cameras, settings, stop_evt, preview=False):
        # Keep samples close in time even if some frames miss one of the marks.
        samples = deque(maxlen=5)
        deadline = time.monotonic() + 15
        detail = "측정값이 안정되지 않았습니다"
        while time.monotonic() < deadline:
            if stop_evt.is_set():
                raise InterruptedError("정지됨")
            result = self.measure(cameras, settings, preview)
            now = time.monotonic()
            if result["error"]:
                detail = result["error"]
                continue
            samples.append((now, result))
            if len(samples) < 5 or now - samples[0][0] > 4:
                continue
            # Both cameras must be stable, including across 0°/180° and ±90°.
            if any(max(abs(axis_difference(r[key]["angle"], result[key]["angle"]))
                       for _, r in samples) > 1.5 for key in ("mask", "wafer")):
                detail = "키 방향이 움직이고 있습니다"
                continue
            errors = [axis_difference(r["turn_deg"], result["turn_deg"]) for _, r in samples]
            mid = statistics.median(errors)
            return min(samples, key=lambda pair: abs(axis_difference(pair[1]["turn_deg"], result["turn_deg"]) - mid))[1]
        raise ValueError(f"YOLO 각도 측정 중단 · {detail}")
