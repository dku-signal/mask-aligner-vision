"""SAM2 (hiera-small) segmentation front end for CornerKeys.

SAM2 proposes regions; subpixel.PreciseKeys does the measuring.
- mask squares: SAM2 masks -> quad filter -> CornerKeys.mask_geometry grouping
- wafer cross: unchanged masked template matching (SAM2 has no text prompt)
mode='sam2' uses SAM2 squares only, mode='hybrid' merges them with OpenCV's.
YoloSam2Keys / OpenCvSam2Keys: YOLO or rule-based boxes -> SAM2 box prompts.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from mask_aligner_optimized_v4 import CornerKeys
from subpixel import PreciseKeys

MODEL_ID = 'facebook/sam2.1-hiera-small'
_RUNS = Path(__file__).resolve().parent/'runs'
# Classes 0 square, 1 cross. Prefer the model fine-tuned on real rig frames (2026-11-05,
# tools/autolabel.py); fall back to the synthetic one (tools/make_dataset.py, 2026-09-28).
YOLO_WEIGHTS_REAL = _RUNS/'keys_real'/'weights'/'best.pt'
YOLO_WEIGHTS_SYNTH = _RUNS/'keys'/'weights'/'best.pt'
YOLO_WEIGHTS = YOLO_WEIGHTS_REAL if YOLO_WEIGHTS_REAL.is_file() else YOLO_WEIGHTS_SYNTH


def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class Sam2Keys(PreciseKeys):
    def __init__(self, mode='sam2', points_per_side=16, device=None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.device = device or pick_device()
        self.sam_squares = []
        self.sam_ms = 0.0
        self.sam_mask_count = 0
        if mode in ('yolo-sam2', 'opencv-sam2'):
            return
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        self.generator = SAM2AutomaticMaskGenerator.from_pretrained(
            MODEL_ID, device=self.device, points_per_side=points_per_side,
            points_per_batch=64, pred_iou_thresh=.70, stability_score_thresh=.85)

    def quad(self, gray, c):
        # Same acceptance rules as CornerKeys.squares, applied to one SAM2 contour.
        area = cv2.contourArea(c)
        max_side = min(gray.shape)*.24
        if area < 64 or area > max_side**2:
            return None
        peri = cv2.arcLength(c, True)
        p = cv2.approxPolyDP(c, .035*peri, True)
        if len(p) != 4 or not cv2.isContourConvex(p):
            return None
        pts = p[:, 0, :].astype(np.float32)
        edges = np.roll(pts, -1, axis=0)-pts
        lengths = np.linalg.norm(edges, axis=1)
        side = float(np.mean(lengths))
        if lengths.min() < 7 or lengths.max()/lengths.min() > 1.45:
            return None
        if np.max(abs(np.sum(edges*np.roll(edges, -1, axis=0), axis=1) /
                      (lengths*np.roll(lengths, -1)))) > .40:
            return None
        if area/side**2 < .73:
            return None
        x, y, w, h = cv2.boundingRect(p)
        if x < 2 or y < 2 or x+w >= gray.shape[1]-2 or y+h >= gray.shape[0]-2:
            return None
        pad = max(3, int(side*.13))
        x0, y0 = max(0, x-pad), max(0, y-pad)
        roi = gray[y0:min(gray.shape[0], y+h+pad), x0:min(gray.shape[1], x+w+pad)]
        mask = np.zeros(roi.shape, np.uint8)
        cv2.fillConvexPoly(mask, (pts-[x0, y0]).astype(np.int32), 1)
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
        ring = (cv2.dilate(mask, np.ones((2*pad+1, 2*pad+1), np.uint8)) > 0) & (mask == 0)
        if not inner.any() or not ring.any():
            return None
        ink = float(np.median(roi[inner]))
        outside = float(np.percentile(roi[ring], 65))
        if outside-ink < 12:
            return None
        return dict(center=pts.mean(axis=0), side=side, quad=pts, ink=ink, contrast=outside-ink)

    def segment(self, frame, gray):
        started = time.monotonic()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.inference_mode():
            masks = self.generator.generate(rgb)
        found = self.squares_from_masks(gray, [m['segmentation'] for m in masks])
        self.sam_ms = (time.monotonic()-started)*1000
        self.sam_mask_count = len(masks)
        return found

    def squares_from_masks(self, gray, masks):
        found = []
        for seg in masks:
            contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                item = self.quad(gray, c)
                if item is not None:
                    found.append(item)
        return found

    def squares(self, gray):
        result = list(self.sam_squares)
        if self.mode == 'hybrid':
            for t in super().squares(gray):
                if all(np.linalg.norm(s['center']-t['center']) >= .3*t['side'] for s in result):
                    result.append(t)
        return sorted(result, key=lambda t: t['contrast'], reverse=True)[:48]

    def process(self, frame, phase, token=None, stationary=True):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        self.sam_squares = self.segment(frame, gray)
        return super().process(frame, phase, token, stationary)


class BoxSam2Keys(Sam2Keys):
    """Box proposals -> SAM2 box prompts. Subclasses decide where boxes come from."""

    def __init__(self, mode, box_pad=3, device=None, **kwargs):
        super().__init__(mode=mode, device=device, **kwargs)
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.box_pad = box_pad
        self.predictor = SAM2ImagePredictor.from_pretrained(MODEL_ID, device=self.device)
        self.boxes = np.zeros((0, 4))

    def refine(self, frame, gray, boxes):
        """One SAM2 mask per box, index-aligned with boxes."""
        started = time.monotonic()
        # Pad so SAM2 sees the square's edge against its surround.
        boxes = boxes+[-self.box_pad, -self.box_pad, self.box_pad, self.box_pad]
        with torch.inference_mode():
            self.predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            masks, _, _ = self.predictor.predict(box=boxes, multimask_output=False)
        masks = np.asarray(masks).reshape(-1, *gray.shape) > 0
        self.sam_ms = (time.monotonic()-started)*1000
        self.sam_mask_count = len(masks)
        return masks


class YoloSam2Keys(BoxSam2Keys):
    """YOLO finds both keys; SAM2 segments the mask squares only when the key is (re)acquired.

    Acquire (no tracked mask yet): YOLO square boxes -> SAM2 box prompts -> quads.
    Track: YOLO square boxes -> OpenCV squares inside those boxes (no SAM2: fast).
    Wafer: the template search sees only the best YOLO cross box (class 1).
    Every centre is still measured on the original image by PreciseKeys (edge fits).
    """

    BOX_PAD = .3  # search margin around a YOLO box, as a fraction of its size

    def __init__(self, weights=YOLO_WEIGHTS, conf=.25, device=None, **kwargs):
        super().__init__(mode='yolo-sam2', device=device, **kwargs)
        from ultralytics import YOLO
        self.yolo = YOLO(str(weights))
        self.conf = conf
        self.yolo_ms = 0.0
        self.cross_box = None
        self.used_sam = False

    def inside(self, point, boxes):
        x, y = point
        for x0, y0, x1, y1 in boxes:
            pad = self.BOX_PAD*max(x1-x0, y1-y0)
            if x0-pad <= x <= x1+pad and y0-pad <= y <= y1+pad:
                return True
        return False

    def segment(self, frame, gray):
        started = time.monotonic()
        result = self.yolo.predict(frame, imgsz=640, conf=self.conf, device=self.device, verbose=False)[0]
        cls = result.boxes.cls.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        self.boxes = boxes[cls == 0]
        crosses = np.flatnonzero(cls == 1)
        self.cross_box = boxes[crosses[np.argmax(conf[crosses])]] if len(crosses) else None
        self.yolo_ms = (time.monotonic()-started)*1000
        self.sam_ms, self.sam_mask_count, self.used_sam = 0.0, 0, False
        if not len(self.boxes):
            return []
        if self.previous_mask is None:
            found = self.squares_from_masks(gray, self.refine(frame, gray, self.boxes))
            self.used_sam = True
            if len(found) >= 3:
                return found
        # Tracking (or SAM2 found too few): rule-based squares, kept only inside YOLO boxes.
        return [t for t in CornerKeys.squares(self, gray) if self.inside(t['center'], self.boxes)]

    def wafer(self, gray, mask):
        if self.cross_box is None:
            return super().wafer(gray, mask)
        x0, y0, x1, y1 = self.cross_box
        pad = self.BOX_PAD*max(x1-x0, y1-y0)+8
        h, w = gray.shape
        xa, ya = max(0, int(x0-pad)), max(0, int(y0-pad))
        xb, yb = min(w, int(x1+pad)+1), min(h, int(y1+pad)+1)
        # Flat fill outside the box has no contrast, so the masked correlation rejects it there.
        only = np.full_like(gray, int(np.median(gray)))
        only[ya:yb, xa:xb] = gray[ya:yb, xa:xb]
        return super().wafer(only, mask)


class OpenCvSam2Keys(BoxSam2Keys):
    """Rule-based square candidates -> SAM2 re-segments each one.

    SAM2 cannot recover a square OpenCV missed; it only redraws the ones found.
    A candidate whose SAM2 mask fails the quad test keeps its OpenCV outline.
    """

    def __init__(self, device=None, **kwargs):
        super().__init__(mode='opencv-sam2', device=device, **kwargs)
        self.opencv_ms = 0.0

    def segment(self, frame, gray):
        started = time.monotonic()
        candidates = CornerKeys.squares(self, gray)
        self.opencv_ms = (time.monotonic()-started)*1000
        self.boxes = np.array([[*t['quad'].min(axis=0), *t['quad'].max(axis=0)] for t in candidates]).reshape(-1, 4)
        if not candidates:
            self.sam_ms, self.sam_mask_count = 0.0, 0
            return []
        found = []
        for t, seg in zip(candidates, self.refine(frame, gray, self.boxes)):
            refined = [q for q in self.squares_from_masks(gray, [seg])
                       if np.linalg.norm(q['center']-t['center']) < .3*t['side']]
            found.append(max(refined, key=lambda q: q['contrast']) if refined else t)
        return found


def make_detector(name, **kwargs):
    if name == 'opencv-v4':
        return CornerKeys(**kwargs)
    if name == 'opencv':
        return PreciseKeys(**kwargs)
    if name == 'yolo-sam2':
        return YoloSam2Keys(**kwargs)
    if name == 'opencv-sam2':
        return OpenCvSam2Keys(**kwargs)
    return Sam2Keys(mode=name, **kwargs)
