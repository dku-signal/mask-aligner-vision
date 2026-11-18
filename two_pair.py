"""Two key pairs in one view: mask = 8 squares (two 2x2 groups side by side), wafer = 2 crosses.

  python two_pair.py frame.jpg [...]   # print both pairs, theta and offset; save *_pairs.jpg

YOLO (runs/keys_real, imgsz 1280 so ~20 px keys are found) gives square and cross boxes;
each box is refined to the centroid of its dark blob. The squares are split into the two
groups along their long axis (projection order), each group paired with the cross on the
same side. Theta = angle(cross1 -> cross2) - angle(group1 -> group2), measured over the
whole key-pair baseline instead of one key. Centre offset = mean of the two pair offsets.

The mask plate does not move (the stage carries the wafer), so the mask is kept from the
last frame where all 8 squares were seen. That keeps the mask exact while the crosses slide
over the squares near the aligned state, when some squares merge with the cross arms.
Pixels of the kept squares are left out of the cross centroid for the same reason.

process() returns the same tuple as the other detectors, so the app needs no other change:
md/wd centre = midpoint of the two mask groups / two crosses, ma/wa = baseline angles.
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from sam2_keys import YOLO_WEIGHTS, pick_device

IMGSZ = 1280
CONF = 0.15
MASK_SQUARES = 8
FINISH_SCORE = 0.5
# The kept mask survives an app restart; it is trusted again only once enough of the squares
# seen now sit on it (the camera may have been bumped in between).
MASK_FILE = Path(__file__).resolve().parent/'out'/'two_pair_mask.json'
MASK_CONFIRM = 3


def blob_center(gray, box, exclude=None, pad=0.15):
    """Sub-pixel centroid of the dark blob in a box (Otsu inside the box, darkness weighted)."""
    x0, y0, x1, y1 = box
    w, h = x1-x0, y1-y0
    x0, y0 = int(max(0, x0-pad*w)), int(max(0, y0-pad*h))
    x1, y1 = int(min(gray.shape[1], x1+pad*w)), int(min(gray.shape[0], y1+pad*h))
    roi = gray[y0:y1, x0:x1].astype(np.float64)
    if roi.size < 9:
        return None
    t, _ = cv2.threshold(roi.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    weight = np.clip(t-roi, 0, None)
    if exclude is not None:
        weight[exclude[y0:y1, x0:x1]] = 0
    total = weight.sum()
    if total <= 0:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return np.array([(xs*weight).sum()/total, (ys*weight).sum()/total])


class TwoPairKeys:
    def __init__(self, weights=YOLO_WEIGHTS, device=None, center_tolerance=6.0, **_):
        from ultralytics import YOLO
        self.yolo = YOLO(str(weights))
        self.device = device or pick_device()
        self.center_tolerance = center_tolerance
        self.mask = None        # kept mask: dict(groups=(c1, c2), squares=[...], boxes=[...], side)
        self.mask_pixels = None
        self.token = None
        self.shape = None
        # Attributes the app reads from every detector (pattern cache size, tracking reset).
        self.bank_bytes = 0
        self.previous_mask = self.previous_wafer = None
        self.saved = None
        try:
            d = json.loads(MASK_FILE.read_text())
            self.saved = dict(groups=tuple(np.array(g) for g in d['groups']), squares=np.array(d['squares']),
                              boxes=np.array(d['boxes']), axis=np.array(d['axis']), size=d['size'],
                              keys=[np.array(k) for k in d['keys']])
        except (OSError, ValueError, KeyError):
            pass

    def detect(self, frame):
        r = self.yolo.predict(frame, imgsz=IMGSZ, conf=CONF, device=self.device, verbose=False)[0]
        cls = r.boxes.cls.cpu().numpy().astype(int)
        box = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        order = np.argsort(-conf)
        return box[order][cls[order] == 0], box[order][cls[order] == 1], conf[order][cls[order] == 1]

    @staticmethod
    def more_squares(gray, boxes, reach=2.5):
        """Square boxes YOLO missed: dark square blobs of the same size near the YOLO squares."""
        size = float(np.median(np.r_[boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]]))
        x0, y0 = (boxes[:, :2].min(0)-reach*size).astype(int).clip(0)
        x1, y1 = (boxes[:, 2:].max(0)+reach*size).astype(int)
        roi = gray[y0:y1, x0:x1]
        # Local threshold: the squares sit on bright and dark backgrounds alike.
        block = int(4*size) | 1
        dark = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15)
        found = []
        for c in cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if not (0.6*size <= w <= 1.5*size and 0.6*size <= h <= 1.5*size and area >= 0.6*w*h):
                continue
            found.append([x0+x, y0+y, x0+x+w, y0+y+h])
        found = np.array(found, float).reshape(-1, 4)
        centers = lambda b: (b[:, :2]+b[:, 2:])/2
        merged = list(boxes)
        for b, c in zip(found, centers(found)):
            if min(np.linalg.norm(centers(np.array(merged))-c, axis=1)) > size/2:
                merged.append(b)
        return np.array(merged)

    @staticmethod
    def is_grid(pts, tol=0.25):
        """True if 4 points form a square (4 equal sides, 2 diagonals = side * sqrt 2)."""
        d = np.sort([np.linalg.norm(a-b) for i, a in enumerate(pts) for b in pts[i+1:]])
        side = np.median(d[:4])
        return bool(side > 0 and np.all(abs(d[:4]-side) < tol*side) and np.all(abs(d[4:]-side*np.sqrt(2)) < tol*side))

    def measure_mask(self, gray, square_boxes):
        """Two square groups from MASK_SQUARES squares (YOLO + nearby blobs), or None."""
        if len(square_boxes) < 2:
            return None
        boxes = self.more_squares(gray, square_boxes)
        if len(boxes) < MASK_SQUARES:
            return None
        centers = [blob_center(gray, b) for b in boxes]
        if any(c is None for c in centers):
            return None
        pts = np.array(centers)
        if len(pts) > MASK_SQUARES:
            # Keep the MASK_SQUARES closest to their median: stray dark blobs lie further out.
            keep = np.argsort(np.linalg.norm(pts-np.median(pts, 0), axis=1))[:MASK_SQUARES]
            pts, boxes = pts[keep], boxes[keep]
        mean = pts.mean(0)
        _, _, vt = np.linalg.svd(pts-mean)
        axis = vt[0]
        if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
            axis = -axis
        order = np.argsort((pts-mean)@axis)
        half = MASK_SQUARES//2
        groups = pts[order[:half]], pts[order[half:]]
        if not all(self.is_grid(g) for g in groups):
            return None  # stray dark blobs, not the 2 x (2x2) mask pattern
        g1, g2 = groups[0].mean(0), groups[1].mean(0)
        keys = [np.r_[g.min(0), g.max(0)] for g in groups]  # one box per 2x2 key (square centres)
        size = float(np.median(np.r_[boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]]))
        keys = [k+[-size*0.7, -size*0.7, size*0.7, size*0.7] for k in keys]
        return dict(groups=(g1, g2), squares=pts, boxes=boxes, axis=axis, size=size, keys=keys)

    def confirm_saved(self, gray, square_boxes):
        """Adopt the mask saved by an earlier run if the squares seen now sit on it."""
        if self.saved is None or not len(square_boxes):
            return
        seen = [c for c in (blob_center(gray, b) for b in square_boxes) if c is not None]
        hits = sum(min(np.linalg.norm(self.saved['squares']-c, axis=1)) < self.saved['size']/2 for c in seen)
        if hits >= MASK_CONFIRM:
            self.keep_mask(gray, self.saved, save=False)

    def keep_mask(self, gray, mask, save=True):
        self.mask = mask
        if save:
            try:
                MASK_FILE.parent.mkdir(exist_ok=True)
                MASK_FILE.write_text(json.dumps({k: (np.asarray(v).tolist() if k != 'groups' else [g.tolist() for g in v])
                                                 for k, v in mask.items()}))
            except OSError:
                pass
        m = np.zeros(gray.shape, bool)
        for x0, y0, x1, y1 in mask['boxes']:
            p = 0.15*(x1-x0)
            m[int(max(0, y0-p)):int(y1+p)+1, int(max(0, x0-p)):int(x1+p)+1] = True
        self.mask_pixels = m

    def crosses(self, gray, cross_boxes):
        """The two cross centres ordered like the mask groups, or None."""
        if len(cross_boxes) < 2 or self.mask is None:
            return None
        c = [blob_center(gray, b, self.mask_pixels) for b in cross_boxes[:2]]
        if any(v is None for v in c):
            return None
        c = np.array(c)
        boxes = cross_boxes[:2]
        if (c[1]-c[0])@self.mask['axis'] < 0:
            c, boxes = c[::-1], boxes[::-1]
        self.cross_keys = [tuple(int(round(v)) for v in b) for b in boxes]
        return c

    def detection(self, center, size, score, quads=(), observed=0):
        n = int(math.ceil(size))
        return dict(center=(float(center[0]), float(center[1])), rect=(int(center[0]-n/2), int(center[1]-n/2), n, n),
                    area=float(size*size), score=float(score), quads=list(quads), observed=observed)

    def process(self, frame, phase, token=None, stationary=True):
        # The app's token changes on every motor move; only a camera change (its last two fields)
        # invalidates the kept mask, which stays put while the stage moves the wafer.
        camera = token[-2:] if isinstance(token, tuple) else token
        if camera != self.token or frame.shape != self.shape:
            self.mask = self.mask_pixels = None
            self.token, self.shape = camera, frame.shape
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        square_boxes, cross_boxes, cross_conf = self.detect(frame)
        fresh = self.measure_mask(gray, square_boxes)
        if fresh is not None and stationary:
            self.keep_mask(gray, fresh)
        elif self.mask is None:
            self.confirm_saved(gray, square_boxes)
        mask = self.mask or fresh
        if mask is None:
            return None, None, None, None, False
        g1, g2 = mask['groups']
        s = mask['size']
        quads = [np.array([[x-s/2, y-s/2], [x+s/2, y-s/2], [x+s/2, y+s/2], [x-s/2, y+s/2]]) for x, y in mask['squares']]
        md = self.detection((g1+g2)/2, float(np.linalg.norm(g2-g1))+2*s, 1.0, quads, len(quads))
        md['keys'] = [tuple(int(round(v)) for v in k) for k in mask['keys']]
        ma = (math.degrees(math.atan2(*(g2-g1)[::-1])), 1.0)
        c = self.crosses(gray, cross_boxes)
        if c is None:
            return md, None, ma, None, False
        score = float(cross_conf[:2].min())
        wd = self.detection(c.mean(0), float(np.linalg.norm(c[1]-c[0]))+2*s, score)
        wd['pairs'] = [(g1, c[0]), (g2, c[1])]
        wd['keys'] = self.cross_keys
        wa = (math.degrees(math.atan2(*(c[1]-c[0])[::-1])), score)
        offset = c.mean(0)-(g1+g2)/2
        finish = bool(phase in ('XY', 'DONE') and max(abs(offset)) < self.center_tolerance and score >= FINISH_SCORE)
        return md, wd, ma, wa, finish


def draw(frame, md, wd, ma, wa):
    out = frame.copy()
    if md:
        for q in md['quads']:
            cv2.polylines(out, [q.astype(np.int32)], True, (0, 200, 0), 2)
    if wd:
        for g, c in wd['pairs']:
            cv2.drawMarker(out, tuple(int(v) for v in c), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.line(out, tuple(int(v) for v in g), tuple(int(v) for v in c), (255, 0, 255), 2)
    if ma and wa:
        d = np.subtract(wd['center'], md['center'])
        cv2.putText(out, f'theta {(wa[0]-ma[0]+90) % 180-90:+.2f} deg  offset ({d[0]:+.1f}, {d[1]:+.1f}) px',
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return out


if __name__ == '__main__':
    det = TwoPairKeys()
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        md, wd, ma, wa, _ = det.process(img, 'THETA', token=path)
        if md is None:
            print(f'{path}: mask not found (need {MASK_SQUARES} squares)')
        elif wd is None:
            print(f'{path}: mask ok, crosses not found')
        else:
            d = np.subtract(wd['center'], md['center'])
            print(f'{path}: theta {(wa[0]-ma[0]+90) % 180-90:+.2f} deg, offset ({d[0]:+.2f}, {d[1]:+.2f}) px, '
                  f'mask groups {np.round(md["center"], 1)}, cross score {wa[1]:.2f}')
        cv2.imwrite(str(Path(path).with_name(Path(path).stem+'_pairs.jpg')), draw(img, md, wd, ma, wa))
