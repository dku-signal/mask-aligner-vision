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

import prealign
from sam2_keys import YOLO_WEIGHTS, pick_device

IMGSZ = 1280
CONF = 0.15
MASK_SQUARES = 8
MASK_MIN_SEEN = 6  # one full 2x2 key plus at least a row of the other
FINISH_SCORE = 0.5
# KEEP_MASK = False: the mask must be found in every frame (all 8 squares), nothing is reused.
# True: the last full mask is kept while squares are hidden (fixed plate) and saved to MASK_FILE.
KEEP_MASK = False
# The kept mask survives an app restart; it is trusted again only once enough of the squares
# seen now sit on it (the camera may have been bumped in between).
MASK_FILE = Path(__file__).resolve().parent/'out'/'two_pair_mask.json'
MASK_CONFIRM = 3
DARK_MARGIN = 60  # grey levels a held square must stay below the gap between the keys
FIT_RADIUS = 6     # px the model fit may move a cross away from its disc-based place
NEAR_SIDES = 2.5  # a cross this close to its key (in square widths) may be found without YOLO
# A square is found by shape, not by darkness: darker than 3 of its 4 sides by this fraction
# of their brightness (over the black stage the film round a square is ~0.3 brighter, over the
# white wafer ~0.7) and by this many grey levels.
SQUARE_CONTRAST = 0.12
SQUARE_STEP = 8
FIT_MAX_RMS = 0.12  # square widths: a worse two-key fit (squares merged with cross arms) is no mask
SEED_SIZE = 25     # px square side to search with before any square was measured (1080p camera 0)
SEED_PEAKS = 30    # clearest shape peaks tried as one edge of a key when YOLO finds no square
FIT_RMS_SLACK = 0.3  # px: a fresh full mask replaces the held one unless its fit is clearly worse


def fit_overlap(obs, plate, cross, guess, radius):
    """Sub-pixel top-left of the cross template on an overlapped key, by analysis by synthesis.

    Model (transmission, 1 = white): obs = plate * cross(t). plate is this key area seen with no
    cross on it, cross the cross seen clear of the squares. SSD(t) = sum((obs - plate - plate*c)^2)
    with c = cross - 1 (zero outside the template), expanded into two correlations, so every
    shift is scored at once. Unlike a centroid this keeps the gaps between the cross arms and the
    squares, which is where the position is when they touch. Returns (x, y) in obs or None.
    """
    e = (obs-plate)*plate
    c = cross-1.0
    score = -2*cv2.matchTemplate(e, c, cv2.TM_CCORR)+cv2.matchTemplate(plate*plate, c*c, cv2.TM_CCORR)
    h, w = score.shape
    gx, gy = int(round(guess[0])), int(round(guess[1]))
    x0, y0 = max(0, gx-radius), max(0, gy-radius)
    x1, y1 = min(w, gx+radius+1), min(h, gy+radius+1)
    if x1-x0 < 3 or y1-y0 < 3:
        return None
    win = score[y0:y1, x0:x1]
    j, i = np.unravel_index(int(np.argmin(win)), win.shape)
    y, x = y0+j, x0+i
    if not (0 < x < w-1 and 0 < y < h-1):
        return float(x), float(y)

    def vertex(a, b, c_):
        d = a-2*b+c_
        return 0.0 if d <= 0 else float(np.clip(0.5*(a-c_)/d, -0.5, 0.5))
    return x+vertex(score[y, x-1], score[y, x], score[y, x+1]), y+vertex(score[y-1, x], score[y, x], score[y+1, x])


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
        self.run = None          # this align run (app generation): the mask is held only within it
        self.plate = None        # (x0, y0, transmission image) of both keys with no cross on them
        self.cross_tpl = [None, None]  # (transmission patch, centre in patch) per cross, seen clear
        self.last_offset = None  # cross - key per pair from the last good frame
        self.last_crosses = None # both crosses from the last good frame
        self.baseline = None     # cross 2 - cross 1 when both were seen by YOLO (rigid wafer)
        self.disc = None         # wafer disc centre in this frame (prealign.find_disc)
        self.disc_arm = None     # cross - disc centre per cross, from the last frame they were clear
        self.disc_last = None    # (cx, cy, r) of the last disc found
        self.gray = None
        self.tracked = 0
        self.square_size = SEED_SIZE  # last measured square side: the shape search scale without YOLO
        try:
            if not KEEP_MASK:
                raise OSError
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
    def square_contrast(gray, size):
        """(ratio, step) per pixel: how much darker a square-sized patch there is than its sides.

        The reference is the second darkest of the 4 side strips, so one side may lie on the black
        stage, while a straight edge (one bright side only) is no square.
        """
        g = gray.astype(np.float32)
        s = max(3, int(round(0.8*size)))  # inside the square, clear of its blurred border
        w = max(3, int(round(0.35*size)))
        off = int(round(size/2+w/2+0.1*size))
        inner = cv2.boxFilter(g, -1, (s, s))
        sides = []
        for dx, dy, k in ((-off, 0, (w, s)), (off, 0, (w, s)), (0, -off, (s, w)), (0, off, (s, w))):
            shift = np.float32([[1, 0, -dx], [0, 1, -dy]])
            sides.append(cv2.warpAffine(cv2.boxFilter(g, -1, k), shift, g.shape[::-1], borderMode=cv2.BORDER_REPLICATE))
        sides = np.sort(sides, 0)
        side = sides[1]
        ratio = (side-inner)/np.maximum(side, 10)
        # A darker side as dark as the patch is the same dark shape going on (a cross arm), no square;
        # a square at the black stage has that side clearly darker than itself.
        ratio[(sides[0] < side-SQUARE_STEP) & (abs(sides[0]-inner) < SQUARE_STEP)] = 0
        return ratio, side-inner

    @classmethod
    def more_squares(cls, gray, boxes, reach=4.0):
        """YOLO square boxes plus the squares it missed nearby: (boxes, contrast, centre or NaN).

        A found square's centre is its sub-pixel contrast peak: next to the black stage a darkness
        centroid is pulled towards the black side. YOLO squares get NaN (blob centroid later).

        YOLO misses squares over the black stage (the film there is only ~25 grey levels
        brighter than the square), so the others are found by their shape: local contrast peaks.
        """
        size = float(np.median(np.r_[boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]]))
        x0, y0 = (boxes[:, :2].min(0)-reach*size).astype(int).clip(0)
        x1, y1 = np.minimum((boxes[:, 2:].max(0)+reach*size).astype(int), gray.shape[::-1])
        ratio, step = cls.square_contrast(gray[y0:y1, x0:x1], size)
        k = int(size) | 1
        peak = (ratio == cv2.dilate(ratio, np.ones((k, k), np.uint8))) & (ratio > SQUARE_CONTRAST) & (step > SQUARE_STEP)
        near = cv2.dilate(ratio, np.ones((k//2 | 1,)*2, np.uint8))
        centers = lambda b: (b[:, :2]+b[:, 2:])/2
        merged = list(boxes)
        contrast = [float(near[int(np.clip(y-y0, 0, y1-y0-1)), int(np.clip(x-x0, 0, x1-x0-1))]) for x, y in centers(boxes)]
        found = [np.full(2, np.nan)]*len(boxes)
        ys, xs = np.nonzero(peak)
        h, w = ratio.shape
        for i in np.argsort(-ratio[ys, xs]):
            y, x = ys[i], xs[i]
            c = np.array([x0+x, y0+y], float)
            if min(np.linalg.norm(centers(np.array(merged))-c, axis=1)) > size/2:
                if 0 < x < w-1 and 0 < y < h-1:  # parabola through the peak and its neighbours
                    for k, (lo, mid, hi) in enumerate(((ratio[y, x-1], ratio[y, x], ratio[y, x+1]),
                                                       (ratio[y-1, x], ratio[y, x], ratio[y+1, x]))):
                        den = lo-2*mid+hi
                        c[k] += 0.5*(lo-hi)/den if den < 0 else 0
                merged.append(np.r_[c-size/2, c+size/2])
                contrast.append(float(ratio[y, x]))
                found.append(c)
        return np.array(merged), np.array(contrast), np.array(found)

    @staticmethod
    def is_grid(pts, tol=0.25):
        """True if 4 points form a square (4 equal sides, 2 diagonals = side * sqrt 2)."""
        d = np.sort([np.linalg.norm(a-b) for i, a in enumerate(pts) for b in pts[i+1:]])
        side = np.median(d[:4])
        # Sides within 20% of each other: two rows of different keys (closer together) are no square.
        return bool(side > 0 and d[3] < 1.2*d[0] and np.all(abs(d[4:]-side*np.sqrt(2)) < tol*side))

    @classmethod
    def complete_groups(cls, pts, hidden_ok=None):
        """Two 2x2 groups from 6-8 square centres: (group1, group2, measured flags) or None.

        One group must be fully seen; the other may miss a corner (parallelogram rule) or a whole
        row of two (shifted by the seen group's edge that points towards it). Every frame is
        measured afresh: nothing comes from an earlier frame. hidden_ok(point) may veto a filled-in
        square (one that would have been seen there).
        """
        from itertools import combinations
        n = len(pts)
        best = None
        for idx in combinations(range(n), 4):
            a = pts[list(idx)]
            if not cls.is_grid(a):
                continue
            others = pts[[i for i in range(n) if i not in idx]]
            # Stray dark blobs may be among the others: try the other key from its best subset down.
            for k in (4, 3, 2):
                for sub in combinations(range(len(others)), k):
                    found = cls.second_group(a, others[list(sub)])
                    if found is not None and hidden_ok is not None and \
                            not all(hidden_ok(q) for q in np.vstack(found[:2])[~found[2]]):
                        found = None
                    if found is not None and (best is None or found[2].sum() > best[2].sum()):
                        best = found
                if best is not None and best[2].sum() == 4+k:
                    break
            if best is not None and best[2].all():
                return best
        return best

    @classmethod
    def second_group(cls, a, rest):
        """The other 2x2 key from 2-4 of its squares next to the full key a: (a, b, measured) or None."""
        d = np.sort([np.linalg.norm(p-q) for i, p in enumerate(a) for q in a[i+1:]])
        side = np.median(d[:4])
        if np.linalg.norm(rest.mean(0)-a.mean(0)) > 4*side:
            return None  # a key further than two key widths away is a stray blob, not the pair
        if len(rest) == 4:
            if cls.is_grid(rest) and np.linalg.norm(rest.mean(0)-a.mean(0)) > 1.5*side:
                return a, rest, np.ones(8, bool)
            return None
        if len(rest) == 3:
            # The corner whose two neighbours are one side away is the right angle.
            for k in range(3):
                i, j = [m for m in range(3) if m != k]
                if all(abs(np.linalg.norm(rest[m]-rest[k])-side) < .25*side for m in (i, j)):
                    b = np.vstack([rest, rest[i]+rest[j]-rest[k]])
                    if cls.is_grid(b):
                        return a, b, np.r_[np.ones(7, bool), False]
        if len(rest) == 2 and abs(np.linalg.norm(rest[1]-rest[0])-side) < .25*side:
            edge = rest[1]-rest[0]
            away = rest.mean(0)-a.mean(0)
            if abs(edge@away) > 0.5*np.linalg.norm(edge)*np.linalg.norm(away):
                return None  # the seen row must face key a; a column pointing at it gives no side to grow
            v = np.array([-edge[1], edge[0]])
            v *= side/np.linalg.norm(v)*np.sign(v@away)
            b = np.vstack([rest, rest+v])
            if cls.is_grid(b) and np.linalg.norm(b.mean(0)-a.mean(0)) > 1.5*side:
                return a, b, np.r_[np.ones(6, bool), False, False]
        return None

    @staticmethod
    def fit_keys(ga, gb, measured, weight):
        """Both keys refitted as two equal 2x2 squares through the measured ones: (ga, gb, rms).

        One plate carries both keys, so they share one side vector u and every square is
        c_k + a*u + b*u_perp (a, b = +-1/2): 6 unknowns, linear. Weighted by contrast, the
        clear squares fix the fit and the faint ones over the black stage only nudge it.
        """
        pts = np.vstack([ga, gb])
        c = [ga.mean(0), gb.mean(0)]
        r = ga[0]-c[0]
        u = np.array([r[0]+r[1], r[1]-r[0]])  # ga[0] = c + (u + u_perp)/2
        basis = np.linalg.inv(np.array([[u[0], -u[1]], [u[1], u[0]]]))
        rows, rhs, w = [], [], []
        for i in np.nonzero(measured)[0]:
            k = int(i >= 4)
            a, b = 0.5*np.sign(basis@(pts[i]-c[k]))
            rows += [[1-k, 0, k, 0, a, -b], [0, 1-k, 0, k, b, a]]
            rhs += list(pts[i])
            w += [weight[i]]*2
        A, y, sw = np.array(rows), np.array(rhs), np.sqrt(np.array(w))
        x = np.linalg.lstsq(A*sw[:, None], y*sw, rcond=None)[0]
        res = (A@x-y).reshape(-1, 2)
        rms = float(np.sqrt((np.linalg.norm(res, axis=1)**2*w[::2]).sum()/sum(w[::2])))
        model = []
        for i, p in enumerate(pts):
            k = int(i >= 4)
            a, b = 0.5*np.sign(basis@(p-c[k]))
            model.append(x[2*k:2*k+2]+a*x[4:]+b*np.array([-x[5], x[4]]))
        model = np.array(model)
        return model[:4], model[4:], rms

    def seed_squares(self, gray):
        """Square boxes of one 2x2 key found by shape alone, for frames where YOLO finds none.

        Over the black stage YOLO may miss every square; then an edge of two of the clearest contrast
        peaks over the whole frame, with any two peaks completing the square (a faint column over the
        black stage), is the seed for the usual search around it.
        """
        s = self.square_size
        ratio, step = self.square_contrast(gray, s)
        k = int(s) | 1
        peak = (ratio == cv2.dilate(ratio, np.ones((k, k), np.uint8))) & (ratio > SQUARE_CONTRAST) & (step > SQUARE_STEP)
        ys, xs = np.nonzero(peak)
        order = np.argsort(-ratio[ys, xs])
        pts = np.c_[xs[order], ys[order]].astype(float)
        value = ratio[ys[order], xs[order]]
        best = None
        for i, a in enumerate(pts[:SEED_PEAKS]):
            for j, b in enumerate(pts[:SEED_PEAKS]):
                e = b-a
                side = np.linalg.norm(e)
                if i == j or not 1.3*s < side < 3*s:
                    continue
                # (a, b) and (b, a) give the square on either side of the edge.
                c, d = a+[-e[1], e[0]], b+[-e[1], e[0]]
                ic, id_ = np.linalg.norm(pts-c, axis=1).argmin(), np.linalg.norm(pts-d, axis=1).argmin()
                if np.linalg.norm(pts[ic]-c) < 0.25*side and np.linalg.norm(pts[id_]-d) < 0.25*side:
                    score = value[[i, j, ic, id_]].sum()
                    if best is None or score > best[0]:
                        best = score, pts[[i, j, ic, id_]]
        if best is None:
            return np.empty((0, 4))
        return np.c_[best[1]-s/2, best[1]+s/2]

    def measure_mask(self, gray, square_boxes):
        """Two square groups from 6-8 squares (YOLO + nearby blobs) in this frame, or None."""
        seeded = len(square_boxes) < 2
        if seeded:
            square_boxes = self.seed_squares(gray)
        if len(square_boxes) < 2:
            return None
        boxes, contrast, peaks = self.more_squares(gray, square_boxes)
        if seeded:  # the seed squares are contrast peaks too, centred in their boxes
            peaks[:len(square_boxes)] = (square_boxes[:, :2]+square_boxes[:, 2:])/2
        # The combination search below is meant for 6-12 blobs: YOLO's first, then the clearest.
        keep = np.argsort(-(contrast+(np.arange(len(boxes)) < len(square_boxes))), kind='stable')[:12]
        boxes, contrast, peaks = boxes[keep], contrast[keep], peaks[keep]
        if len(boxes) < MASK_MIN_SEEN:
            return None
        centers = [blob_center(gray, b) if np.isnan(p[0]) else p for b, p in zip(boxes, peaks)]
        if any(c is None for c in centers):
            return None
        pts = np.array(centers)
        size = float(np.median(np.r_[boxes[:, 2]-boxes[:, 0], boxes[:, 3]-boxes[:, 1]]))
        h = max(1, int(size/4))
        inner = lambda q: float(gray[int(q[1])-h:int(q[1])+h+1, int(q[0])-h:int(q[0])+h+1].mean())
        x0, y0 = (pts.min(0)-2*size).astype(int).clip(0)
        x1, y1 = (pts.max(0)+2*size).astype(int)
        # A filled-in square must be hidden (under a cross, or dark over the black stage): one on a
        # bright spot would have been seen, so that grouping is wrong (keys this close have two).
        cut = (np.median([inner(q) for q in pts])+np.percentile(gray[y0:y1, x0:x1], 75))/2
        inside = lambda q: h <= q[0] < gray.shape[1]-h and h <= q[1] < gray.shape[0]-h
        found = self.complete_groups(pts, lambda q: inside(q) and inner(q) < cut)
        if found is None:
            return None  # stray dark blobs, not the 2 x (2x2) mask pattern
        ga, gb, measured = found
        weight = [contrast[np.argmin(np.linalg.norm(pts-p, axis=1))] for p in np.vstack([ga, gb])]
        ga, gb, rms = self.fit_keys(ga, gb, measured, np.clip(weight, 0.05, 1)**2)
        if rms > FIT_MAX_RMS*size:
            return None  # squares merged with cross arms: their centres are off
        self.square_size = size
        mean = np.vstack([ga, gb]).mean(0)
        axis = gb.mean(0)-ga.mean(0)
        axis /= np.linalg.norm(axis)
        if axis[0] < 0 or (abs(axis[0]) < 1e-9 and axis[1] < 0):
            axis, ga, gb = -axis, gb, ga
            measured = np.r_[measured[4:], measured[:4]] if not measured.all() else measured
        groups = ga, gb
        g1, g2 = ga.mean(0), gb.mean(0)
        keys = [np.r_[g.min(0), g.max(0)]+[-size*0.7, -size*0.7, size*0.7, size*0.7] for g in groups]
        squares = np.vstack(groups)
        order = np.argsort(~measured, kind='stable')  # measured first: the app draws the rest dimmer
        return dict(groups=(g1, g2), squares=squares[order], boxes=boxes, axis=axis, size=size, keys=keys,
                    observed=int(measured.sum()), rms=rms)

    def confirm_held(self, gray, square_boxes):
        if self.mask is None:
            return None
        seen = [c for c in (blob_center(gray, b, None) for b in square_boxes) if c is not None]
        hits = sum(min(np.linalg.norm(self.mask['squares']-c, axis=1)) < self.mask['size']/3 for c in seen)
        if hits >= MASK_CONFIRM:
            return dict(self.mask, observed=int(hits))
        # Aligned: the cross arms touch every square, so no square stands alone. Then each held
        # square must still be dark in this frame, against the bright gaps between the keys.
        s = self.mask['size']
        inner = [gray[int(y-s/4):int(y+s/4)+1, int(x-s/4):int(x+s/4)+1].mean() for x, y in self.mask['squares']]
        # Reference: the strip just outside both keys (plate or wafer, beyond the cross arms).
        k = np.array(self.mask['keys'])
        x0, y0 = (k[:, :2].min(0)-1.6*s).astype(int).clip(0)
        x1, y1 = (k[:, 2:].max(0)+1.6*s).astype(int)
        roi = gray[y0:y1, x0:x1]
        w = int(0.5*s)
        strip = np.r_[roi[:w].ravel(), roi[-w:].ravel(), roi[:, :w].ravel(), roi[:, -w:].ravel()]
        bright = np.percentile(strip, 75)
        dark = sum(v < bright-DARK_MARGIN for v in inner)
        if dark < MASK_SQUARES:
            return None
        return dict(self.mask, observed=0)

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
        # All 8 squares (also the ones filled in), slightly enlarged: they are no part of a cross.
        m = np.zeros(gray.shape, bool)
        h = 0.65*mask['size']
        for x, y in mask['squares']:
            m[int(max(0, y-h)):int(y+h)+1, int(max(0, x-h)):int(x+h)+1] = True
        self.mask_pixels = m

    def crosses(self, gray, cross_boxes):
        """The two cross centres ordered like the mask groups, or None.

        A cross that sits on squares (its own key near the end, or the other key on the way
        there) merges with them: YOLO may miss it and a centroid is pulled by the hidden parts.
        The wafer is rigid and theta is set first, so a missed cross is the seen one plus the
        cross-to-cross vector of the last frame with both; every cross on squares is then placed
        by the plate x cross model fit.
        """
        if self.mask is None:
            return None
        side = self.mask['size']
        seen = [(blob_center(gray, b, self.mask_pixels), b) for b in cross_boxes[:2]]
        seen = [(c, b) for c, b in seen if c is not None]
        if len(seen) == 2 and np.linalg.norm(seen[0][0]-seen[1][0]) < 2*side:
            seen = seen[:1]  # both boxes on the same cross
        if len(seen) == 2:
            if (seen[1][0]-seen[0][0])@self.mask['axis'] < 0:
                seen = seen[::-1]
            c = np.array([seen[0][0], seen[1][0]])
        elif len(seen) == 1 and self.baseline is not None:
            # The missed cross is on squares: of the two places it can be, take the one on squares.
            one = seen[0][0]
            options = [np.array([one, one+self.baseline]), np.array([one-self.baseline, one])]
            hidden = [options[0][1], options[1][0]]
            c = options[int(np.argmin([self.square_distance(h) for h in hidden]))]
        elif not seen and self.last_crosses is not None:
            c = self.last_crosses.copy()
        else:
            return None
        on_squares = [self.square_distance(c[i]) < 2*side for i in (0, 1)]
        if any(on_squares):
            # The square grid repeats every square pitch, so a fit on it alone can lock one pitch off
            # (seen on the rig 2026-11-18: ALIGN FINISH ~50 px off). The disc edge moves rigidly with
            # the wafer and is never covered: disc centre + the cross's arm from the disc, measured
            # while the cross was clear, places it; the model fit only refines within a few px.
            if self.disc_arm is None or self.disc_center() is None:
                return None
            for i in (0, 1):
                if not on_squares[i]:
                    continue
                c[i] = self.disc+self.disc_arm[i]
                if self.plate is not None and self.cross_tpl[i] is not None:
                    fit = self.fit_cross(gray, i, c[i], radius=FIT_RADIUS)
                    if fit is not None and np.linalg.norm(fit-c[i]) < FIT_RADIUS-0.5:
                        c[i] = fit
                self.tracked += 1
        elif min(self.square_distance(ci) for ci in c) < 6*side and self.disc_center() is not None:
            # Clear of the squares but on the way to them: keep the arms (rigid while theta holds).
            self.disc_arm = [c[i]-self.disc for i in (0, 1)]
        for i in (0, 1):
            if self.square_distance(c[i]) >= (NEAR_SIDES+2)*side:
                box = next((b for cc, b in seen if np.linalg.norm(cc-c[i]) < side), None)
                if box is not None:
                    self.keep_cross(gray, i, c[i], box)  # well clear of the squares: keep its look
        half = 1.4*side
        self.cross_keys = [tuple(int(round(v)) for v in np.r_[ci-half, ci+half]) for ci in c]
        self.last_crosses = c.copy()
        if len(seen) == 2:
            self.baseline = c[1]-c[0]
        self.last_offset = [c[i]-self.mask['groups'][i] for i in (0, 1)]
        return c

    def disc_center(self):
        """Wafer disc centre, refined from the last one (fast) or found from scratch (Hough)."""
        if self.disc is None:
            d = None
            if self.disc_last is not None:
                d = prealign._refine(self.gray, *self.disc_last)
                if d is not None and (d[3] < prealign.DISC_MIN_EDGE or d[4] < prealign.DISC_MIN_STEP
                                      or abs(d[2]-self.disc_last[2]) > 5):
                    d = None
            if d is None:
                d = prealign.find_disc(self.gray)
            self.disc_last = None if d is None else d[:3]
            self.disc = None if d is None else np.array(d[:2])
        return self.disc

    def square_distance(self, point):
        return float(np.min(np.linalg.norm(self.mask['squares']-point, axis=1)))

    def keep_cross(self, gray, i, center, box):
        x0, y0, x1, y1 = box
        p = 0.25*max(x1-x0, y1-y0)
        x0, y0 = int(max(0, x0-p)), int(max(0, y0-p))
        x1, y1 = int(min(gray.shape[1], x1+p)), int(min(gray.shape[0], y1+p))
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        white = np.percentile(patch, 90)
        if white < 120 or patch.size < 100:
            return
        self.cross_tpl[i] = (np.clip(patch/white, 0, 1.2), (center[0]-x0, center[1]-y0))

    def keep_plate(self, gray, cross_boxes):
        """Both keys with no cross on them (the wafer under them, so the plate looks as it will)."""
        k = np.array(self.mask['keys'])
        s = self.mask['size']
        x0, y0 = (k[:, :2].min(0)-3*s).astype(int).clip(0)
        x1, y1 = (k[:, 2:].max(0)+3*s).astype(int)
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        white = np.percentile(patch, 90)
        if white < 120:
            return
        plate = np.clip(patch/white, 0, 1.2)
        for bx0, by0, bx1, by1 in cross_boxes:
            # A cross inside the patch would be counted twice by the model: paint it white.
            q = 0.25*max(bx1-bx0, by1-by0)
            plate[int(max(0, by0-q-y0)):int(max(0, by1+q-y0)), int(max(0, bx0-q-x0)):int(max(0, bx1+q-x0))] = 1.0
        self.plate = (x0, y0, plate, white)

    def fit_cross(self, gray, i, guess, radius=None):
        x0, y0, plate, white = self.plate
        tpl, (cx, cy) = self.cross_tpl[i]
        h, w = plate.shape
        obs = np.clip(gray[y0:y0+h, x0:x0+w].astype(np.float32)/white, 0, 1.2)
        if obs.shape != plate.shape:
            return None
        top_left = (guess[0]-x0-cx, guess[1]-y0-cy)
        t = fit_overlap(obs, plate, tpl, top_left, int(radius or 1.5*self.mask['size']))
        if t is None:
            return None
        return np.array([t[0]+x0+cx, t[1]+y0+cy])

    def detection(self, center, size, score, quads=(), observed=0):
        n = int(math.ceil(size))
        return dict(center=(float(center[0]), float(center[1])), rect=(int(center[0]-n/2), int(center[1]-n/2), n, n),
                    area=float(size*size), score=float(score), quads=list(quads), observed=observed)

    def process(self, frame, phase, token=None, stationary=True):
        # The app's token changes on every motor move; only a camera change (its last two fields)
        # invalidates the kept mask, which stays put while the stage moves the wafer.
        camera = token[-2:] if isinstance(token, tuple) else token
        run = token[0] if isinstance(token, tuple) else token
        if camera != self.token or frame.shape != self.shape or run != self.run:
            # New camera or a new START: nothing measured before carries over.
            self.mask = self.mask_pixels = None
            self.last_offset = self.last_crosses = self.baseline = self.disc_arm = None
            self.plate = None
            self.cross_tpl = [None, None]
            self.token, self.shape, self.run = camera, frame.shape, run
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        square_boxes, cross_boxes, cross_conf = self.detect(frame)
        fresh = self.measure_mask(gray, square_boxes)
        if not KEEP_MASK:
            # A cross on the keys merges with the squares and pulls their centres (now that faint
            # squares are found too, all 8 may be), so such a frame never replaces the held mask;
            # otherwise the cleaner fit wins (the plate does not move within a run).
            clear = fresh is not None and not any(
                min(np.linalg.norm(fresh['squares']-c, axis=1)) < NEAR_SIDES*fresh['size']
                for c in (cross_boxes[:, :2]+cross_boxes[:, 2:])/2)
            if fresh is not None and (self.mask is None or fresh['observed'] == MASK_SQUARES and clear
                                      and fresh['rms'] <= self.mask.get('rms', np.inf)+FIT_RMS_SLACK):
                self.keep_mask(gray, fresh, save=False)
                mask = fresh
            else:
                # A partial fresh mask while one is held: cross arms may pass for squares, so
                # the held mask (checked against this frame below) is the safer one.
                # Crosses on the squares hide some of them. The plate does not move, so the mask
                # found earlier in this run holds, but only while MASK_CONFIRM squares seen in this
                # frame still sit on it: a moved plate or camera drops it at once.
                mask = self.confirm_held(gray, square_boxes)
                if mask is not None and fresh is not None:
                    # Squares found by shape this frame that sit on the held mask count as seen too.
                    seen = fresh['squares'][:fresh['observed']]
                    hits = sum(min(np.linalg.norm(mask['squares']-c, axis=1)) < mask['size']/3 for c in seen)
                    mask = dict(mask, observed=max(mask['observed'], int(hits)))
        else:
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
        md = self.detection((g1+g2)/2, float(np.linalg.norm(g2-g1))+2*s, 1.0, quads, mask.get('observed', len(quads)))
        md['keys'] = [tuple(int(round(v)) for v in k) for k in mask['keys']]
        ma = (math.degrees(math.atan2(*(g2-g1)[::-1])), 1.0)
        self.disc = None  # found on demand by disc_center(): only needed near the squares
        self.gray = gray
        c = self.crosses(gray, cross_boxes)
        if c is not None and mask.get('observed') == MASK_SQUARES and \
                min(self.square_distance(ci) for ci in c) > (NEAR_SIDES+2)*s:
            self.keep_plate(gray, cross_boxes[:2])
        if c is None:
            return md, None, ma, None, False
        score = float(cross_conf[:2].min()) if len(cross_conf) >= 2 else FINISH_SCORE
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
