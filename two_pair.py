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
from tools.key_geometry import MaskPairGeometry, blob_center

IMGSZ = 1280
CONF = 0.15
MASK_SQUARES = 8
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
# Rim fraction the wafer disc needs while tracking: on the fine stage part of its rim lies against
# bright aluminium (0.58-0.63 found, rig 2026-11-20); the rim step (>100 grey levels) still holds.
TRACK_DISC_MIN_EDGE = 0.45
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




class TwoPairKeys(MaskPairGeometry):
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
        super().__init__()
        try:
            if not KEEP_MASK:
                raise OSError
            d = json.loads(MASK_FILE.read_text(encoding='utf-8'))
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
                                                 for k, v in mask.items()}), encoding='utf-8')
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
                if d is not None and (d[3] < TRACK_DISC_MIN_EDGE or d[4] < prealign.DISC_MIN_STEP
                                      or abs(d[2]-self.disc_last[2]) > 5):
                    d = None
            if d is None:
                d = prealign.find_disc(self.gray, TRACK_DISC_MIN_EDGE)
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
