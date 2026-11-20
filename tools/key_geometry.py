"""Single-frame key geometry shared by the fine and prealignment cameras."""
import cv2
import numpy as np

MASK_MIN_SEEN = 6  # one full 2x2 key plus at least a row of the other
# A square is found by shape, not by darkness: darker than 3 of its 4 sides by this fraction
# of their brightness (over the black stage the film round a square is ~0.3 brighter, over the
# white wafer ~0.7) and by this many grey levels.
SQUARE_CONTRAST = 0.12
SQUARE_STEP = 8
FIT_MAX_RMS = 0.12  # square widths: a worse two-key fit (squares merged with cross arms) is no mask
SEED_SIZE = 25     # px square side to search with before any square was measured (1080p camera 0)
SEED_PEAKS = 300   # clearest shape peaks searched for the key pattern when YOLO finds no square


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


class MaskPairGeometry:
    def __init__(self, square_size=SEED_SIZE):
        self.square_size = square_size

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
        """Square boxes of the mask found by shape alone, for frames where YOLO finds no square.

        Over the black stage YOLO may miss every square, and those squares are fainter than much
        of the clutter, so the whole pattern decides: every 4 contrast peaks forming a square are a
        key candidate, and two such keys of one size side by side (the mask) win over any single one.
        """
        s = self.square_size
        ratio, step = self.square_contrast(gray, s)
        k = int(s) | 1
        peak = (ratio == cv2.dilate(ratio, np.ones((k, k), np.uint8))) & (ratio > SQUARE_CONTRAST) & (step > SQUARE_STEP)
        ys, xs = np.nonzero(peak)
        order = np.argsort(-ratio[ys, xs])[:SEED_PEAKS]
        pts = np.c_[xs[order], ys[order]].astype(float)
        value = ratio[ys[order], xs[order]]
        if len(pts) < 4:
            return np.empty((0, 4))
        dist = np.linalg.norm(pts[:, None]-pts[None], axis=2)
        i, j = np.nonzero((dist > 1.3*s) & (dist < 3*s))  # both orders: the square on either side
        e = pts[j]-pts[i]
        side = dist[i, j]
        perp = np.c_[-e[:, 1], e[:, 0]]
        nc = np.linalg.norm((pts[i]+perp)[:, None]-pts[None], axis=2)
        nd = np.linalg.norm((pts[j]+perp)[:, None]-pts[None], axis=2)
        ic, id_ = nc.argmin(1), nd.argmin(1)
        ok = (nc.min(1) < 0.25*side) & (nd.min(1) < 0.25*side)
        quads = {}
        for a, b, c, d, w in zip(i[ok], j[ok], ic[ok], id_[ok], side[ok]):
            members = frozenset((a, b, c, d))
            if len(members) == 4 and self.is_grid(pts[list(members)]):
                quads[members] = w
        if not quads:
            return np.empty((0, 4))
        keys = list(quads)
        centre = np.array([pts[list(q)].mean(0) for q in keys])
        sides = np.array([quads[q] for q in keys])
        score = np.array([value[list(q)].sum() for q in keys])
        best, best_score, paired = keys[int(score.argmax())], score.max(), False
        for m, q in enumerate(keys):
            for n in range(m+1, len(keys)):
                r = keys[n]
                gap = np.linalg.norm(centre[n]-centre[m])
                if q & r or not 0.85 < sides[n]/sides[m] < 1.18 or not 1.5*sides[m] < gap < 3*sides[m]:
                    continue
                if not paired or score[m]+score[n] > best_score:  # any pair beats a single key
                    best, best_score, paired = q | r, score[m]+score[n], True
        c = pts[list(best)]
        return np.c_[c-s/2, c+s/2]

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
