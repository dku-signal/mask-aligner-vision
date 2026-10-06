"""Subpixel measurement on top of CornerKeys' integer-pixel detection.

CornerKeys finds the keys; the measured centres are quantised:
- mask squares: approxPolyDP corners are integer contour points
- wafer cross: matchTemplate peak is an integer location (1 deg / 4 px grid)
PreciseKeys keeps detection unchanged and re-measures:
- each square edge: gradient peak along its normal (parabolic, bilinear sampling)
  -> least-squares line per edge -> corners = line intersections
- wafer: parabolic correlation peak, then the cross's 4 long edge lines -> arm centrelines
  -> their intersection (samples hidden under mask squares are skipped)
"""
import math

import cv2
import numpy as np

from mask_aligner_optimized_v4 import CornerKeys


def parabolic(a, b, c):
    """Offset of the extremum of a parabola through (-1,a), (0,b), (1,c)."""
    d = a-2*b+c
    return 0.0 if abs(d) < 1e-12 else float(np.clip(.5*(a-c)/d, -.5, .5))


def intersect(l1, l2):
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    m = np.array([[vx1, -vx2], [vy1, -vy2]])
    if abs(np.linalg.det(m)) < 1e-9:
        return None
    t = np.linalg.solve(m, [x2-x1, y2-y1])[0]
    return np.array([x1+t*vx1, y1+t*vy1])


def refine_quad(gray, quad, samples=12):
    """Coarse-to-fine: a wide pass absorbs a 1-5 px coarse outline error, a narrow pass measures."""
    quad = np.asarray(quad, np.float64)
    side = float(np.mean(np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)))
    coarse = fit_quad(gray, quad, samples, max(4.0, side*.25), max(4.0, side*.3))
    if coarse is None:
        return None
    return fit_quad(gray, coarse, samples, 2.0, 1.5)


def fit_quad(gray, quad, samples, reach, max_shift):
    """Subpixel corners of a dark convex quad, or None if an edge fails."""
    g = gray.astype(np.float32)
    quad = np.asarray(quad, np.float64)
    centre = quad.mean(axis=0)
    side = float(np.mean(np.linalg.norm(np.roll(quad, -1, axis=0)-quad, axis=1)))
    offsets = np.arange(-reach, reach+.001, .5)
    lines = []
    for i in range(4):
        p0, p1 = quad[i], quad[(i+1) % 4]
        d = p1-p0
        d /= np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        if np.dot(n, (p0+p1)/2-centre) < 0:
            n = -n  # outward normal: profile goes dark (inside) -> light (outside)
        pts = []
        for t in np.linspace(.2, .8, samples):
            base = p0+(p1-p0)*t
            xy = base+offsets[:, None]*n
            prof = cv2.remap(g, xy[:, 0].astype(np.float32).reshape(1, -1),
                             xy[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)[0]
            grad = np.gradient(prof)
            # Nearest strong rising edge to the coarse outline, not the strongest one:
            # the wafer cross passing under a square adds a second, often larger step.
            peaks = [k for k in range(1, len(grad)-1)
                     if grad[k] >= grad[k-1] and grad[k] >= grad[k+1] and grad[k] >= .4*grad.max()]
            if not peaks or grad.max() <= 0:
                continue
            k = min(peaks, key=lambda k: abs(offsets[k]))
            off = offsets[k]+.5*parabolic(grad[k-1], grad[k], grad[k+1])
            pts.append(base+off*n)
        if len(pts) < samples//2:
            return None
        lines.append(cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel())
    corners = [intersect(lines[i-1], lines[i]) for i in range(4)]
    if any(c is None for c in corners):
        return None
    corners = np.array(corners)
    # A corner that moves more than max_shift means the fit locked onto another edge.
    order = np.argmin(np.linalg.norm(corners[:, None]-quad[None], axis=2), axis=1)
    if np.max(np.linalg.norm(corners-quad[order], axis=1)) > max_shift:
        return None
    return corners


def edge_points(g, valid, base_pts, n, reach):
    """Subpixel dark->light edge along outward normal n near each base point (skips occluded ones)."""
    offsets = np.arange(-reach, reach+.001, .5)
    pts = []
    for base in base_pts:
        xy = base+offsets[:, None]*n
        if (xy.min() < 1 or xy[:, 0].max() > g.shape[1]-2 or xy[:, 1].max() > g.shape[0]-2 or
                valid[np.round(xy[:, 1]).astype(int), np.round(xy[:, 0]).astype(int)].min() == 0):
            continue
        prof = cv2.remap(g, xy[:, 0].astype(np.float32).reshape(1, -1),
                         xy[:, 1].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR)[0]
        grad = np.gradient(prof)
        peaks = [k for k in range(1, len(grad)-1)
                 if grad[k] >= grad[k-1] and grad[k] >= grad[k+1] and grad[k] >= .4*grad.max()]
        if not peaks or grad.max() <= 0:
            continue
        k = min(peaks, key=lambda k: abs(offsets[k]))
        pts.append(base+(offsets[k]+.5*parabolic(grad[k-1], grad[k], grad[k+1]))*n)
    return pts


def fit_cross(g, valid, centre, size, angle, reach):
    """Centre/angle of a dark cross (arm width size/3) from its 4 long edge lines, or None."""
    a, h = size/6, size/2
    centerlines = []
    for arm in (0, 90):
        r = math.radians(angle+arm)
        d = np.array([math.cos(r), math.sin(r)])
        n = np.array([-d[1], d[0]])
        us = np.r_[np.linspace(-h+2, -a-2, 8), np.linspace(a+2, h-2, 8)]
        lines = []
        for sign in (1, -1):
            pts = edge_points(g, valid, [centre+u*d+sign*a*n for u in us], sign*n, reach)
            if len(pts) < 5:
                return None
            vx, vy, x, y = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
            if vx*d[0]+vy*d[1] < 0:
                vx, vy = -vx, -vy
            lines.append((vx, vy, x, y))
        (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = lines
        v = np.array([vx1+vx2, vy1+vy2])
        v /= np.linalg.norm(v)
        width = abs((x2-x1)*v[1]-(y2-y1)*v[0])
        if abs(width-2*a) > .3*2*a:
            return None  # the two lines are not the two sides of one arm
        centerlines.append((v[0], v[1], (x1+x2)/2, (y1+y2)/2))
    c = intersect(*centerlines)
    if c is None:
        return None
    ang = math.degrees(math.atan2(centerlines[0][1], centerlines[0][0]))
    return c, ang


def refine_cross(gray, valid, centre, size, angle):
    g = gray.astype(np.float32)
    centre = np.asarray(centre, np.float64)
    best = None
    # matchTemplate's rotation sign convention is not the image's; try both and keep the fit that lands.
    for a0 in (angle, -angle):
        fit = fit_cross(g, valid, centre, size, a0, max(3.0, size*.08))
        if fit is None or np.linalg.norm(fit[0]-centre) > max(3.0, size*.1):
            continue
        fine = fit_cross(g, valid, fit[0], size, fit[1], 2.0)
        if fine is not None and np.linalg.norm(fine[0]-fit[0]) < 1.5:
            fit = fine
        if best is None or np.linalg.norm(fit[0]-centre) < np.linalg.norm(best[0]-centre):
            best = fit
    return best


class PreciseKeys(CornerKeys):
    def mask_geometry(self, gray):
        g = super().mask_geometry(gray)
        if g is None:
            return None
        observed = g['observed']
        quads = g['quads'][:observed if observed == 4 else 3]
        refined = [refine_quad(gray, q) for q in quads]
        if all(r is None for r in refined):
            return g
        if any(r is None for r in refined):
            # Keep the key's grid: shift each unrefined square by the mean correction of the refined ones.
            shift = np.mean([r.mean(axis=0)-np.asarray(q).mean(axis=0) for r, q in zip(refined, quads) if r is not None], axis=0)
            refined = [r if r is not None else np.asarray(q, np.float64)+shift for r, q in zip(refined, quads)]
        centres = np.array([r.mean(axis=0) for r in refined])
        if observed == 4:
            centre = centres.mean(axis=0)
        else:
            # With three squares the two farthest apart are diagonal; the key centre is their midpoint.
            i, j = max(((0, 1), (0, 2), (1, 2)), key=lambda p: np.linalg.norm(centres[p[0]]-centres[p[1]]))
            centre = (centres[i]+centres[j])/2
        edges = np.concatenate([np.roll(r, -1, axis=0)-r for r in refined])
        z = np.exp(4j*np.arctan2(edges[:, 1], edges[:, 0])).mean()
        g = dict(g, center=centre, angle=float(np.angle(z)/4*180/np.pi) % 90,
                 quads=[r.astype(np.float32) for r in refined]+g['quads'][len(refined):])
        return g

    def wafer(self, gray, mask):
        best = super().wafer(gray, mask)
        if best is None:
            return None
        valid = np.ones(gray.shape, np.float32)
        for q in mask['quads']:
            cv2.fillConvexPoly(valid, np.round(q).astype(np.int32), 0)
        valid = 1-cv2.dilate(1-valid, np.ones((5, 5), np.uint8))
        image = 1-gray.astype(np.float32)/255
        p, s = self.pattern(best['size'], best['angle'])
        half = (p.shape[0]-1)/2
        bx, by = best['center']
        r = int(half+4)
        x0, y0 = max(0, int(round(bx))-r), max(0, int(round(by))-r)
        x1, y1 = min(gray.shape[1], int(round(bx))+r+1), min(gray.shape[0], int(round(by))+r+1)
        scene, weights = image[y0:y1, x0:x1], valid[y0:y1, x0:x1]
        if min(scene.shape) <= p.shape[0]+2:
            return best
        score = self.masked_scores(scene, weights, p, s)
        _, _, _, (lx, ly) = cv2.minMaxLoc(score)
        if not (0 < lx < score.shape[1]-1 and 0 < ly < score.shape[0]-1):
            return best
        dx = parabolic(score[ly, lx-1], score[ly, lx], score[ly, lx+1])
        dy = parabolic(score[ly-1, lx], score[ly, lx], score[ly+1, lx])
        best = dict(best, center=(x0+lx+dx+half, y0+ly+dy+half))
        # Edge-line fit; the correlation peak above is the fallback when too many edges are hidden.
        fit = refine_cross(gray, valid, best['center'], best['size'], best['angle'])
        # Correlation peak is good to ~0.3 px (p95); a fit further away locked onto the wrong edge.
        if fit is not None and np.linalg.norm(fit[0]-best['center']) < .6:
            best = dict(best, center=(float(fit[0][0]), float(fit[0][1])))
        return best
