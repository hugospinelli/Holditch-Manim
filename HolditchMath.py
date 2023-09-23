import bisect

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np

from Ring import Ring


norm = np.linalg.norm


def get_segment_intersections(
    p0: np.ndarray, r: float, p1: np.ndarray, p2: np.ndarray
) -> list[np.ndarray]:
    """Get a list of points with distance `r` to point `p0` which
    intersect the segment from point `p1` to `p2`.
    """

    x1, y1 = p1 - p0
    x2, y2 = p2 - p0
    d1 = np.hypot(x1, y1)
    d2 = np.hypot(x2, y2)
    # Check if both inside or both outside
    if (d1 - r)*(d2 - r) > 0:
        return []  # segment does not cross the circle
    dx = x2 - x1
    dy = y2 - y1
    dr = np.hypot(dx, dy)
    D = x1*y2 - x2*y1
    sign_dy = -1 if dy < 0 else 1
    discriminant = (r*dr)**2 - D**2
    if discriminant < 0 or dr == 0:
        return []
    intersections = []
    signs = (1,) if discriminant == 0 else (1, -1)
    for sign in signs:
        x = (D*dy + sign*sign_dy*dx*np.sqrt(discriminant))/dr**2
        y = (-D*dx + sign*abs(dy)*np.sqrt(discriminant))/dr**2
        px = np.array((x, y)) + p0
        if np.inner(px-p1, p2-p1) >= 0 and np.inner(px-p2, p1-p2) >= 0:
            intersections.append(px)
    return intersections


def _get_bezier(
    anchor_1: np.ndarray, anchor_2: np.ndarray, handle: np.ndarray,
    min_dist: float = 1.0
) -> list[np.ndarray]:
    """Get a Bezier curve between points `anchor_1` and `anchor_2`."""

    a = norm(handle - anchor_1)
    b = norm(anchor_2 - handle)
    n = max(1, round((a + b)/min_dist))
    bezier = []
    for k in range(n):
        t = k/n
        pa = anchor_1 + t*(handle - anchor_1)
        pb = handle + t*(anchor_2 - handle)
        bezier.append(pa + t*(pb - pa))
    return bezier


def get_curve_from_control_points(
    ps: Sequence[Sequence[float]]
) -> Ring[np.ndarray]:
    """Get a Bezier curve through a list of control points.

    The control points are treated as the handles and the anchors are
    their pairwise midpoints.
    """

    ps = np.array(ps)
    match ps.shape:
        case (n, 2):
            #ps = np.pad(ps, ((0, 0), (0, 1)))  # Convert to 3D
            pass
        case _:
            raise ValueError('Control points must be a list of 2D arrays.'
                             f' Got dimensions {ps.shape} instead.')
    if n < 3:
        raise ValueError(f'Got only {n} control points. '
                         'Minimum of 3 required.')
    ps = Ring(ps)
    curve_ps = Ring()
    for p1, p2, p3 in ps.triples():
        c1 = (p1 + p2)/2
        c2 = (p2 + p3)/2
        curve_ps.extend(_get_bezier(c1, c2, p2))
    return curve_ps


@dataclass
class CurvePoint:
    p: np.ndarray
    """Position as a 2D vector."""

    k: int
    """Index of the segment where `p` lies in, i.e., `p` sits between
    points `ps[k]` and `ps[k+1]` of a curve defined by a list of points
    `ps`.
    """

    d: float | None = None
    """Distance of `p` along the curve from the starting point."""


class Curve:
    """A closed curve where a chord can slide along.
    Contains methods for calculating distances and intersection points.
    """

    ps: Ring[np.ndarray]
    """"Vertices as a Ring of 2D arrays."""

    n: int
    """Number of vertices."""

    length: float
    """Total length of the curve."""

    ds: Ring[float]
    """Distance of each vertex along the curve from the starting point."""

    shortcut: bool
    """If `True`, move only along positive direction for both `A` and `B`."""

    def __init__(self, ps: Ring[np.ndarray], shortcut: bool = False):
        self.ps = ps
        self.n = len(ps)
        ds = np.cumsum([norm(p2 - p1) for p1, p2 in ps.pairs()])
        self.length = 0 if len(ds) == 0 else ds[-1]
        self.ds = Ring([0])
        self.ds.extend(ds[:-1])
        self.shortcut = shortcut

    def get_d(self, p: np.ndarray, k: int) -> float:
        """Distance of `p` along the curve from the starting point."""
        return self.ds[k] + norm(p - self.ps[k])

    def get_dist(self, d1: float, d2: float) -> float:
        """Signed distance between two points along the curve, given
        their respective distances from the starting point.
        """

        s = self.length
        if s == 0:
            return 0
        return (d2 - d1 + s/2) % s - s/2  # goes from -s/2 to +s/2

    def get_cp(self, d: float) -> CurvePoint:
        """Get the point at distance `d` along the curve from the start."""

        d %= self.length
        k = bisect.bisect(self.ds, d) - 1
        v = self.ps[k + 1] - self.ps[k]
        if (v_length := norm(v)) == 0:
            p = self.ps[k]
        else:
            p = self.ps[k] + (d - self.ds[k])*v/v_length
        return CurvePoint(p, k, d)

    def slide(self, cp: CurvePoint, dist: float) -> CurvePoint:
        """Get the point at distance `dist` along the curve from `cp`."""

        return self.get_cp(cp.d + dist)

    def get_curve_intersections(
        self, r: float, cp_fixed: CurvePoint, cp0: CurvePoint,
        cutoff: float = np.inf
    ) -> list[CurvePoint]:
        """Get a list of points with distance `r` to `cp_fixed` which
        intersect a point in the curve at a distance `d` along the curve
        from `cp0`, where `d <= cutoff`.
        """

        cps = []
        if self.n < 2:
            return cps

        # Increasing k
        for dk in range(self.n//2):
            k = (cp0.k + dk) % self.n
            p1 = self.ps[k]
            p2 = self.ps[k + 1]
            if dk > 0 and self.get_dist(cp0.d, self.ds[k]) > cutoff:
                break
            for p in get_segment_intersections(cp_fixed.p, r, p1, p2):
                d = self.get_d(p, k)
                if abs(self.get_dist(d, cp0.d)) <= cutoff:
                    cps.append(CurvePoint(p, k, d))

        # Decreasing k (commented with asterisk where different from before)
        for dk in range(self.n//2):
            k = (cp0.k - dk - 1) % self.n  # *
            p1 = self.ps[k]
            p2 = self.ps[k + 1]
            if self.get_dist(self.ds[k + 1], cp0.d) > cutoff:  # *
                break
            for p in get_segment_intersections(cp_fixed.p, r, p1, p2):
                d = self.get_d(p, k)
                if abs(self.get_dist(d, cp0.d)) <= cutoff:
                    cps.append(CurvePoint(p, k, d))

        return cps


class Chord:
    """A line segment whose ends belong to the same curve."""

    curve: Curve
    """A curve containing the ends of the line segment."""

    cpa: CurvePoint
    """End point `A` as a `CurvePoint`."""

    cpb: CurvePoint
    """End point `B` as a `CurvePoint`."""

    a: float
    """Distance from point `A` to the midpoint `C`."""

    b: float
    """Distance from point `B` to the midpoint `C`."""

    ab: np.ndarray
    """Vector from point `A` to point `B`."""

    pc: np.ndarray
    """Point `C` at distance `a` from point `A` and `b` from point `B`."""

    def __init__(
        self, curve: Curve, cpa: CurvePoint, cpb: CurvePoint,
        a: float, b: float
    ):
        self.curve = curve
        self.cpa = cpa
        self.cpb = cpb
        self.a = a
        self.b = b
        self.ab = cpb.p - cpa.p
        self.pc = self.cpa.p + a*self.ab/(a + b)

    def is_ahead(self, other: Self) -> bool:
        """Check whether `other` is ahead of `self`."""

        da12 = self.curve.get_dist(self.cpa.d, other.cpa.d)
        db12 = self.curve.get_dist(self.cpb.d, other.cpb.d)
        if self.curve.shortcut and da12 > 0 and db12 > 0:
            return True
        a12 = other.cpa.p - self.cpa.p
        b12 = other.cpb.p - self.cpb.p
        rot_12 = np.cross(da12*a12, db12*b12)
        rot_ab = np.cross(self.ab, other.ab)
        if abs(rot_12) < 1e-10 and abs(rot_ab) < 1e-10:
            return da12 > 0
        return rot_12*rot_ab > 0  # same sign

    def slide(self, dist: float) -> Self | None:
        """Return a new valid chord at a distance `dist` (unsigned) from
        either end points of this chord, if any is found. Return `None`
        otherwise.
        """

        for sign in (1, -1):
            # Slide `cpa` by exactly `sign*dist`
            cpa2 = self.curve.slide(self.cpa, sign*dist)
            for cpb2 in self.curve.get_curve_intersections(
                self.a + self.b, cpa2, self.cpb, (1 + 1e-6)*abs(dist)
            ):
                new_chord = Chord(self.curve, cpa2, cpb2, self.a, self.b)
                if self.is_ahead(new_chord):
                    if dist > 0:
                        return new_chord
                elif dist < 0:
                    return new_chord

            # Slide `cpb` by exactly `sign*dist`
            cpb2 = self.curve.slide(self.cpb, sign*dist)
            for cpa2 in self.curve.get_curve_intersections(
                self.a + self.b, cpb2, self.cpa, (1 + 1e-6)*abs(dist)
            ):
                new_chord = Chord(self.curve, cpa2, cpb2, self.a, self.b)
                if self.is_ahead(new_chord):
                    if dist > 0:
                        return new_chord
                elif dist < 0:
                    return new_chord

        return None


class Locus:
    """Locus of a point of a sliding chord in a closed curve."""

    curve: Curve
    """A curve containing the ends of the line segment."""

    a: float
    """Distance from point `A` to the midpoint `C`."""

    b: float
    """Distance from point `B` to the midpoint `C`."""

    r: float
    """Length of the chord (`a + b`)."""

    ps: Ring[np.ndarray]
    """Vertices as a Ring of 2D arrays."""

    closed: bool
    """Indicate whether the locus has completed by closing a loop."""

    chord: Chord | None
    """The current chord."""

    start_chord: Chord | None
    """The starting chord. Used for checking when the locus has closed."""

    build_path: bool
    """Flag for keeping track of the C point and building its path."""

    starting_index: int | None
    """Index of the vertex to be used as the starting point.
    If `None`, use the vertex with smallest y-coordinate value.
    """

    b_index: int | None
    """Force the starting chord to take the `b_index`-th intersection."""

    def __init__(
            self, curve: Curve, a: float, b: float, speed: float,
            build_path: bool = True, stop_when_closed: bool = False,
            starting_index: int | None = None, b_index: int | None = None
    ):
        self.curve = curve
        self.speed = speed
        self._build_path = build_path
        self.stop_when_closed = stop_when_closed
        self.a = a
        self.b = b
        self.r = self.a + self.b
        self.ps = Ring()
        self.closed = False
        self.chord = None
        self.start_chord = None

        if len(curve.ps) < 2:
            return
        if starting_index is None:
            # start at the lowest point
            starting_index = np.argmin([p[1] for p in curve.ps])
        self.chord = self.get_chord_by_index(starting_index, b_index)
        if self.chord is None:
            raise RuntimeError('No starting chord found. Perhaps the '
                               '`a` and `b` lenghts are too big.')

        self.build_path = build_path  # trigger property setter

    def get_chord_by_index(self, index: int, b_index: int | None = None):
        """Get a Chord with point `A` exactly at vertex of given index."""

        cpa = CurvePoint(self.curve.ps[index], index, self.curve.ds[index])
        cpbs = self.curve.get_curve_intersections(self.r, cpa, cpa)
        if len(cpbs) == 0:
            return
        if b_index is not None:
            return Chord(self.curve, cpa, cpbs[b_index], self.a, self.b)
        cpb = cpbs[0]
        # Find the nearest one along the positive direction
        dist0 = (cpb.d - cpa.d) % self.curve.length
        for cp in cpbs[1:]:
            dist = (cp.d - cpa.d) % self.curve.length
            if dist < dist0:
                cpb = cp
                dist0 = dist
        return Chord(self.curve, cpa, cpb, self.a, self.b)

    @property
    def build_path(self):
        return self._build_path

    @build_path.setter
    def build_path(self, build_path: bool):
        build_path = bool(build_path)
        self._build_path = build_path
        if build_path:
            self.start_chord = self.chord
            self.ps = Ring([self.chord.pc])
        else:
            self.ps = Ring()
        self.closed = False

    def update(self, dt: float):
        """Slide the chord and update the curve."""

        dist = dt*self.speed
        if self.chord is None or abs(dist) < 1e-10:
            return
        if self.closed and self.stop_when_closed:
            return
        # Try up to 10 times if larger distances fail
        for _ in range(10):
            if (new_chord := self.chord.slide(dist)) is None:
                dist /= 2
        if new_chord is None:
            return
        if self.closed:
            self.chord = new_chord
            return
        if self._build_path:
            dist_start = self.curve.get_dist(new_chord.cpa.d,
                                             self.start_chord.cpa.d)
            if (
                abs(dist_start) < (1 + 0.01)*dist  # plus small tolerance
                and self.chord.is_ahead(self.start_chord)
                and not new_chord.is_ahead(self.start_chord)
            ):
                self.closed = True
                if self.stop_when_closed:
                    self.chord = self.start_chord
                else:
                    self.chord = new_chord
                return
            self.ps.append(new_chord.pc)
        self.chord = new_chord
