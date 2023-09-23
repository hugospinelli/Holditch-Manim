import colorsys
import logging
import tomllib

from collections.abc import Sequence

from manim import *
from HolditchMath import get_curve_from_control_points, Curve, Locus, Ring
from CustomMobjects import PolygonalChain
from Regions import split_paths

COLOR_C = '#0000cd'
COLOR_LOCUS = '#4169e1'
COLOR_RED = '#a52a2a'
COLOR_FILL = '#0d0040'


PRESETS_FILE = 'presets.toml'
with open(PRESETS_FILE, 'rb') as f:
    PRESETS_CONFIG = tomllib.load(f)


def get_winding_color(winding, max_winding=4):
    x = min(1, abs(winding)/max_winding)
    h = 0.7 if winding > 0 else 0
    rgb = colorsys.hsv_to_rgb(h, 1, x)
    return '#%02x%02x%02x' % tuple(map(lambda x: int(round(255*x)), rgb))


class ChordVGroup(VGroup):
    def __init__(self, pa, pb, pc, ab_visible=False, lines_visible=True,
                 opacity=1, *args, **kwargs):
        self.pa = pa
        self.pb = pb
        self.pc = pc
        self.ab_visible = ab_visible
        self.lines_visible = lines_visible
        self.opacity = opacity
        (
            self.line_a, self.line_b, self.line_ab,
            self.dot_a, self.dot_b, self.dot_c
        ) = self.get_mobjects()
        super().__init__(self.line_a, self.line_b, self.line_ab,
                         self.dot_a, self.dot_b, self.dot_c,
                         *args, **kwargs)

    def get_mobjects(self):
        opacity = self.opacity if self.lines_visible else 0
        ab_opacity = opacity if self.ab_visible else 0
        line_a = Line(self.pa, self.pc, stroke_width=8, color=COLOR_RED,
                      fill_opacity=opacity, stroke_opacity=opacity,
                      z_index=3)
        line_b = Line(self.pb, self.pc, stroke_width=8, color=COLOR_RED,
                      fill_opacity=opacity, stroke_opacity=opacity,
                      z_index=3)
        line_ab = Line(self.pa, self.pb, stroke_width=8, color=COLOR_RED,
                      fill_opacity=ab_opacity, stroke_opacity=ab_opacity,
                      z_index=3)
        dot_a = Dot(self.pa, stroke_width=2, fill_color=COLOR_RED,
                    fill_opacity=opacity, stroke_opacity=opacity, z_index=4)
        dot_b = Dot(self.pb, stroke_width=2, fill_color=COLOR_RED,
                    fill_opacity=opacity, stroke_opacity=opacity, z_index=4)
        dot_c = Dot(self.pc, stroke_width=2, fill_color=COLOR_C,
                    fill_opacity=self.opacity, stroke_opacity=self.opacity,
                    z_index=5)
        return line_a, line_b, line_ab, dot_a, dot_b, dot_c

    def toggle_lines_visibility(self):
        self.lines_visible = not self.lines_visible
        line_a, line_b, line_ab, dot_a, dot_b, dot_c = self.get_mobjects()
        self.line_a.become(line_a)
        self.line_b.become(line_b)
        self.line_ab.become(line_ab)
        self.dot_a.become(dot_a)
        self.dot_b.become(dot_b)
        self.dot_c.become(dot_c)


class Holditch:
    def __init__(
        self, vt: ValueTracker, ps: str | Sequence[np.ndarray],
        center: np.ndarray = ORIGIN,
        a: float | None = None, b: float | None = None,
        scale: float = 1, speed: float = 1, min_dt: float = 1/120,
        build_path=True, stop_when_closed=False,
        starting_index: int | None = None, b_index: int | None = None,
        shortcut: bool = False
    ):
        self.vt = vt
        self.t = vt.get_value()
        self.center = center
        self.scale = scale
        self.speed = speed
        self.min_dt = min_dt
        self.config = None
        if isinstance(ps, str):
            self.config = PRESETS_CONFIG[ps]
            self.a = self.config['a'] if a is None else a/scale
            self.b = self.config['b'] if b is None else b/scale
            self._control_ps = self.config['control_points']
            self._curve_ps = get_curve_from_control_points(self._control_ps)
        else:
            if a is None or b is None:
                raise ValueError
            self.a, self.b = a, b
            if len(ps) < 2:
                raise ValueError
            self._curve_ps = Ring([p[:2] for p in ps])
        self.closed = False
        self._build_path = build_path
        self.stop_when_closed = stop_when_closed

        self._curve = Curve(self._curve_ps, shortcut)
        self._locus = Locus(self._curve, self.a, self.b, self.speed,
                            build_path=build_path,
                            stop_when_closed=stop_when_closed,
                            starting_index=starting_index,
                            b_index=b_index)

        self.curve = Polygon(*self.curve_ps, color=WHITE, z_index=2)
        self.chord = ChordVGroup(*self._get_chord_points())
        pc = self._to_manim_coordinates(self._chord.pc)
        self.locus = PolygonalChain(pc, color=COLOR_LOCUS, z_index=1)
        self.strip = VGroup()
        self.group = VGroup(self.locus, self.chord, self.strip)
        self.polys = []

        self.group.add_updater(self.update)

    @property
    def curve_ps(self):
        return Ring(self._to_manim_coordinates(self._curve_ps))

    @property
    def _chord(self):
        return self._locus.chord

    @property
    def build_path(self):
        return self._build_path

    @build_path.setter
    def build_path(self, build_path: bool):
        build_path = bool(build_path)
        if self._build_path == build_path:
            return
        self._build_path = build_path
        self._locus.build_path = build_path
        self.closed = False

    @staticmethod
    def _to_3d(ps):
        ps = np.array(ps)
        match ps.shape:
            case (2,):
                return np.append(ps, 0)
            case (_, 2):
                return np.pad(ps, ((0, 0), (0, 1)))
        return ps

    def _to_manim_coordinates(self, ps):
        return self.center + self.scale*self._to_3d(ps)

    def _get_chord_points(self):
        pa = self._chord.cpa.p
        pb = self._chord.cpb.p
        pc = self._chord.pc
        return self._to_manim_coordinates([pa, pb, pc])

    def update(self, _: VGroup | None = None):
        t = self.vt.get_value()
        dt = t - self.t
        self.t = t

        if self.closed and self.stop_when_closed:
            return

        closed_before = self.closed
        n, last_dt = divmod(dt, self.min_dt)
        for _ in range(int(n)):
            self._locus.update(self.min_dt)
            self._update_locus()
        self._locus.update(last_dt)
        self._update_locus()

        self.chord.become(ChordVGroup(
            *self._get_chord_points(),
            lines_visible=self.chord.lines_visible,
            opacity=self.chord.opacity))

        if self.closed and not closed_before:
            paths, windings, area = split_paths(self._curve.ps, self._locus.ps)
            self.polys = [
                Polygon(*self._to_manim_coordinates(path),
                        fill_color=get_winding_color(winding),
                        fill_opacity=1,
                        stroke_opacity=0,
                        z_index=0)
                for path, winding in zip(paths, windings)
            ]
            self.strip.become(VGroup(*self.polys, z_index=0))

    def _update_locus(self):
        if self.closed or not self.build_path:
            return
        if self._locus.closed:
            self.locus.add_line_to(self.locus.points[0])
            logging.info(f'Locus closed at {self.t = }.')
            print(f'\nLocus closed at {self.t = }.')
            self.closed = True
        else:
            pc = self._to_manim_coordinates(self._chord.pc)
            self.locus.add_line_to(pc)
