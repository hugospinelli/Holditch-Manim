import itertools
import os

from manim import *
import numpy as np
import scipy as sp

from HolditchMath import get_segment_intersections
from HolditchMobjects import Holditch, ChordVGroup
from CustomMobjects import PolygonalChain
from Trammel import Trammel
from Ring import Ring


COLOR_RED = '#800000'
COLOR_DARK_RED = '#400000'
COLOR_GREEN = '#008000'
COLOR_DARK_GREEN = '#004000'
COLOR_BLUE = '#4169e1'
COLOR_DARK_BLUE = '#0d0040'


IMAGES_DIR = 'Images'
_CC0_PATH = os.path.join(IMAGES_DIR, 'CC0', 'cc-zero.svg')
CC0 = SVGMobject(_CC0_PATH, height=0.3, z_index=100).to_edge(DR, buff=0.05)


norm = np.linalg.norm


class Delay(Animation):
    def __init__(self, run_time: float):
        super().__init__(Mobject(), run_time=run_time)


def set_opacity(mob, stroke=None, fill=None):
    if stroke is None:
        stroke = mob.get_stroke_opacity()
    if fill is None:
        fill = mob.get_fill_opacity()
    mob.set_stroke(mob.get_stroke_color(), opacity=stroke)
    mob.set_fill(mob.get_fill_color(), opacity=fill)


def rotate(v, angle, about_point=ORIGIN):
    s, c = np.sin(angle), np.cos(angle)
    m = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    if len(v) == 2:
        v = v - about_point[:2]
        return m[:2,:2]@v + about_point
    v = v - about_point
    return m@v + about_point


def get_rotate_and_shift_homotopy(angle, about_point, shift):
    def homotopy(x, y, z, t):
        return rotate((x, y, z), angle*t, about_point) + shift*t
    return homotopy


def shift_and_fade(mob, shift, run_time=1, out=False, fill=True, stroke=True):
    start = mob.get_center()
    def update(mob, t, start=start, run_time=run_time, out=out,
               fill=fill, stroke=stroke):
        opacity = t/run_time
        if out:
            opacity = 1 - opacity
        if fill:
            mob.set_fill(mob.get_fill_color(), opacity=opacity)
        if stroke:
            mob.set_stroke(mob.get_stroke_color(), opacity=opacity)
        mob.move_to(start + shift*t/run_time)
    return UpdateFromAlphaFunc(mob, update)


def indicate_line(group, color=TEAL, revert_color=True):
    scale = 2.0
    submobjects = getattr(group, 'submobjects', group)
    f = lambda x: rate_functions.there_and_back(0.5*x)
    anims1 = []
    for part in submobjects:
        target = part.copy()
        target.set_fill(color, opacity=part.fill_opacity)
        target.set_stroke(color, width=scale*part.stroke_width,
                          opacity=part.stroke_opacity)
        anims1.append(Transform(part, target, rate_func=f))
    anims2 = []
    for part in submobjects:
        target = part.copy()
        if not revert_color:
            target.set_stroke(color, width=part.stroke_width)
        anims2.append(Transform(part, target, rate_func=f))
    return Succession(AnimationGroup(*anims1), AnimationGroup(*anims2))


def indicate_tex(group, color=TEAL, revert_color=True):
    scale = 1.2
    submobjects = getattr(group, 'submobjects', group)
    f = lambda x: rate_functions.there_and_back(0.5*x)
    anims1 = []
    for part in submobjects:
        target = part.copy()
        target.scale(scale)
        target.set_color(color)
        anims1.append(Transform(part, target, rate_func=f))
    anims2 = []
    for part in submobjects:
        target = part.copy()
        if not revert_color:
            target.set_color(color)
        anims2.append(Transform(part, target, rate_func=f))
    return Succession(AnimationGroup(*anims1), AnimationGroup(*anims2))


def indicate_group(group, color=TEAL, revert_color=True):
    lines = []
    texs = []
    for part in group.submobjects:
        if isinstance(part, MathTex):
            texs.append(part)
        else:
            lines.append(part)
    return AnimationGroup(indicate_line(lines, color, revert_color),
                          indicate_tex(texs, color, revert_color))


def trim_sharp_vertices(ps, min_angle=5*DEGREES, min_dist=0.01):
    if len(ps) < 3:
         return np.array([])
    ps = Ring(ps)
    k = 0
    while k < len(ps):
        p1, p2, p3 = ps[k:k+3]
        d12, d23 = norm(p2 - p1), norm(p3 - p2)
        sharp_angle = False
        if d12 >= min_dist and d23 >= min_dist:
            if angle_between_vectors(p1 - p2, p3 - p2) >= min_angle:
                k += 1
                continue
            sharp_angle = True
        if d12 < min_dist or sharp_angle:  # remove p2
            ps.pop(k + 1)
            if k == len(ps):
                k = max(0, k - 2)
            else:
                k = max(0, k - 1)
        else:  # remove p3
            ps.pop(k + 2)
            if k >= len(ps) - 1:
                k = max(0, k - 1)
    return np.array(ps)


def make_convex(poly, min_angle=45*DEGREES, dist=0.005):
    #length = poly.get_nth_curve_length(0)
    #n = int(length//dist)
    #ps = Ring([poly.point_from_proportion(k/n)
    #           for k in range(n)])
    ps = Ring(poly.points[:-1])
    v = np.array([1, 1j, 0])
    k = 0
    while k < len(ps):
        p1, p2, p3 = ps[k:k+3]
        angle1 = np.angle((p2 - p1)@v)
        angle2 = np.angle((p3 - p2)@v)
        da = (angle2 - angle1 + PI) % TAU - PI
        if -PI + min_angle <= da < 0.02:
            k += 1
        else:
            ps.pop(k + 1)
            if k + 1 > len(ps):
                k -= 1
            k -= 1
    return np.array(ps)


def triangle_wave(x, period):
    return 2*abs(sp.signal.sawtooth(np.pi*x/period - np.pi/4, 0.5)) - 1


def get_tex_map_anims(texs, maps, target, copy_flags=None):
    if copy_flags is None:
        copy_flags = len(texs)*[False]
    anims = []
    mapped = set()
    for tex, map, copy in zip(texs, maps, copy_flags):
        if copy:
            tex = tex.copy()
        for k, v in map.items():
            v = [v] if isinstance(v, int) else v
            anims.append(ReplacementTransform(tex[k], target[v[0]]))
            anims.extend([ReplacementTransform(tex[k].copy(), target[t])
                          for t in v[1:]])
            mapped.update(v)
        if not copy:
            anims.extend([FadeOut(v, shift=RIGHT + UP, scale=0)
                          for k, v in enumerate(tex) if k not in map])
    anims.extend([FadeIn(v) for k, v in enumerate(target) if k not in mapped])
    return anims


class Scene02(Scene):
    def construct(self):
        self.add(CC0)

        scale = 0.04
        A, B = scale*30, scale*60
        D = scale*50
        text = Tex('What about other curves?').move_to(3*UP)
        self.wait()
        self.play(Write(text), run_time=0.7)
        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'rectangle', ORIGIN, a=A, b=B, scale=scale,
                            speed=100, stop_when_closed=True)
        self.play(FadeIn(holditch.curve))
        self.play(FadeIn(holditch.chord.line_a),
                  FadeIn(holditch.chord.line_b),
                  FadeIn(holditch.chord.dot_a),
                  FadeIn(holditch.chord.dot_b),
                  FadeIn(holditch.chord.dot_c))
        self.add(holditch.group)
        self.play(FadeOut(text), run_time=0.5)

        dt = 7.55
        self.play(vt.animate.set_value(time := time + dt),
                  run_time=dt, rate_func=linear)
        self.wait(0.5)

        fill_color = holditch.polys[0].get_fill_color()
        stroke_color = holditch.locus.get_stroke_color()
        for poly in holditch.polys:
            ps = trim_sharp_vertices(poly.get_vertices(), 85*DEGREES)
            poly.become(Polygon(*ps,
                                fill_color=fill_color,
                                fill_opacity=1,
                                stroke_color=stroke_color,
                                stroke_opacity=1))
        self.remove(holditch.strip)
        m = [[0, -1], [-1, 0]]
        self.play(
            holditch.polys[0].animate.apply_matrix(m).shift(D*LEFT + 2*D*DOWN),
            holditch.polys[1].animate.shift(2*D*LEFT + D*UP),
            holditch.polys[2].animate.apply_matrix(m).shift(D*RIGHT + 2*D*UP),
            holditch.polys[3].animate.shift(2*D*RIGHT + D*DOWN),
            run_time=2
        )
        for poly in holditch.polys:
            poly.set_z_index(1)
        self.wait()

        chord_a = VGroup(holditch.chord.line_a,
                         holditch.chord.dot_a,
                         holditch.chord.dot_c).set_z_index(2)
        chord_b = VGroup(holditch.chord.line_b,
                         holditch.chord.dot_b,
                         holditch.chord.dot_c.copy()).set_z_index(3)
        run_time = 1
        homotopy = get_rotate_and_shift_homotopy(
            90*DEGREES, (2*D - A)*LEFT + D*DOWN, A*LEFT
        )
        self.play(Homotopy(homotopy, chord_b), run_time=run_time)
        homotopy = get_rotate_and_shift_homotopy(
            -90*DEGREES, 2*D*LEFT + D*DOWN, 2*D*RIGHT + D*UP
        )
        self.wait(0.5)
        ellipse = Ellipse(2*B, 2*A,
                          stroke_color=stroke_color,
                          stroke_opacity=1,
                          fill_color=fill_color,
                          fill_opacity=1).set_z_index(0)
        self.play(
            Homotopy(homotopy, chord_a),
            Homotopy(homotopy, chord_b),
            FadeIn(ellipse),
            run_time=run_time
        )
        self.add(ellipse)
        brace_a = BraceLabel(chord_a, 'a', brace_direction=LEFT)
        brace_b = BraceLabel(chord_b, 'b', brace_direction=UP)
        self.play(
            FadeIn(brace_a), FadeIn(brace_b),
            *[FadeOut(poly) for poly in holditch.polys]
        )
        text = MathTex(r'\text{Area} &= \pi ab', r'\\&= \pi').move_to(3*UP)
        self.play(Write(text[0]), run_time=0.7)
        self.play(Write(text[1]), run_time=0.7)
        self.wait(2)

        temp_axis = VMobject().set_points_as_corners((
            D*DOWN,
            2*D*RIGHT + D*DOWN,
            2*D*RIGHT + 0.9*D*UP
        ))
        homotopy = get_rotate_and_shift_homotopy(
            run_time, 90*DEGREES, A*DOWN, (2*D - B)*RIGHT + (D - A)*DOWN
        )
        self.play(
            FadeOut(text), FadeOut(brace_a), FadeOut(brace_b),
            Homotopy(homotopy, chord_a),
            chord_b.animate.shift((2*D - B)*RIGHT + D*DOWN),
            ellipse.animate.shift(2*D*RIGHT + D*DOWN),
            FadeIn(temp_axis),
            run_time=run_time
        )
        self.wait(0.5)

        line_x = Line(D*DOWN, D*DOWN + 4*D*RIGHT)
        line_x.set_fill(WHITE, opacity=0)
        line_y = Line(0.9*D*UP + 2*D*RIGHT, 2.9*D*DOWN + 2*D*RIGHT)
        line_y.set_fill(WHITE, opacity=0)
        shift = 2*D*LEFT + D*UP
        self.play(
            shift_and_fade(holditch.locus, shift,
                           run_time=run_time, out=True, fill=False),
            shift_and_fade(holditch.curve, shift,
                           run_time=run_time, out=True, fill=False),
            ellipse.animate.shift(shift),
            chord_a.animate.shift(shift),
            chord_b.animate.shift(shift),
            temp_axis.animate.shift(shift),
            shift_and_fade(line_x, shift, run_time=run_time),
            shift_and_fade(line_y, shift, run_time=run_time),
            run_time=run_time
        )

        self.remove(chord_a, chord_b, temp_axis)
        chord = ChordVGroup(holditch.chord.dot_a.get_center(),
                            holditch.chord.dot_b.get_center(),
                            holditch.chord.dot_c.get_center())
        self.add(chord)
        time = 0
        vt.set_value(time)
        T = 3  # period
        T8 = T/8
        T4 = T/4
        v = 4*(A + B)*np.sqrt(2)/T
        def xy(t):
            k, t = divmod(t + T8, T4)
            k %= 4
            t -= T8
            a = v*t
            b = np.sqrt((A + B)**2 - a**2)
            if k == 0:
                return -b, a
            if k == 1:
                return a, b
            if k == 2:
                return b, -a
            return -a, -b
        vo = ValueTracker(1)  # opacity
        def updater(chord):
            x, y = xy(vt.get_value())
            pa = x*RIGHT
            pb = y*UP
            pc = scale*30*(pb - pa)/norm(pb - pa) + pa
            chord.become(ChordVGroup(pa, pb, pc, opacity=vo.get_value()))
        dt = 3
        chord.add_updater(updater)
        self.play(vt.animate.set_value(time := time + dt),
                  run_time=dt, rate_func=linear)
        text = Tex(r'This is the ellipsograph,\\or trammel of Archimedes.',
                   font_size=36)
        text.move_to(3*LEFT + 3*UP)
        self.play(Write(text),
                  vt.animate.set_value(time := time + dt),
                  run_time=dt, rate_func=linear)
        text2 = Tex(r'This is a well-known mechanism\\for drawing an ellipse.',
                   font_size=36)
        text2.move_to(3*RIGHT + 3*UP)
        self.play(Write(text2),
                  vt.animate.set_value(time := time + dt),
                  run_time=dt, rate_func=linear)
        self.play(vt.animate.set_value(time := time + dt),
                  vo.animate.set_value(0),
                  FadeOut(line_x),
                  FadeOut(line_y),
                  FadeOut(ellipse),
                  Succession(Unwrite(text, reverse=False),
                             Unwrite(text2, reverse=False)),
                  run_time=dt, rate_func=linear)
        self.wait(0.1)


class Scene03(Scene):
    def construct(self):
        self.add(CC0)

        img = ImageMobject(
            os.path.join(IMAGES_DIR, 'Trammel', 'PNGs', '0002.png')
        )

        class GIF(ImageMobject):
            def __init__(self, img_paths, time_per_frame, alpha=0,
                         *args, **kwargs):
                super().__init__(img_paths[0], *args, **kwargs)
                self.img_paths = img_paths
                self.time = 0
                self.time_per_frame = time_per_frame
                self.frame_count = 0
                self.alpha = alpha
                self.fade_alpha = None
                self.fade_time = None
                self.fade_start_time = None
                self.add_updater(self.update_img)

            def set_fade(self, alpha, fade_time):
                self.fade_alpha = alpha
                self.fade_time = fade_time
                self.fade_start_time = self.time
                return self

            def update_img(self, _, dt):
                self.time += dt
                self.frame_count = int(self.time//self.time_per_frame)
                new_image = ImageMobject(
                    self.img_paths[self.frame_count],
                    z_index = self.get_z_index()
                )
                new_image.width = self.width
                new_image.height = self.height
                new_image.move_to(self)

                alpha = self.alpha
                if self.fade_alpha is not None:
                    ratio = (self.time - self.fade_start_time)/self.fade_time
                    alpha = self.alpha + ratio*(self.fade_alpha - self.alpha)
                    alpha = np.clip(alpha, 0, 1)
                    if self.time >= self.fade_start_time + self.fade_time:
                        self.fade_alpha = None
                        self.fade_time = None
                        self.fade_start_time = None
                        self.alpha = alpha
                new_image.pixel_array[:,:,3] = np.round(
                    alpha*new_image.pixel_array[:,:,3]
                ).astype(int)
                self.become(new_image)

        N = 120
        trammel_paths = Ring([
            os.path.join(IMAGES_DIR, 'Trammel', 'PNGs', '%04d.png' % (k % N))
            for k in range(2, N + 2)
        ])
        trammel = GIF(trammel_paths, 1/30, alpha=1)

        text = Tex(r'The C point is usually placed\\'
                   'beyond the AB segment.').move_to(3*UP)
        text_a = Tex('A').move_to(1.95*LEFT + 0.5*UP)
        text_b = Tex('B').move_to(0.55*RIGHT + 1*UP)
        text_c = Tex('C').move_to(6.5*LEFT + 0.5*DOWN)
        self.play(FadeIn(img))
        self.play(LaggedStart(
            Write(text),
            AnimationGroup(Write(text_a), Write(text_b), Write(text_c)),
            lag_ratio=0.5))
        self.wait(3)

        self.remove(img)
        self.add(trammel)
        self.play(
            Unwrite(text, reverse=False),
            AnimationGroup(Unwrite(text_a), Unwrite(text_b), Unwrite(text_c)),
            run_time=0.5
        )
        self.wait(8)
        trammel.set_fade(alpha=0, fade_time=1.5)
        self.wait(1.5)
        trammel.remove_updater(trammel.update_img)
        self.remove(trammel)
        self.wait(0.1)


class MaskExample(Scene):
    def construct(self):

        class TextMask:
            color = DARK_BLUE
            def __init__(self, vt: ValueTracker, texts: list[Tex],
                         times: list[float], min_z_index: int = 0):
                self.vt = vt
                self.texts = texts
                self.times = np.cumsum(times)
                self.step = 0
                self.set_dims()
                self.rect = Rectangle(
                    self.color, self.height, self.width,
                    fill_color=self.color, fill_opacity=1
                ).move_to(self.center)

                self.rect.set_z_index(min_z_index + 2*len(texts) - 2)
                for k, text in enumerate(reversed(texts)):
                    text.set_z_index(min_z_index + 2*k + 1)

            def set_dims(self):
                minx, miny = maxx, maxy = self.texts[1].get_center()[:2]
                for text in self.texts[1:]:
                    c = text.get_center()
                    minx = min(minx, c[0] - text.width/2)
                    maxx = max(maxx, c[0] + text.width/2)
                    miny = min(miny, c[1] - text.height/2)
                    maxy = max(maxy, c[1] + text.height/2)
                self.center = np.array([(maxx + minx)/2, (maxy + miny)/2, 0])
                self.width = maxx - minx
                self.height = maxy - miny

            def update(self):
                if self.step >= len(self.texts) - 1:
                    return
                t = self.vt.get_value()
                if t < self.times[self.step]:
                    return
                self.step += 1
                self.rect.set_z_index(self.rect.get_z_index() - 2)

        texts = [Tex('A'), Tex('BB'), Tex('C').shift(0.1*DOWN), Tex('D')]
        times = [2, 3, 1, 1.5]

        vt = ValueTracker(0)
        vt_updater = lambda vt, dt: vt.increment_value(dt)
        vt.add_updater(vt_updater)

        mask = TextMask(vt, texts, times)
        mask_updater = lambda _: mask.update()
        mask.rect.add_updater(mask_updater)

        self.add(vt, mask.rect)
        dt_out = 0.5
        anims = [Succession(FadeIn(text, run_time = time - dt_out),
                            FadeOut(text, run_time = dt_out))
                 for text, time in zip(texts, times)]
        self.play(Succession(*anims))
        self.wait()


class Scene04(Scene):
    def construct(self):
        self.add(CC0)

        A, B = 0.9, 1.8
        time = 0
        vt = ValueTracker(time)
        trammel = Trammel(vt, DOWN, 90*DEGREES, A, B, v=4.45)
        trammel_updater = lambda _: trammel.update()
        trammel_line_y_updater = lambda _: trammel.update_line_y()
        trammel_chord_updater = lambda _: trammel.update_chord()
        trammel_ellipse_updater = lambda _: trammel.update_ellipse()
        vt.add_updater(trammel_updater)
        trammel.line_y.add_updater(trammel_line_y_updater)
        trammel.chord.add_updater(trammel_chord_updater)
        trammel.ellipse.add_updater(trammel_ellipse_updater)

        period = 8
        def update_angle():
            angle = 120*DEGREES
            t = vt.get_value()
            angle = (90 - 30*triangle_wave(t, period))*DEGREES
            if t < (3/4)*period:
                trammel.angle = angle
            elif t < (1 + 3/4)*period:
                trammel.angle = angle
                trammel.h = triangle_wave(t - (3/4)*period, 2*period)
            else:
                trammel.angle = 120*DEGREES
                trammel.h = 0
                trammel.stop_at_chord_angle(0)

        trammel_angle_updater = lambda _: update_angle()
        vt.add_updater(trammel_angle_updater)


        class TextMask:
            color = BLACK
            def __init__(self, vt: ValueTracker, texts: list[Tex],
                         times: list[float], min_z_index: int = 0):
                self.vt = vt
                self.texts = texts
                self.times = np.cumsum(times)
                self.step = 0
                self.set_dims()
                self.rect = Rectangle(
                    self.color, self.height, self.width,
                    fill_color=self.color, fill_opacity=1
                ).move_to(self.center)

                self.rect.set_z_index(min_z_index + 2*len(texts) - 2)
                for k, text in enumerate(reversed(texts)):
                    text.set_z_index(min_z_index + 2*k + 1)

            def set_dims(self):
                minx, miny = maxx, maxy = self.texts[1].get_center()[:2]
                for text in self.texts[1:]:
                    c = text.get_center()
                    minx = min(minx, c[0] - text.width/2)
                    maxx = max(maxx, c[0] + text.width/2)
                    miny = min(miny, c[1] - text.height/2)
                    maxy = max(maxy, c[1] + text.height/2)
                self.center = np.array([(maxx + minx)/2, (maxy + miny)/2, 0])
                self.width = maxx - minx
                self.height = maxy - miny

            def update(self):
                if self.step >= len(self.texts) - 1:
                    return
                t = self.vt.get_value()
                if t < self.times[self.step]:
                    return
                self.step += 1
                self.rect.set_z_index(self.rect.get_z_index() - 2)

        texts = [
            Tex(r'What is less well-known is that\\',
                'it also works when the angle is oblique.'),
            Tex(r'And the point C does not need\\',
                'to lie on the AB line.'),
            Tex(r'This is known as\\'
                "van Schooten's Locus Problem.")
        ]
        for text in texts:
            text.move_to(3*UP)
        times = [4, 5, 5.5]

        self.play(Write(texts[0]),
                  FadeIn(trammel.line_x),
                  FadeIn(trammel.line_y),
                  FadeIn(trammel.chord),
                  FadeIn(trammel.ellipse),
                  run_time=2)

        #dt = (1 + 3/4)*period + 0.1

        vt_updater = lambda vt, dt: vt.increment_value(dt)
        vt.add_updater(vt_updater)

        mask = TextMask(vt, texts, times)
        mask_updater = lambda _: mask.update()
        mask.rect.add_updater(mask_updater)

        self.add(vt, mask.rect)
        dt_out = 0.5
        write1 = Write(texts[1])
        write2 = Write(texts[2])
        delay1 = Delay(times[1] - dt_out - write1.get_run_time())
        delay2 = Delay(times[2] - dt_out - write2.get_run_time())
        self.play(Succession(
            Delay(times[0] - dt_out),
            FadeOut(texts[0], run_time=dt_out),
            Succession(write1, delay1),
            FadeOut(texts[1], run_time=dt_out),
            Succession(write2, delay2),
            Unwrite(texts[2], reverse=False, run_time=dt_out)
        ))
        self.wait(0.1)


class Scene05(Scene):
    def construct(self):
        self.add(CC0)

        A, B = 0.9, 1.8
        L = A + B
        time = 0
        vt = ValueTracker(time)
        trammel = Trammel(vt, DOWN, 120*DEGREES, A, B, v=4.45)

        trammel_group = VGroup(trammel.line_x, trammel.line_y,
                               trammel.chord, trammel.ellipse)
        trammel_group_copy = trammel_group.copy()
        scale = 2
        center = trammel.center + 1.5*LEFT + 0.5*DOWN
        trammel_group.scale(scale).move_to(center)
        trammel_group[2].dot_a.scale(1/scale)
        trammel_group[2].dot_b.scale(1/scale)
        trammel_group[2].dot_c.scale(1/scale)
        A *= scale
        B *= scale
        L *= scale

        self.play(ReplacementTransform(trammel_group_copy, trammel_group))
        line_x, line_y, chord, ellipse = trammel_group

        def get_angle_pos(c, angle, r):
            return r*rotate_vector(RIGHT, angle) + c

        def get_line_pos(p1, angle, l, r):
            p2 = p1 + l*rotate_vector(RIGHT, angle)
            return (p2 + p1)/2 + r*get_unit_normal(p2 - p1, IN)

        r = 0.3*scale
        a2, b2, beta = trammel.get_ellipse_params()
        a2 *= scale
        b2 *= scale

        dash_a = DashedLine(1.2*a2*LEFT, 1.2*a2*RIGHT)
        dash_a.rotate(beta).shift(center)

        dash_b = DashedLine(1.2*b2*LEFT, 1.2*b2*RIGHT)
        dash_b.rotate(beta + PI/2).shift(center)

        seg_a2 = Line(ORIGIN, a2*RIGHT, color=GREEN)
        seg_a2.rotate(beta + PI, about_point=ORIGIN).shift(center)

        seg_b2 = Line(ORIGIN, b2*RIGHT, color=GREEN)
        seg_b2.rotate(beta + PI/2, about_point=ORIGIN).shift(center)

        angle_alpha = Angle(line_x, line_y, r)
        pos = get_angle_pos(center, trammel.angle/2, 1.5*r)
        text_alpha = MathTex(r'\alpha').move_to(pos)

        angle_beta = Angle(line_x, dash_a, r, (-1, -1), color=PURPLE_B)
        pos = get_angle_pos(center, PI + beta/2 + 0.1, 1.5*r)
        text_beta = MathTex(r'\beta', color=PURPLE_B).move_to(pos)
        text_beta.set_z_index(2)

        pos = get_line_pos(center, beta + PI, a2, -0.5*r)
        text_a2 = MathTex("a'", color=GREEN).move_to(pos)

        pos = get_line_pos(center, beta + PI/2, b2, -0.5*r)
        text_b2 = MathTex("b'", color=GREEN).move_to(pos)

        pos = get_line_pos(center + L*LEFT, 0, A, -0.5*r)
        text_a = MathTex('a', color=chord.color).move_to(pos)

        pos = get_line_pos(center + B*LEFT, 0, B, -0.5*r)
        text_b = MathTex('b', color=chord.color).move_to(pos)

        self.play(
            Create(dash_a), Create(dash_b),
            Create(seg_a2), Create(seg_b2),
            Create(angle_alpha), Create(angle_beta),
            Write(text_alpha), Write(text_beta),
            Write(text_a2), Write(text_b2),
            Write(text_a), Write(text_b)
        )
        self.wait()

        tex_beta, tex_a2_b2 = trammel.get_tex_formulas()
        a2_b2_formula = MathTex(tex_a2_b2, font_size=DEFAULT_FONT_SIZE/2,
                                color=GREEN)
        a2_b2_formula.move_to(2.8*UP + 1.5*RIGHT)
        beta_formula = MathTex(tex_beta, font_size=DEFAULT_FONT_SIZE/2,
                               color=PURPLE_B)
        beta_formula.move_to(1.2*UP + 2*RIGHT)
        self.play(Write(beta_formula),
                  Write(a2_b2_formula))
        self.wait(2)

        line_y_copy = line_y.copy()
        line_y.become(Line(1.5*A*LEFT, 1.5*A*RIGHT))
        line_y.rotate(trammel.angle, about_point=ORIGIN).shift(center)
        line_y.set_z_index(line_y_copy.get_z_index())
        self.play(
            Unwrite(beta_formula), Unwrite(a2_b2_formula),
            Uncreate(seg_a2), Uncreate(seg_b2),
            Uncreate(dash_a), Uncreate(dash_b),
            Unwrite(text_beta), Uncreate(angle_beta),
            Unwrite(text_a2), Unwrite(text_b2),
            ReplacementTransform(line_y_copy, line_y)
        )

        triangle = Polygon(center + 10*LEFT, center,
                           center + rotate_vector(10*RIGHT, trammel.angle))
        sector = Intersection(ellipse, triangle,
                              fill_color=ellipse.get_fill_color(),
                              fill_opacity=1,
                              stroke_color=ellipse.get_stroke_color(),
                              stroke_opacity=1,
                              z_index=-3)
        original_sector = sector.copy()
        sector.set_z_index(-1)
        sector.set_fill(PURPLE_E, opacity=0.5)
        sector.set_stroke(PURPLE_A, opacity=1)
        ellipse.set_z_index(-2)
        original_ellipse = ellipse.copy()
        self.add(original_ellipse)

        dots = [Dot(ellipse.point_from_proportion(x), color=MAROON, z_index=5)
                for x in np.linspace(0, 1, num=12, endpoint=False)]

        shear_scale = 0.05
        def shear(pos):
            c = L/(A*np.tan(trammel.angle))
            return -shear_scale*c*RIGHT*(pos - center)[1]

        w = 2 + config['frame_width']//2
        vector_field = ArrowVectorField(shear, length_func=linear,
                                        x_range=[-w, w],
                                        colors=[MAROON_A, MAROON_E])
        nudge_run_time = 1.5
        speed = 1/(nudge_run_time*shear_scale)
        vf_pointwise_updater = vector_field.get_nudge_updater(
            speed=speed, pointwise=True)
        vf_updater = vector_field.get_nudge_updater(
            speed=speed, pointwise=False)
        def vt_self_updater(mob, dt):
            for arrow in mob:
                vector_field.nudge(arrow, 0.2*speed*dt)
                x = arrow.get_center()[0]
                if x > w:
                    arrow.shift(2*w*LEFT)
                if x < -w:
                    arrow.shift(2*w*RIGHT)

        vector_field.add_updater(vt_self_updater)
        self.play(FadeIn(vector_field),
                  FadeIn(sector), FadeIn(original_sector),
                  *[GrowFromCenter(dot) for dot in dots])
        self.wait(0.5)
        ellipse.add_updater(vf_pointwise_updater)
        sector.add_updater(vf_pointwise_updater)
        for dot in dots:
            dot.add_updater(vf_updater)
        self.wait(nudge_run_time)
        self.add(ellipse)
        self.add(sector)
        self.add(*dots)
        ellipse.remove_updater(vf_pointwise_updater)
        sector.remove_updater(vf_pointwise_updater)
        for dot in dots:
            dot.remove_updater(vf_updater)
        self.play(*[FadeOut(dot) for dot in dots], FadeOut(vector_field))
        vector_field.remove_updater(vt_self_updater)


        formula_font_size = round(0.8*DEFAULT_FONT_SIZE)
        formula1 = MathTex(
            #0     1              2       3              4
            '(', r'a\cos\alpha', ',\,', r'a\sin\alpha', ')',
            font_size=formula_font_size
        )
        formula2 = MathTex(
            #0    1     2      3     4                      5
            '(', 'x', r',\,', 'y', r') \longrightarrow ', r'\left(',
            #6    7      8     9      10    11
            'x', '+ c', 'y', r',\,', 'y', r'\right)',
            font_size=formula_font_size
        )
        formula3 = MathTex(
            #0    1     2      3     4                      5         6
            '(', 'x', r',\,', 'y', r') \longrightarrow ', r'\left(', 'x',
            # 7
            r'-\frac{(a + b)\cos\alpha}{a\sin\alpha}',
            # 8         9     10     11    12
            r'\cdot ', 'y', r',\,', 'y', r'\right)',
            font_size=formula_font_size
        )
        formula4 = MathTex(
            '(', r'a\cos\alpha', ',\,', r'a\sin\alpha',
            r') \longrightarrow ', r'\left(', r'a\cos\alpha',
            r'-\frac{(a + b)\cos\alpha}{a\sin\alpha}',
            r'\cdot ', r'a\sin\alpha', r',\,', r'a\sin\alpha', r'\right)',
            font_size=formula_font_size
        )
        formula5 = MathTex(
            #0     1               2       3
            '(', r'a\cos\alpha', r',\,', r'a\sin\alpha',
            # 4                      5          6
            r') \longrightarrow ', r'\left(', r'-b\cos\alpha',
            # 7       8               9
            r',\,', r'a\sin\alpha', r'\right)',
            font_size=formula_font_size
        )
        formula6 = MathTex(
            r'\left(', r'-b\cos\alpha', r',\,', r'a\sin\alpha', r'\right)',
            font_size=formula_font_size
        )

        text = Tex('This is a shear transformation.').move_to(3*UP + 1.5*RIGHT)
        self.play(Write(text))
        formula2.move_to(text).shift(1.5*DOWN)
        self.play(Write(formula2))
        self.wait(0.5)
        self.play(Unwrite(text, reverse=False, run_time=0.3))
        text = Tex(r'It maps the oblique ellipse\\into an orthogonal one.'
                   ).move_to(text)
        self.play(Write(text))
        self.wait(0.5)
        self.play(Unwrite(text, reverse=False, run_time=0.3))

        formula3.move_to(formula2)
        map23 = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7,
                 8:9, 9:10, 10:11, 11:12}
        self.play(*get_tex_map_anims((formula2,), (map23,), formula3))
        self.wait(0.5)

        p10 = trammel.chord.dot_a.get_center()
        p20 = trammel.chord.dot_c.get_center()
        v0 = p10 - p20
        vp2 = center - p20
        angle = PI - trammel.angle
        line_a = trammel.chord.line_a
        dot_a = trammel.chord.dot_a
        vt2 = ValueTracker(0)
        previous_vt2 = ValueTracker(0)
        def update_a():
            t = vt2.get_value()
            dt = t - previous_vt2.get_value()
            previous_vt2.set_value(t)
            p2 = p20 + t*vp2
            p1 = p2 + rotate_vector(v0, -t*angle)
            line_a.shift(dt*vp2)
            line_a.rotate(-dt*angle, about_point=p2)
            dot_a.move_to(p1)
            text_a.move_to(get_line_pos(p1, -t*angle, A, -0.5*r))

        updater_a = lambda _: update_a()
        line_a_group = VGroup(line_a, dot_a, text_a)
        line_a_group.add_updater(updater_a)
        trammel.chord.dot_c.set_z_index(5)
        trammel.chord.dot_b.set_z_index(4)
        self.add(line_a_group)
        self.play(vt2.animate.set_value(1), formula3.animate.shift(UP))
        line_a_group.remove_updater(updater_a)

        formula1.move_to(dot_a.get_center() + 1.2*RIGHT + 0.6*UP)
        self.play(Write(formula1))

        formula4.move_to(formula3)
        map13 = {1:[1, 6], 3:[3, 9, 11]}
        mapped = [1, 3, 6, 9, 11]
        map34 = {k:k for k in range(len(formula3)) if k not in mapped}
        self.play(*get_tex_map_anims(
            (formula1, formula3), (map13, map34), formula4, (True, False)))
        self.wait()

        map45 = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 10:7, 11:8, 12:9}
        formula5.move_to(formula4)
        self.play(*get_tex_map_anims((formula4,), (map45,), formula5))
        self.wait()

        angle = trammel.angle
        pos = center + B*np.cos(angle)*LEFT + A*np.sin(angle)*UP
        formula6.move_to(pos + 1.6*RIGHT + 0.3*UP)
        dot_p = dot_a.copy()
        dot_p.add_updater(vf_updater)
        self.add(dot_p)
        map56 = {5:0, 6:1, 7:2, 8:3, 9:4}
        self.play(
            *get_tex_map_anims((formula5,), (map56,), formula6, (True,)),
            run_time=nudge_run_time
        )
        dot_p.remove_updater(vf_updater)
        self.play(Unwrite(formula5, reverse=False, run_time=0.3))

        text = Tex(r'This is the standard parametric\\'
                   'representation of an ellipse.').move_to(text)
        self.play(Write(text))
        self.wait(0.5)
        self.play(Unwrite(text, reverse=False, run_time=0.3))
        text = Tex(r'The area of the sector is $\frac{1}{2}\alpha$'
                   ).move_to(text)
        self.play(Write(text))
        self.wait(0.5)
        formula = MathTex(r'\text{Area}', r' = \frac{1}{2}\alpha')
        formula[0].set_color(PURPLE_A)
        formula.move_to(trammel.chord.dot_c.get_center() + RIGHT + 2.5*UP)
        self.play(Write(formula))
        self.wait(0.5)
        self.play(Unwrite(text, reverse=False, run_time=0.3))
        text = Tex(r"By Cavalieri's principle,\\"
                   'the blue and the purple areas are equal.').move_to(text)
        self.play(Write(text))
        self.wait()
        formula_vt = ValueTracker(0)
        def update_formula(dt):
            t = formula_vt.get_value()
            color = interpolate_color(PURPLE_A, ellipse.get_stroke_color(), t)
            formula[0].set_color(color)
            formula.shift(dt*LEFT)

        formula_updater = lambda _, dt: update_formula(dt)
        formula.add_updater(formula_updater)
        self.play(formula_vt.animate.set_value(1), run_time=1)
        formula.remove_updater(formula_updater)
        self.wait(1.5)
        self.play(
            Unwrite(text, reverse=False, run_time=0.3),
            Unwrite(formula1, reverse=False, run_time=0.3),
            Unwrite(formula6, reverse=False, run_time=0.3),
            Unwrite(formula, reverse=False, run_time=0.3),
            FadeOut(text_a), FadeOut(text_b),
            FadeOut(trammel_group), FadeOut(dot_p),
            FadeOut(angle_alpha), FadeOut(text_alpha),
            FadeOut(original_ellipse),
            FadeOut(sector), FadeOut(original_sector),
            run_time=0.3
        )

        self.wait(0.1)


class Scene06(Scene):
    def construct(self):
        self.add(CC0)

        A, B = 1.5, 0.5
        ps = Ring([
            2*DOWN,
            DOWN + 3*RIGHT,
            1.5*UP + 3*RIGHT,
            2*UP,
            3*LEFT + UP,
            2*LEFT + DOWN
        ])
        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, ps, a=A, b=B, speed=4)

        self.add(holditch.strip)
        self.play(FadeIn(holditch.curve))
        self.play(FadeIn(holditch.chord.line_a),
                  FadeIn(holditch.chord.line_b),
                  FadeIn(holditch.chord.dot_a),
                  FadeIn(holditch.chord.dot_b),
                  FadeIn(holditch.chord.dot_c))
        self.add(holditch.group)

        def get_angle_pos(c, p1, p2, r):
            v = np.array([1, 1j, 0])
            angle1 = np.angle((p1 - c)@v)
            angle2 = np.angle((p2 - c)@v)
            if angle2 < angle1:
                angle2 += TAU
            return c + r*rotate_vector(RIGHT, (angle1 + angle2)/2)

        lines = []
        angles = []
        angle_labels = []
        area_texs = []
        arrows = []
        anims = []
        anims2 = []
        r = 0.3
        for p1, p2, p3 in ps.triples():
            v12 = normalize(p2 - p1)
            u12 = rotate_vector(v12, PI/2)
            p = p2 + v12
            line = DashedLine(p2, p)
            lines.append(line)
            angle = Angle(lines[-1], Line(p2, p3), radius=r)
            angles.append(angle)
            pos = get_angle_pos(p2, p, p3, 2*r)
            angle_label = MathTex(fr'\alpha_{{{len(lines)}}}',
                                  font_size=round(0.7*DEFAULT_FONT_SIZE))
            angle_label.move_to(pos)
            angle_labels.append(angle_label)
            anims.append(AnimationGroup(Create(line), Create(angle),
                                        Write(angle_label)))
            pos1 = get_angle_pos(p2, p3, p1, r*2*angle.get_value()/PI)
            pos2 = p2 - 0.5*v12 - u12
            area_tex = MathTex(
                r'\frac{1}{2}', fr'\alpha_{{{len(lines)}}}',
                color=holditch.locus.get_stroke_color(),
                font_size=round(0.7*DEFAULT_FONT_SIZE))
            area_tex.move_to(pos2)
            area_texs.append(area_tex)
            p = area_tex.get_center()
            arrow = CurvedArrow(pos1, pos1 + 0.6*(p - pos1), color=GRAY,
                                tip_length=0.2, angle=PI/4, z_index=3)
            arrows.append(arrow)
            anims2.append(AnimationGroup(Write(area_tex),  FadeIn(arrow)))

        dt = 4.8
        self.play(
            vt.animate.set_value(time := time + dt),
            Succession(*anims, lag_ratio=1.0),
            run_time=dt, rate_func=linear
        )
        self.play(
            vt.animate.set_value(time := time + dt),
            Succession(*anims2, lag_ratio=1.0),
            run_time=dt, rate_func=linear
        )

        holditch.locus.set_stroke(holditch.locus.get_stroke_color(), opacity=0)
        for k, poly in enumerate(holditch.strip):
            if k in [1, 2, 4]:
                ps = make_convex(poly)
            else:
                ps = make_convex(poly, 10*DEGREES)
            poly.become(Polygon(
                *ps, fill_color=poly.get_fill_color(),
                stroke_color=holditch.locus.get_stroke_color(),
                fill_opacity=1, stroke_opacity=1, z_index=-1
            ))

        holditch.chord.opacity = 0
        chord = ChordVGroup(holditch.chord.dot_a.get_center(),
                            holditch.chord.dot_b.get_center(),
                            holditch.chord.dot_c.get_center())
        self.add(chord.line_a, chord.line_b,
                 chord.dot_a, chord.dot_b, chord.dot_c)
        self.add(chord)
        start_time = time
        dt = 1
        def update_chord():
            chord.become(ChordVGroup(
                holditch.chord.dot_a.get_center(),
                holditch.chord.dot_b.get_center(),
                holditch.chord.dot_c.get_center(),
                opacity=1 - (vt.get_value() - start_time)/dt))
        chord_updater = lambda _: update_chord()
        chord.add_updater(chord_updater)
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        self.remove(holditch.chord, chord, holditch.locus)

        plus = [MathTex('+') for _ in range(5)]
        shift = 1.5*DOWN
        self.play(
            *[FadeOut(arrow) for arrow in arrows],
            holditch.curve.animate.shift(shift),
            *[line.animate.shift(shift) for line in lines],
            *[angle.animate.shift(shift) for angle in angles],
            *[label.animate.shift(shift) for label in angle_labels],
            *[poly.animate.move_to(1.5*UP + 5*LEFT + 2*((k - 1) % 6)*RIGHT)
              for k, poly in enumerate(holditch.strip)],
            *[tex.animate.move_to(3*UP + 5*LEFT + 2*k*RIGHT)
              for k, tex in enumerate(area_texs)],
            *[FadeIn(tex.move_to(3*UP + 4*LEFT + 2*k*RIGHT))
              for k, tex in enumerate(plus)],
            run_time=2
        )
        self.wait()

        formula1 = MathTex(r'\frac{1}{2}', r'\cdot', r'2\pi').move_to(3*UP)
        formula2 = MathTex(r'\frac{1}{2}\cdot 2', r'\pi').move_to(formula1)
        formula3 = MathTex(r'\text{Area}', '=', r'\pi').move_to(formula1)
        formula3[0].set_color(holditch.locus.get_stroke_color())
        maps1 = 6*[{0:0, 1:2}]
        map23 = {1:2}
        self.play(*get_tex_map_anims(area_texs, maps1, formula1),
                  *[FadeOut(tex) for tex in plus])
        self.wait()
        self.remove(*formula1)
        self.play(*get_tex_map_anims((formula2,), (map23,), formula3))
        self.wait()

        self.play(
            FadeOut(formula3),
            FadeOut(holditch.curve),
            FadeOut(holditch.strip),
            *[FadeOut(line) for line in lines],
            *[FadeOut(angle) for angle in angles],
            *[FadeOut(label) for label in angle_labels],
            run_time=0.3
        )
        self.wait(0.1)


class Scene07(Scene):
    def construct(self):
        self.add(CC0)

        class PolygonalHolditch(Holditch):
            fill_color = COLOR_DARK_BLUE  # '#3400ff' at opacity 0.25
            stroke_color = COLOR_BLUE
            opacity = 1

            def __init__(
                self, vt, ps, a, b, speed=4, build_path=False, starting_index=0
            ):
                super().__init__(
                    vt, ps, a=a, b=b, speed=speed, build_path=build_path,
                    starting_index=starting_index
                )
                self.ps = self.curve_ps
                self.L = self.scale*(self.a + self.b)

            def get_ellipse_params(self, alpha):
                a, b = self.scale*self.a, self.scale*self.b
                L = a + b
                s, c = np.sin(alpha), np.cos(alpha)
                if s < 0:
                    a, b = b, a
                k = c/s
                A = 1/b**2
                B = (k*L/(a*b))**2
                C = 1/a**2
                D = (A + B + C)/2
                E = (A - B - C)/2
                F = np.sqrt(A*B + E**2)
                X = D - F
                Y = D + F
                Z = E/F
                a2 = np.sqrt(1/X)
                b2 = np.sqrt(1/Y)
                beta = np.arcsin(np.sqrt((1 + Z)/2))
                if k < 0:
                    beta = np.pi - beta
                return a2, b2, beta

            def get_full_ellipse(self, k1, k2):
                p11, p12 = self.ps[k1: k1 + 2]
                p21, p22 = self.ps[k2: k2 + 2]
                n11 = norm(p21 - p11)
                n12 = norm(p21 - p12)
                n21 = norm(p22 - p11)
                n22 = norm(p22 - p12)
                if min(n11, n12, n21, n22) >= self.L:
                    return None
                v1 = normalize(p12 - p11)
                v2 = normalize(p22 - p21)

                normal = np.cross(v2, np.cross(v1, v2))
                denom = v1@normal
                if abs(denom) < 0.01:
                    return None
                center = p11 + v1*((p21 - p11)@normal)/denom

                v = np.array([1, 1j, 0])
                angle1 = np.angle(v1@v)
                angle2 = np.angle(v2@v)
                alpha = angle2 - angle1
                a2, b2, beta = self.get_ellipse_params(alpha)
                ellipse = Ellipse(
                    2*a2, 2*b2,
                    fill_color=self.fill_color,
                    fill_opacity=self.opacity,
                    stroke_color=self.stroke_color,
                    stroke_opacity=1,
                ).set_z_index(0)
                ellipse.rotate(angle1 + beta).move_to(center)
                return ellipse

            def get_partial_ellipse(self, k1, k2):
                full_ellipse = self.get_full_ellipse(k1, k2)
                if full_ellipse is None:
                    return None
                partial_ellipse = Intersection(
                    self.curve, full_ellipse,
                    fill_color=self.fill_color,
                    fill_opacity=self.opacity,
                    stroke_color=self.stroke_color,
                    stroke_opacity=1
                ).set_z_index(-1)
                return partial_ellipse

            def get_ellipses(self, full=True):
                ellipses = []
                if full:
                    get_ellipse = self.get_full_ellipse
                else:
                    get_ellipse = self.get_partial_ellipse
                for k1, k2 in itertools.combinations(range(len(self.ps)), 2):
                    ellipse = get_ellipse(k1, k2)
                    if ellipse is not None:
                        ellipses.append(ellipse)
                return ellipses

        A, B = 3, 2
        ps = [
            5.5*LEFT + 2.5*DOWN,
            RIGHT + 2.5*DOWN,
            2.5*RIGHT + 3.5*UP,
            3.5*RIGHT + 7.5*UP,
            5.5*LEFT + 7.5*UP,
        ]
        time = 0
        vt = ValueTracker(time)
        holditch = PolygonalHolditch(vt, ps, a=A, b=B, speed=4.5,
                                     build_path=True)
        curve1 = PolygonalChain(*ps[:3], color=WHITE, z_index=2)
        self.play(
            FadeIn(curve1),
            *[FadeIn(mob) for mob in holditch.chord],
            run_time=0.5
        )
        self.add(holditch.group)

        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(1.5, 2)
        holditch.group.add_updater(chord_opacity_updater)

        ellipse1_partial = holditch.get_partial_ellipse(0, 1)
        ellipse1_full = holditch.get_full_ellipse(0, 1)
        set_opacity(ellipse1_partial, 0, 0)
        set_opacity(ellipse1_full, 0, 0)
        self.add(ellipse1_partial)
        def update_ellipse_opacity(mob, t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                opacity = (t - t1)/(t2 - t1)
                set_opacity(mob, fill=opacity)
        ellipse1_partial_updater = lambda mob: update_ellipse_opacity(
            mob, 1.5, 2)
        ellipse1_partial.add_updater(ellipse1_partial_updater)

        dt = 2
        self.play(vt.animate.set_value(time := time + dt),
                  run_time=dt, rate_func=linear)
        self.remove(holditch.group)
        set_opacity(ellipse1_partial, 1, 1)

        dash = DashedLine((1 + 3/8)*LEFT + 4*DOWN, (2 + 3/4)*RIGHT + 0.5*UP)
        self.play(Create(dash))
        triangle = Polygon(
            2.5*DOWN, RIGHT + 2.5*DOWN, (1 + 3/8)*RIGHT + DOWN,
            stroke_color=COLOR_RED, stroke_opacity=1,
            fill_color=COLOR_DARK_RED, fill_opacity=1,
            z_index=0
        )

        ps = [
            5.5*LEFT + 2.5*DOWN,
            2.5*DOWN,
            (1 + 3/8)*RIGHT + DOWN,
            2.5*RIGHT + 3.5*UP,
            3.5*RIGHT + 7.5*UP,
            5.5*LEFT + 7.5*UP,
        ]
        holditch = PolygonalHolditch(vt, ps, a=A, b=B, speed=4.5,
                                     build_path=True)
        curve2 = PolygonalChain(*ps[:4], color=WHITE, z_index=2)
        self.play(FadeOut(curve1), FadeIn(curve2), FadeIn(triangle),
                  run_time=0.5)
        self.play(FadeOut(dash), run_time=0.5)
        self.play(*[FadeIn(mob) for mob in holditch.chord], run_time=0.5)
        self.add(holditch.group)
        chord_opacity_updater = lambda _: update_chord_opacity(3.5, 4)
        holditch.group.add_updater(chord_opacity_updater)

        ellipse2_partial = holditch.get_partial_ellipse(0, 1)
        ellipse2_full = holditch.get_full_ellipse(0, 1).set_z_index(-2)
        ellipse3_partial = holditch.get_partial_ellipse(1, 2)
        ellipse2_partial = Difference(
            ellipse2_partial, ellipse1_partial,
            stroke_color=ellipse2_partial.get_stroke_color(), stroke_opacity=0,
            fill_color=COLOR_DARK_GREEN, fill_opacity=0,
            z_index=-2
        )
        ellipse3_partial = Difference(
            ellipse3_partial, ellipse1_partial,
            stroke_color=ellipse3_partial.get_stroke_color(), stroke_opacity=0,
            fill_color=COLOR_DARK_GREEN, fill_opacity=0,
            z_index=-2
        )
        set_opacity(ellipse2_full, 0, 0)
        self.add(ellipse2_partial)
        self.add(ellipse3_partial)
        ellipse23_opacity_updater = lambda mob: update_ellipse_opacity(
            mob, 3.5, 4)
        ellipse2_partial.add_updater(ellipse23_opacity_updater)
        ellipse3_partial.add_updater(ellipse23_opacity_updater)

        self.play(vt.animate.set_value(time := time + dt),
                  run_time=dt, rate_func=linear)
        holditch.group.remove_updater(chord_opacity_updater)
        ellipse2_partial.remove_updater(ellipse23_opacity_updater)
        ellipse3_partial.remove_updater(ellipse23_opacity_updater)
        self.remove(holditch.group)
        set_opacity(ellipse2_partial, 1, 1)
        set_opacity(ellipse3_partial, 1, 1)
        self.wait(0.5)

        tex1 = MathTex(
            r'\text{Area}_{new}', r'\quad=\quad', r'\text{Area}_{old}', '+',
            r'\text{Area}_a', '+', r'\text{Area}_b', '-', r'\text{Area}_c'
        )
        tex1[2].set_color(COLOR_BLUE)
        tex1[4].set_color(COLOR_GREEN)
        tex1[6].set_color(COLOR_GREEN)
        tex1[8].set_color(COLOR_RED)
        ellipse1_redux = Difference(
            ellipse1_partial, triangle,
            stroke_color=ellipse1_partial.get_stroke_color(), stroke_opacity=1,
            fill_color=ellipse1_partial.get_fill_color(), fill_opacity=1
        )
        area_new = VGroup(ellipse2_partial.copy(), ellipse3_partial.copy(),
                          ellipse1_redux).set_z_index(3)
        area_old = ellipse1_partial.copy().set_z_index(4)
        area_a = ellipse2_partial.copy().set_z_index(5)
        area_b = ellipse3_partial.copy().set_z_index(5)
        area_c = triangle.copy().set_z_index(5)
        group = VGroup(
            ellipse2_full, ellipse1_full,
            ellipse2_partial, ellipse3_partial, ellipse1_partial,
            triangle, curve2
        )
        tex1.move_to(3*UP)
        self.play(
            Write(tex1),
            area_new.animate.move_to(tex1[0].get_center() + 3*DOWN),
            area_old.animate.move_to(tex1[2].get_center() + 2*DOWN),
            area_a.animate.move_to(tex1[4].get_center() + 1.5*DOWN),
            area_b.animate.move_to(tex1[6].get_center() + 1.5*DOWN),
            area_c.animate.move_to(tex1[8].get_center() + 1.5*DOWN),
            group.animate.shift(4*RIGHT + 1*DOWN),
            run_time=2
        )
        self.wait(2.5)
        self.play(Unwrite(tex1), FadeOut(area_new), FadeOut(area_old),
                  FadeOut(area_a), FadeOut(area_b), FadeOut(area_c),
                  run_time=0.5)
        self.play(group.animate.shift(4*LEFT + 1*UP))

        tex_a = MathTex(r'\text{Area}_a', r'=\,?')
        tex_a[0].set_color(COLOR_GREEN)
        tex_a.move_to(ellipse2_partial.get_center() + LEFT + UP)
        tex_b = MathTex(r'\text{Area}_b', r'=\,?')
        tex_b[0].set_color(COLOR_GREEN)
        tex_b.move_to(ellipse3_partial.get_center() + 2.5*LEFT)
        tex_c = MathTex(r'\text{Area}_c', r'= \frac{1}{2}cH')
        tex_c.move_to(triangle.get_center() + 3*RIGHT + 1.5*UP)
        tex_c[0].set_color(COLOR_RED)
        dot_ra = Dot((1 + 3/8)*RIGHT + 2.5*DOWN,
                     stroke_opacity=0, fill_opacity=0)
        dot_c1 = Dot(2.5*DOWN, stroke_opacity=0, fill_opacity=0)
        dot_c2 = Dot(RIGHT + 2.5*DOWN, stroke_opacity=0, fill_opacity=0)
        dot_H = Dot(dot_ra.get_center() + 1.5*UP,
                    stroke_opacity=0, fill_opacity=0)
        corner_pos = (1 + 3/8)*RIGHT + 2.5*DOWN
        base_c = Line(dot_c1.get_center(), dot_c2.get_center(),
                      stroke_opacity=0, fill_opacity=0)
        dash_x = DashedLine(dot_c1.get_center(), dot_ra.get_center() + RIGHT,
                            z_index=3)
        dash_H = DashedLine(dot_ra.get_center(), dot_H.get_center(), z_index=3)
        right_angle = RightAngle(dash_x, dash_H, 0.15,
                                 quadrant=(-1, 1), z_index=3)
        brace_c = BraceLabel(base_c, 'c', z_index=3)
        brace_H = BraceLabel(dash_H, 'H', RIGHT, z_index=3)
        p = ellipse2_partial.get_center()
        arrow_a = CurvedArrow(p, p + 0.6*(tex_a.get_center() - p), color=GRAY,
                              radius=-1, tip_length=0.2, angle=PI/4)
        p = ellipse3_partial.get_center()
        arrow_b = CurvedArrow(p, p + 0.5*(tex_b.get_center() - p), color=GRAY,
                              tip_length=0.2, angle=PI/4)
        p = (RIGHT + 2.5*DOWN + corner_pos + 1.5*UP)/2
        arrow_c = CurvedArrow(p, p + 0.6*(tex_c.get_center() - p), color=GRAY,
                              tip_length=0.2, angle=PI/4, z_index=4)
        self.play(Create(dash_x), Create(dash_H), Create(right_angle),
                  run_time=0.5)
        self.play(FadeIn(brace_c), FadeIn(brace_H), run_time=0.5)
        self.play(Write(tex_c), FadeIn(arrow_c), run_time=1)
        self.play(Write(tex_a), Write(tex_b),
                  FadeIn(arrow_a), FadeIn(arrow_b),
                  run_time=1)
        self.wait(2)

        ps = [
            3.5*LEFT + 2.5*DOWN,
            2.5*DOWN,
            (1 + 3/8)*RIGHT + DOWN,
            2*RIGHT + 1.5*UP,
        ]
        curve3 = PolygonalChain(*ps, color=WHITE, z_index=2)
        self.play(
            Unwrite(tex_a), Unwrite(tex_b), Unwrite(tex_c),
            FadeOut(arrow_a), FadeOut(arrow_b), FadeOut(arrow_c),
            FadeOut(ellipse3_partial), Transform(curve2, curve3),
            run_time=0.5
        )
        set_opacity(ellipse3_partial, 0, 0)
        self.remove(curve3)

        ellipse1_compl = Difference(
            ellipse1_full, ellipse1_partial,
            stroke_color=ellipse1_full.get_stroke_color(), stroke_opacity=1,
            fill_color=ellipse1_full.get_fill_color(), fill_opacity=0,
            z_index=-1
        )
        ellipse2_compl = Difference(
            ellipse2_full, ellipse2_partial,
            stroke_color=ellipse2_full.get_stroke_color(), stroke_opacity=1,
            fill_color=ellipse2_full.get_fill_color(), fill_opacity=0,
            z_index=-1
        )
        self.play(Create(ellipse1_compl), Create(ellipse2_compl))
        self.wait(0.5)

        def update_brace(mob, anchor, text, direction):
            mob.become(BraceLabel(anchor, text, direction,
                                  z_index=mob.get_z_index()))
        brace_c_updater = lambda mob: update_brace(mob, base_c, 'c', DOWN)
        brace_H_updater = lambda mob: update_brace(mob, dash_H, 'H', RIGHT)
        brace_c.add_updater(brace_c_updater)
        brace_H.add_updater(brace_H_updater)
        group2 = VGroup(*group, base_c, dash_x, dash_H, right_angle,
                        ellipse1_compl, ellipse2_compl,
                        dot_ra, dot_c1, dot_c2, dot_H)
        scale = 2
        self.play(group2.animate.scale(scale).shift(UP))
        A *= scale
        B *= scale
        brace_c.remove_updater(brace_c_updater)
        brace_H.remove_updater(brace_H_updater)

        pb = dot_c2.get_center()
        chord = ChordVGroup(pb + (A + B)*LEFT, pb, pb + B*LEFT)
        self.play(*[FadeIn(mob) for mob in chord], FadeOut(brace_c))
        self.add(chord)
        chord_dot_c0 = chord.dot_c.copy()
        def update_brace_c(mob, dot1, dot2):
            line = Line(dot1.get_center(), dot2.get_center())
            mob.become(BraceLabel(line, 'c', z_index=mob.get_z_index()))
        brace_c1_updater = lambda mob: update_brace_c(mob, chord.dot_c,
                                                      chord_dot_c0)
        brace_c2_updater = lambda mob: update_brace_c(mob, chord.dot_b,
                                                      dot_c2)
        brace_c1 = brace_c.copy()
        brace_c2 = brace_c.copy()
        brace_c1_updater(brace_c1)
        brace_c2_updater(brace_c2)
        brace_c1.add_updater(brace_c1_updater)
        brace_c2.add_updater(brace_c2_updater)
        self.add(brace_c1, brace_c2)
        self.remove(brace_c)
        c = norm(pb - dot_c1.get_center())
        set_opacity(ellipse1_full, 1, 0)
        self.play(
            chord.animate.shift(c*LEFT),
            ellipse1_full.animate.shift(c*LEFT)
        )
        brace_c1.remove_updater(brace_c1_updater)
        brace_c2.remove_updater(brace_c2_updater)
        self.wait(0.5)

        pc1 = chord.dot_c.get_center()
        pc2 = chord_dot_c0.get_center()
        pH = dot_H.get_center()
        pra = dot_ra.get_center()
        H = norm(pH - pra)
        h = H*A/(A + B)
        W = np.sqrt((A + B)**2 - H**2)
        w = W*B/(A + B)
        pd1 = pc2
        pd2 = dot_ra.get_center() + w*LEFT
        ph = pd2 + h*UP
        cut_red = Intersection(
            ellipse1_partial, Polygon(pc1, pd2, ph),
            fill_color=COLOR_DARK_RED, fill_opacity=1,
            stroke_color=COLOR_RED, stroke_opacity=1,
            z_index=1
        )
        cut_green = Union(
            ellipse2_partial, cut_red,
            fill_color=COLOR_DARK_GREEN, fill_opacity=1,
            stroke_color=COLOR_BLUE, stroke_opacity=1,
            z_index=0
        )
        right_angle2 = RightAngle(chord.line_b, Line(pd2, ph),
                                  0.3, quadrant=(1, 1), z_index=2)
        brace_d = BraceLabel(Line(pd1, pd2), 'd', z_index=4)
        brace_h = BraceLabel(cut_red, 'h', RIGHT, z_index=4)
        dot_h = Dot(ph, z_index=2)
        dot_h0 = dot_h.copy()
        self.play(FadeIn(cut_red), run_time=0.5)
        self.play(
            FadeIn(cut_green), FadeIn(dot_h0), Create(right_angle2),
            FadeIn(brace_d), FadeIn(brace_h),
            run_time=1
        )
        self.remove(ellipse2_partial)
        self.wait(0.5)

        self.play(
            cut_red.animate.shift(3*LEFT + 3*UP),
            run_time=1
        )
        self.wait(0.5)

        dash_h = DashedLine(pd2, ph, z_index=0)
        dash_c = DashedLine(ph, ph, z_index=0)
        def update_dash_c():
            dash_c.become(DashedLine(dot_h.get_center(), ph, z_index=0))
        dash_c_updater = lambda _: update_dash_c()
        dash_c.add_updater(dash_c_updater)

        shear_scale = 0.07
        def shear(pos):
            return shear_scale*LEFT*(pos - pd2)[1]*c/h
        w = 2 + config['frame_width']//2
        vector_field = ArrowVectorField(shear, length_func=linear,
                                        x_range=[-w, w],
                                        colors=[MAROON_A, MAROON_E])
        nudge_run_time = 1.5
        speed = 1/(nudge_run_time*shear_scale)
        vf_pointwise_updater = vector_field.get_nudge_updater(
            speed=speed, pointwise=True)
        vf_updater = vector_field.get_nudge_updater(
            speed=speed, pointwise=False)
        def vt_self_updater(mob, dt):
            for arrow in mob:
                vector_field.nudge(arrow, 0.2*speed*dt)
                x = arrow.get_center()[0]
                if x > w:
                    arrow.shift(2*w*LEFT)
                if x < -w:
                    arrow.shift(2*w*RIGHT)
        vector_field.add_updater(vt_self_updater)
        self.play(FadeIn(vector_field), run_time=0.5)
        self.wait(0.5)
        self.add(dash_h, dash_c)
        self.add(dot_h)
        dot_h.add_updater(vf_updater)
        cut_green.add_updater(vf_pointwise_updater)
        ellipse2_compl.add_updater(vf_pointwise_updater)
        self.wait(nudge_run_time)
        self.remove(ellipse2_compl)
        cut_green.remove_updater(vf_pointwise_updater)
        ellipse2_compl.remove_updater(vf_pointwise_updater)
        dot_h.remove_updater(vf_updater)
        dash_c.remove_updater(dash_c_updater)
        dot_h.move_to(dot_h0.get_center() + c*LEFT)
        cut_green.become(Intersection(
            ellipse1_full, Polygon(pc1, pd2, ph + c*LEFT, pc1 + h*UP),
            fill_color=COLOR_DARK_GREEN, fill_opacity=1,
            stroke_color=COLOR_BLUE, stroke_opacity=1,
            z_index=0
        ))
        self.play(FadeOut(vector_field), run_time=0.5)
        vector_field.remove_updater(vt_self_updater)

        self.wait()
        self.play(cut_red.animate.shift((3 - c)*RIGHT + 3*DOWN), run_time=1)
        cut_green.become(Polygon(
            dot_h.get_center(), dot_h.get_center() + h*DOWN, pd2,
            fill_color=cut_green.get_fill_color(), fill_opacity=1,
            stroke_color=cut_green.get_stroke_color(), stroke_opacity=1,
            z_index=cut_green.get_z_index()
        ))
        right_angle3 = RightAngle(chord.line_b, dash_h.copy().shift(c*LEFT),
                                  0.3, quadrant=(1, 1), z_index=0)
        self.add(right_angle3)
        self.play(FadeOut(cut_red), run_time=0.5)
        self.wait(0.5)

        arc_d = ArcBetweenPoints(
            brace_d.get_center(), brace_d.get_center() + c*LEFT,
            radius=-1, angle=60*DEGREES
        )
        brace_h.set_z_index(4)
        self.play(
            MoveAlongPath(brace_d, arc_d),
            brace_c1.animate.shift(norm(pd2 - pd1)*RIGHT),
            brace_h.animate.shift(c*LEFT),
            run_time=1.5
        )
        self.wait(0.5)

        tex1 = MathTex(r'\text{Area}_a', '=', r'\frac{1}{2}c', 'h')
        tex1.move_to(4.5*LEFT + 2.5*UP)
        tex1[0].set_color(COLOR_GREEN)
        self.play(Write(tex1))
        self.wait(1.5)

        run_time=1
        vt = ValueTracker(0)
        pa0 = chord.dot_a.get_center()
        pb0 = chord.dot_b.get_center()
        def update_chord():
            t = vt.get_value()
            pb = pb0 + t*(pH - pb0)
            pa = get_segment_intersections(pb[:2], A + B, pa0[:2], pb0[:2])[0]
            pa = pa[0]*RIGHT + pa[1]*UP
            pc = pa + (pb - pa)*A/(A + B)
            chord.become(ChordVGroup(pa, pb, pc))
        chord_updater = lambda _: update_chord()
        chord.add_updater(chord_updater)
        self.play(vt.animate.set_value(run_time),
                  brace_h.animate.shift(c*RIGHT),
                  run_time=run_time, rate_func=linear)
        chord.remove_updater(chord_updater)
        self.wait(0.5)

        pa = chord.dot_a.get_center()
        pb = chord.dot_b.get_center()
        normal = rotate(pb - pa, PI/2)
        brace_a = BraceLabel(chord.line_a, 'a', normal)
        brace_b = BraceLabel(chord.line_b, 'b', normal)
        self.play(FadeOut(brace_d), FadeIn(brace_a), FadeIn(brace_b))
        self.wait(0.5)

        tex2 = MathTex('{h', r'\over', 'a}', '=',
                       '{H', r'\over', 'a', '+', 'b}')
        tex3 = MathTex('h', '=', '{a', r'\over', 'a', '+', 'b}',
                       r'\cdot', 'H')
        tex4 = MathTex('h', '=', r'{a \over a + b}', r'\cdot', 'H')
        tex5 = MathTex(r'\text{Area}_a', '=', r'\frac{1}{2}c', r'\cdot',
                       r'{a \over a + b}', r'\cdot', 'H')
        tex5[0].set_color(COLOR_GREEN)
        tex6 = MathTex(r'\text{Area}_a', '=', r'{a \over a + b}', r'\cdot',
                       r'\frac{1}{2}c', 'H')
        tex6[0].set_color(COLOR_GREEN)
        tex7 = MathTex(r'\text{Area}_a', '=', r'{a \over a + b}', r'\cdot',
                       r'\text{Area}_c')
        tex7[0].set_color(COLOR_GREEN)
        tex7[4].set_color(COLOR_RED)

        tex2.next_to(tex1, DOWN, MED_LARGE_BUFF)
        tex3.move_to(tex2).align_to(tex2, LEFT)
        tex4.move_to(tex3)
        tex5.move_to(tex1).shift(0.5*RIGHT)
        tex6.move_to(tex5)
        tex7.move_to(tex6)

        map_h2 = {1:0}
        map_a2 = {1:2}
        self.play(indicate_group(VGroup(brace_h, brace_a)))
        self.play(*get_tex_map_anims(
            (brace_h, brace_a), (map_h2, map_a2), tex2[:4], (True, True)))

        map_H2 = {1:0}
        map_a2 = {1:2}
        map_b2 = {1:4}
        self.play(indicate_group(VGroup(brace_H, brace_a, brace_b)))
        self.play(*get_tex_map_anims(
            (brace_H, brace_a, brace_b),
            (map_H2, map_a2, map_b2), tex2[4:], 3*(True,)
        ))
        self.wait()

        map_23 = {0:0, 2:2, 3:1, 4:8, 5:3, 6:4, 7:5, 8:6}
        self.play(*get_tex_map_anims((tex2,), (map_23,), tex3))
        self.wait(0.5)

        self.remove(tex3)
        self.add(tex4)
        map_15 = {0:0, 1:1, 2:2}
        map_45 = {2:4, 3:5, 4:6}
        self.play(*get_tex_map_anims(
            (tex1, tex4), (map_15, map_45), tex5, (False, True)
        ))
        self.wait(0.5)

        map_56 = {0:0, 1:1, 2:4, 4:2, 5:3, 6:5}
        self.play(*get_tex_map_anims((tex5,), (map_56,), tex6))
        self.wait(0.5)

        map_67 = {0:0, 1:1, 2:2, 3:3, 4:4, 5:4}
        self.play(*get_tex_map_anims((tex6,), (map_67,), tex7))
        self.wait()

        self.remove(*tex3)
        self.play(
            Unwrite(tex4),
            *[FadeOut(mob) for mob in chord],
            *[FadeOut(mob) for mob in [
                brace_a, brace_b, brace_h, brace_H, brace_c1, brace_c2,
                curve2, dash_x, dash_H, dash_h, dash_c, dot_h0, dot_h,
                right_angle, right_angle2, right_angle3,
                cut_green, triangle,
                ellipse1_full, ellipse1_compl, ellipse1_partial
            ]],
            run_time=0.5
        )
        self.wait(0.1)


class Scene08(Scene):
    def construct(self):
        self.add(CC0)

        class PolygonalHolditch(Holditch):
            fill_color = COLOR_DARK_BLUE  # '#3400ff' at opacity 0.25
            stroke_color = COLOR_BLUE
            opacity = 1

            def __init__(
                self, vt, ps, a, b, speed=4, build_path=False,
                starting_index=0, stop_index=None
            ):
                super().__init__(
                    vt, ps, a=a, b=b, speed=speed, build_path=build_path,
                    starting_index=starting_index
                )
                self.ps = self.curve_ps
                self.L = self.scale*(self.a + self.b)
                self.stop_at_index = stop_index
                if stop_index is None:
                    self.end_chord = None
                else:
                    self.end_chord = self._locus.get_chord_by_index(stop_index)
                self.stopped = False

            def update(self, group: VGroup):
                if self.stopped:
                    return
                if self.stop_at_index is None:
                    super().update(group)
                    return
                dt = self.vt.get_value() - self.t
                dist = dt*self.speed
                prev_chord = self._locus.chord
                super().update(group)
                new_chord = self._locus.chord
                dist_end = self._curve.get_dist(new_chord.cpa.d,
                                                self.end_chord.cpa.d)
                if (
                    abs(dist_end) < (1 + 0.01)*dist  # plus small tolerance
                    and prev_chord.is_ahead(self.end_chord)
                    and not new_chord.is_ahead(self.end_chord)
                ):
                    self._locus.chord = self.end_chord
                    self.stopped = True
                    if self.build_path:
                        self._locus.ps[-1] = self.end_chord.pc
                    self.chord.become(ChordVGroup(
                        *self._get_chord_points(),
                        lines_visible=self.chord.lines_visible,
                        opacity=self.chord.opacity))

            def get_ellipse_params(self, alpha):
                a, b = self.scale*self.a, self.scale*self.b
                L = a + b
                s, c = np.sin(alpha), np.cos(alpha)
                if s < 0:
                    a, b = b, a
                k = c/s
                A = 1/b**2
                B = (k*L/(a*b))**2
                C = 1/a**2
                D = (A + B + C)/2
                E = (A - B - C)/2
                F = np.sqrt(A*B + E**2)
                X = D - F
                Y = D + F
                Z = E/F
                a2 = np.sqrt(1/X)
                b2 = np.sqrt(1/Y)
                beta = np.arcsin(np.sqrt((1 + Z)/2))
                if k < 0:
                    beta = np.pi - beta
                return a2, b2, beta

            def get_full_ellipse(self, k1, k2):
                p11, p12 = self.ps[k1: k1 + 2]
                p21, p22 = self.ps[k2: k2 + 2]
                n11 = norm(p21 - p11)
                n12 = norm(p21 - p12)
                n21 = norm(p22 - p11)
                n22 = norm(p22 - p12)
                if min(n11, n12, n21, n22) >= self.L:
                    return None
                v1 = normalize(p12 - p11)
                v2 = normalize(p22 - p21)

                normal = np.cross(v2, np.cross(v1, v2))
                denom = v1@normal
                if abs(denom) < 0.01:
                    return None
                center = p11 + v1*((p21 - p11)@normal)/denom

                v = np.array([1, 1j, 0])
                angle1 = np.angle(v1@v)
                angle2 = np.angle(v2@v)
                alpha = angle2 - angle1
                a2, b2, beta = self.get_ellipse_params(alpha)
                ellipse = Ellipse(
                    2*a2, 2*b2,
                    fill_color=self.fill_color,
                    fill_opacity=self.opacity,
                    stroke_color=self.stroke_color,
                    stroke_opacity=1,
                ).set_z_index(0)
                ellipse.rotate(angle1 + beta).move_to(center)
                return ellipse

            def get_partial_ellipse(self, k1, k2):
                full_ellipse = self.get_full_ellipse(k1, k2)
                if full_ellipse is None:
                    return None
                partial_ellipse = Intersection(
                    self.curve, full_ellipse,
                    fill_color=self.fill_color,
                    fill_opacity=self.opacity,
                    stroke_color=self.stroke_color,
                    stroke_opacity=1
                ).set_z_index(-1)
                return partial_ellipse

            def get_ellipses(self, full=True):
                ellipses = []
                if full:
                    get_ellipse = self.get_full_ellipse
                else:
                    get_ellipse = self.get_partial_ellipse
                for k1, k2 in itertools.combinations(range(len(self.ps)), 2):
                    ellipse = get_ellipse(k1, k2)
                    if ellipse is not None:
                        ellipses.append(ellipse)
                return ellipses

        tex1 = MathTex(r'\text{Area}_a', '=', '{a', r'\over a + b}', r'\cdot',
                       r'\text{Area}_c')
        tex1[0].set_color(COLOR_GREEN)
        tex1[5].set_color(COLOR_RED)
        tex2 = MathTex(r'\text{Area}_b', '=', '{b', r'\over a + b}', r'\cdot',
                       r'\text{Area}_c')
        tex2[0].set_color(COLOR_GREEN)
        tex2[5].set_color(COLOR_RED)
        tex3 = MathTex(r'\text{Area}_a', '+', r'\text{Area}_b', '=', r'{a',
                       '+', 'b', r'\over a + b}', r'\cdot', r'\text{Area}_c')
        tex3[0].set_color(COLOR_GREEN)
        tex3[2].set_color(COLOR_GREEN)
        tex3[9].set_color(COLOR_RED)
        tex4 = MathTex(r'\text{Area}_a', '+', r'\text{Area}_b', '=',
                       r'\text{Area}_c')
        tex4[0].set_color(COLOR_GREEN)
        tex4[2].set_color(COLOR_GREEN)
        tex4[4].set_color(COLOR_RED)

        tex1.move_to(4*LEFT + 2.5*UP)
        tex2.next_to(tex1, DOWN, LARGE_BUFF)
        tex3.move_to(3*LEFT + 1*UP)
        tex4.move_to(tex3)
        self.add(tex1)

        center = 3*RIGHT + 2.8*DOWN
        triangle = Polygon(
            center + LEFT, center, center + (3/8)*RIGHT + 1.5*UP,
            stroke_color=COLOR_RED, stroke_opacity=1,
            fill_color=COLOR_DARK_RED, fill_opacity=1,
            z_index=0
        )

        A, B = 3, 2
        ps = [
            center + 6.5*LEFT,
            center + 6*LEFT,  # 1: start
            center + LEFT,  # 2
            center + (3/8)*RIGHT + 1.5*UP,  # 3: end of A
            center + 1.5*RIGHT + 6*UP,  # 4: end of `curve`
            center + (1.5 + 3/16)*RIGHT + (6 + 3/4)*UP,
            center + 2.5*RIGHT + 10*UP,
            center + 6.5*LEFT + 10*UP,
        ]
        time = 0
        vt = ValueTracker(time)
        holditch = PolygonalHolditch(
            vt, ps, a=A, b=B, speed=4.5,
            build_path=False, starting_index=1, stop_index=3
        )
        curve = PolygonalChain(*ps[:6], color=WHITE, z_index=2)
        brace_a = BraceLabel(holditch.chord.line_a, 'a')
        brace_b = BraceLabel(holditch.chord.line_b, 'b')

        ellipse1_partial = holditch.get_partial_ellipse(1, 3)
        ellipse2_partial = holditch.get_partial_ellipse(1, 2)
        ellipse3_partial = holditch.get_partial_ellipse(2, 3)
        ellipse2_partial = Difference(
            ellipse2_partial, ellipse1_partial,
            stroke_color=ellipse2_partial.get_stroke_color(), stroke_opacity=1,
            fill_color=COLOR_DARK_GREEN, fill_opacity=1,
            z_index=-2
        )
        ellipse3_partial = Difference(
            ellipse3_partial, ellipse1_partial,
            stroke_color=ellipse3_partial.get_stroke_color(), stroke_opacity=1,
            fill_color=COLOR_DARK_GREEN, fill_opacity=1,
            z_index=-2
        )

        self.play(
            FadeIn(curve), FadeIn(triangle), FadeIn(ellipse1_partial),
            FadeIn(ellipse2_partial), FadeIn(ellipse3_partial),
            *[FadeIn(mob) for mob in holditch.chord],
            run_time=0.5
        )
        self.play(FadeIn(brace_a), FadeIn(brace_b), run_time=0.5)
        self.wait(0.5)

        area_a = ellipse2_partial.copy()
        area_ca = triangle.copy()
        self.play(
            area_a.animate.next_to(tex1[0], DOWN, MED_SMALL_BUFF),
            area_ca.animate.next_to(tex1[5], DOWN, MED_SMALL_BUFF),
            run_time=1
        )
        self.wait()

        self.play(FadeOut(brace_a), FadeOut(brace_b), run_time=0.5)
        self.add(holditch.group)

        dt = 2
        print(dt)
        self.play(vt.animate.set_value(time := time + dt),
                  run_time=dt, rate_func=linear)
        self.wait(0.5)

        group = VGroup(curve, *holditch.chord, triangle, ellipse1_partial,
                       ellipse2_partial, ellipse3_partial)
        v = np.array([1, 1j, 0])
        angle1 = np.angle((ps[1] - ps[2])@v)
        angle2 = np.angle((ps[4] - ps[3])@v)
        theta = angle1 + angle2
        s, c = np.sin(theta), np.cos(theta)
        m = [[c, s], [s, -c]]
        self.play(
            group.animate.apply_matrix(m, about_point=center)
        )
        brace_a = BraceLabel(holditch.chord.line_a, 'a')
        brace_b = BraceLabel(holditch.chord.line_b, 'b')
        self.play(FadeIn(brace_a), FadeIn(brace_b), run_time=0.5)
        self.wait()

        area_b = ellipse3_partial.copy()
        area_cb = triangle.copy()
        self.play(
            area_b.animate.next_to(tex2[0], DOWN, MED_SMALL_BUFF),
            area_cb.animate.next_to(tex2[5], DOWN, MED_SMALL_BUFF),
            Write(tex2),
            tex1.animate.shift(0.5*UP),
            area_a.animate.shift(0.5*UP),
            area_ca.animate.shift(0.5*UP),
            run_time=1
        )
        self.wait()

        map13 = {0:0, 1:3, 2:4, 3:7, 4:8, 5:9}
        map23 = {0:2, 1:3, 2:6, 3:7, 4:8, 5:9}
        self.play(
            *get_tex_map_anims((tex1, tex2), (map13, map23), tex3),
            area_a.animate.next_to(tex3[0], DOWN, MED_LARGE_BUFF),
            area_b.animate.next_to(tex3[2], DOWN, MED_LARGE_BUFF),
            area_cb.animate.next_to(tex3[9], DOWN, MED_LARGE_BUFF),
            FadeOut(area_ca)
        )
        self.wait(0.5)

        map34 = {0:0, 1:1, 2:2, 3:3, 9:4}
        self.play(
            *get_tex_map_anims((tex3,), (map34,), tex4),
            area_a.animate.next_to(tex4[0], DOWN, MED_LARGE_BUFF),
            area_b.animate.next_to(tex4[2], DOWN, MED_LARGE_BUFF),
            area_cb.animate.next_to(tex4[4], DOWN, MED_LARGE_BUFF)
        )
        self.wait()

        text = Tex(r'Cutting a corner off preserves\\the total area!')
        text.move_to(3*UP)
        self.play(Write(text))
        self.wait(1.5)

        self.remove(*tex3)
        self.play(
            Unwrite(text), Unwrite(tex4),
            FadeOut(area_a), FadeOut(area_b), FadeOut(area_cb),
            *[FadeOut(mob) for mob in group],
            FadeOut(brace_a), FadeOut(brace_b),
            run_time=0.5
        )
        self.wait(0.1)


class Scene09(Scene):
    def construct(self):
        self.add(CC0)

        class PolygonalHolditch(Holditch):
            stroke_color = COLOR_BLUE
            fill_color = '#3400ff'
            fill_opacity = 0

            def __init__(
                self, vt, ps, a, b, speed=4, build_path=False,
                starting_index=None, stop_index=None, stop_when_closed=False,
                stroke_width=1, stroke_opacity=0.25
            ):
                super().__init__(
                    vt, ps, a=a, b=b, speed=speed, build_path=build_path,
                    stop_when_closed=stop_when_closed,
                    starting_index=starting_index
                )
                self.ps = self.curve_ps
                self.L = self.scale*(self.a + self.b)
                self.stroke_width = stroke_width
                self.stroke_opacity = stroke_opacity
                self.stop_index = stop_index
                if stop_index is None:
                    self.end_chord = None
                else:
                    self.end_chord = self._locus.get_chord_by_index(stop_index)
                self.stopped = False

            @property
            def index(self):
                return self._chord.cpa.k

            def update(self, group: VGroup | None = None):
                if self.stopped:
                    return
                if self.stop_index is None:
                    super().update(group)
                    return
                dt = self.vt.get_value() - self.t
                dist = dt*self.speed
                prev_chord = self._locus.chord
                super().update(group)
                new_chord = self._locus.chord
                dist_end = self._curve.get_dist(new_chord.cpa.d,
                                                self.end_chord.cpa.d)
                if (
                    abs(dist_end) < (1 + 0.01)*dist  # plus small tolerance
                    and prev_chord.is_ahead(self.end_chord)
                    and not new_chord.is_ahead(self.end_chord)
                ):
                    self._locus.chord = self.end_chord
                    self.stopped = True
                    print(f'\nStopped at {self.t = }')
                    if self.build_path:
                        self._locus.ps[-1] = self.end_chord.pc
                    self.chord.become(ChordVGroup(
                        *self._get_chord_points(),
                        lines_visible=self.chord.lines_visible,
                        opacity=self.chord.opacity))

            def get_ellipse_params(self, alpha):
                a, b = self.scale*self.a, self.scale*self.b
                L = a + b
                s, c = np.sin(alpha), np.cos(alpha)
                if s < 0:
                    a, b = b, a
                k = c/s
                A = 1/b**2
                B = (k*L/(a*b))**2
                C = 1/a**2
                D = (A + B + C)/2
                E = (A - B - C)/2
                F = np.sqrt(A*B + E**2)
                X = D - F
                Y = D + F
                Z = E/F
                a2 = np.sqrt(1/X)
                b2 = np.sqrt(1/Y)
                beta = np.arcsin(np.sqrt((1 + Z)/2))
                if k < 0:
                    beta = np.pi - beta
                return a2, b2, beta

            def get_full_ellipse(self, k1, k2):
                p11, p12 = self.ps[k1: k1 + 2]
                p21, p22 = self.ps[k2: k2 + 2]
                n11 = norm(p21 - p11)
                n12 = norm(p21 - p12)
                n21 = norm(p22 - p11)
                n22 = norm(p22 - p12)
                if min(n11, n12, n21, n22) >= self.L:
                    return None
                v1 = normalize(p12 - p11)
                v2 = normalize(p22 - p21)

                normal = np.cross(v2, np.cross(v1, v2))
                denom = v1@normal
                if abs(denom) < 0.01:
                    return None
                center = p11 + v1*((p21 - p11)@normal)/denom

                v = np.array([1, 1j, 0])
                angle1 = np.angle(v1@v)
                angle2 = np.angle(v2@v)
                alpha = angle2 - angle1
                a2, b2, beta = self.get_ellipse_params(alpha)
                ellipse = Ellipse(
                    2*a2, 2*b2,
                    fill_color=self.fill_color,
                    fill_opacity=self.fill_opacity,
                    stroke_color=self.stroke_color,
                    stroke_opacity=self.stroke_opacity,
                    stroke_width=self.stroke_width
                ).set_z_index(0)
                ellipse.rotate(angle1 + beta).move_to(center)
                return ellipse

            def get_partial_ellipse(self, k1, k2):
                full_ellipse = self.get_full_ellipse(k1, k2)
                if full_ellipse is None:
                    return None
                partial_ellipse = Intersection(
                    self.curve, full_ellipse,
                    fill_color=self.fill_color,
                    fill_opacity=self.fill_opacity,
                    stroke_color=self.stroke_color,
                    stroke_opacity=self.stroke_opacity
                ).set_z_index(-1)
                return partial_ellipse

            def get_ellipses(self, full=True):
                ellipses = []
                if full:
                    get_ellipse = self.get_full_ellipse
                else:
                    get_ellipse = self.get_partial_ellipse
                for k1, k2 in itertools.combinations(range(len(self.ps)), 2):
                    ellipse = get_ellipse(k1, k2)
                    if ellipse is not None:
                        ellipses.append(ellipse)
                return ellipses

        A, B = 2, 4
        time = 0

        a, b = 5, 3.1
        ellipse = Ellipse(2*a, 2*b, stroke_color=WHITE, z_index=3)
        self.play(FadeIn(ellipse), run_time=0.5)

        def get_holditch(k, vt, close=False, perpetual=False):
            num = 2**(2 + k)
            ts = np.linspace(-PI, PI, num=num, endpoint=False)
            xs = a*np.cos(ts)
            ys = b*np.sin(ts)
            pes = np.array([xs, ys, 0*ts]).T
            vs = ([[0, -a/b, 0], [b/a, 0, 0], [0, 0, 0]]@pes.T).T
            pes = Ring(pes)
            vs = Ring(vs)
            ps = find_intersection(pes, vs, pes[1:len(pes)+1], vs[1:len(vs)+1])
            ps2 = []
            for p1, p2 in Ring(ps).pairs():
                ps2.append(p1)
                ps2.append((p1 + p2)/2)
            if close or perpetual:
                stop_index = None
            else:
                stop_index = (k + 1)*2**(k + 2) - 1
            holditch = PolygonalHolditch(
                vt, ps2, a=A, b=B, speed=4.5, build_path=close,
                starting_index=max(0, k*2**(k + 2) - 1),
                stop_index=stop_index,
                stop_when_closed = close and not perpetual,
                stroke_width=2**(2 - 3*k/4),
                stroke_opacity=2**(-3*k/4)
            )
            holditch.curve.set_stroke_width(2)
            holditch.curve.set_z_index(2)
            return holditch

        def get_temp_holditch(k):
            temp_vt = ValueTracker(0)
            temp_holditch = get_holditch(k, temp_vt, True)
            temp_vt.set_value(10)
            temp_holditch.update()
            return temp_holditch

        vt = ValueTracker(time)
        holditch = get_holditch(0, vt)
        ellipses = holditch.get_ellipses()
        self.play(
            *[FadeIn(mob) for mob in holditch.chord],
            Create(holditch.curve),
            run_time=1
        )

        temp_holditch = get_temp_holditch(0)
        area = temp_holditch.strip.set_z_index(-1)
        locus = temp_holditch.locus.set_z_index(1)

        self.play(
            *[FadeIn(mob) for mob in ellipses],
            FadeIn(area), FadeIn(locus),
            run_time=1
        )
        self.add(holditch.group)
        dt = 2.02
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        curves = [holditch.curve]
        old_area = area
        old_locus = locus
        all_triangles = []
        max_k = 4
        for k in range(1, max_k + 1):
            print(k)

            temp_holditch = get_temp_holditch(k)
            area = temp_holditch.strip.set_z_index(-1)
            locus = temp_holditch.locus.set_z_index(1)
            new_curve = temp_holditch.curve
            curves.append(new_curve)

            dt = 2
            self.play(
                vt.animate.set_value(time := time + dt),
                Create(new_curve),
                run_time=dt, rate_func=linear
            )

            triangles = Difference(
                holditch.curve, new_curve,
                stroke_opacity=0,
                fill_color=COLOR_DARK_RED, fill_opacity=1,
                z_index=1
            )
            all_triangles.append(triangles)

            self.remove(holditch.group)
            vt = ValueTracker(time)
            if k == max_k:
                holditch = get_holditch(k, vt, perpetual=True)
            else:
                holditch = get_holditch(k, vt)
            old_ellipses = ellipses
            ellipses = holditch.get_ellipses()

            self.add(holditch.group)
            dt = 1
            self.play(
                vt.animate.set_value(time := time + dt),
                FadeIn(triangles),
                *[FadeOut(mob) for mob in old_ellipses],
                *[FadeIn(mob) for mob in ellipses],
                FadeOut(old_area), FadeIn(area),
                FadeOut(old_locus), FadeIn(locus),
                run_time=dt, rate_func=linear
            )
            old_area = area
            old_locus = locus
            dt = [0.94, 0.84, 0.8, 2][k - 1]
            print(f'\n{time = }')
            self.play(
                vt.animate.set_value(time := time + dt),
                run_time=dt, rate_func=linear
            )

        tex = MathTex(r'\text{Area}', r'= \pi').set_z_index(10)
        tex[0].set_color(COLOR_BLUE)
        dt = 1
        self.play(
            vt.animate.set_value(time := time + dt),
            Write(tex),
            run_time=dt, rate_func=linear
        )

        dt = 2
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        dt = 1
        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(time - dt, time)
        holditch.group.add_updater(chord_opacity_updater)
        self.play(
            vt.animate.set_value(time := time + dt),
            Unwrite(tex),
            *[FadeOut(mob) for mob in curves],
            *[FadeOut(mob) for mob in ellipses],
            *[FadeOut(mob) for mob in all_triangles],
            FadeOut(area), FadeOut(locus),
            FadeOut(ellipse),
            run_time=dt, rate_func=linear
        )
        self.remove(holditch.group)
        self.wait(0.1)


class Scene10(Scene):
    def construct(self):
        self.add(CC0)

        text = Tex(r'This proof was discovered by Juan Monterde\\'
                   'and David Rochera in 2017.')
        self.play(Write(text))
        self.wait(1.5)
        self.play(Unwrite(text, reverse=False), run_time=0.5)
        text = Tex(r'It applies to any sufficiently large\\'
                   'closed convex curve.')
        self.play(Write(text))
        self.wait(1.5)
        self.play(Unwrite(text, reverse=False), run_time=0.5)

        text = Tex(r'If the curve is too small,\\'
                   'the chord will not complete a full turn.')
        text.move_to(2.5*UP)
        self.play(Write(text))

        scale = 0.04
        A, B = 40*scale, 70*scale
        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'ellipse', center=DOWN, a=A, b=B,
                            speed=100, scale=scale)
        self.play(
            FadeIn(holditch.curve),
            *[FadeIn(mob) for mob in holditch.chord]
        )
        self.add(holditch.group)

        dt = 5.5
        self.play(
            vt.animate.set_value(time := time + dt),
            Succession(
                Delay(dt - 0.5),
                Unwrite(text, reverse=False, run_time=0.5)
            ),
            run_time=dt, rate_func=linear
        )

        text = Tex('And the area will be different.').move_to(text)
        dt = 4.5
        self.play(
            vt.animate.set_value(time := time + dt),
            Succession(
                Write(text, run_time=1),
                Delay(dt - 1)
            ),
            run_time=dt, rate_func=linear
        )

        dt = 1
        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(time - dt, time)
        holditch.group.add_updater(chord_opacity_updater)
        self.play(
            vt.animate.set_value(time := time + dt),
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            FadeOut(holditch.strip),
            Unwrite(text),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch.group)
        self.wait(0.1)
