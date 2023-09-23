import itertools
import os

from manim import *
import numpy as np

from HolditchMobjects import Holditch, ChordVGroup
from Ring import Ring
from ManimUtils import (
    rotate, get_rotate_and_shift_homotopy, get_tex_map_anims, indicate_tex
)


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


class Scene11(Scene):
    def construct(self):
        self.add(CC0)


        class LabelWithParameter(VGroup):
            def __init__(self, vt: ValueTracker, max_vt, label: str,
                         parameter=None):
                self.vt = vt
                self.max_vt = max_vt
                self.dot = dot
                self.chord = chord
                self.label = label
                self.parameter = None
                self.scale = 0.7
                self.text = self.get_text(label, parameter)
                self.decimal = DecimalNumber(
                    vt.get_value(),
                    show_ellipsis=False,
                    num_decimal_places=2,
                    include_sign=False,
                    font_size=DEFAULT_FONT_SIZE*self.scale,
                    z_index=6
                )
                self.decimal.add_updater(self.update_decimal)
                if self.is_decimal_visible:
                    self.decimal.move_to(text[2])
                else:
                    self.decimal.set_opacity(0)

                super().__init__(self.text, self.decimal)

            @property
            def is_decimal_visible(self):
                return self.parameter is not None and self.parameter != 't'

            def get_anims(self, label, parameter):
                self.update_decimal()
                text_copy = self.text.copy()
                new_text = self.get_text(label, parameter)
                new_text.move_to(self.text)
                anims = []
                if len(new_text) > 1:
                    if len(self.text) == 1:
                        self.text.become(new_text)
                        self.text[0].become(text_copy[0])
                        anims.append(Transform(self.text[0], new_text[0]))
                        anims.append(FadeIn(self.text[1:]))
                    else:
                        anims.extend([
                            Transform(self.text[k], new_text[k])
                            for k in range(4)
                        ])
                else:
                    text_copy
                    self.text.become(new_text)
                    self.text[0].become(text_copy[0])
                    anims.append(Transform(self.text[0], new_text[0]))
                    if len(text_copy) > 1:
                        anims.append(FadeOut(text_copy[1:]))

                if parameter is None or parameter == 't':
                    if self.is_decimal_visible:
                        anims.append(FadeOut(self.decimal))
                else:
                    if self.is_decimal_visible:
                        anims.append(self.decimal.animate.move_to(new_text[2]))
                    else:
                        self.decimal.move_to(new_text[2])
                        self.decimal.set_opacity(1)
                        anims.append(FadeIn(self.decimal))

                self.label = label
                self.parameter = parameter
                return anims

            def get_text(self, label, parameter):
                font_size = DEFAULT_FONT_SIZE*self.scale
                if parameter is None:
                    text = MathTex(label, font_size=font_size)
                else:
                    text = MathTex(label, '(', parameter, ')',
                                   font_size=font_size)
                    if parameter != 't':
                        text[2].set_opacity(0)
                return text.set_z_index(6)

            def update_decimal(self, _=None):
                self.decimal.set_value(self.vt.get_value()/self.max_vt)


        class ChordLabelWithParameter(LabelWithParameter):
            def __init__(
                self, vt: ValueTracker, max_vt,
                dot: Dot, chord: ChordVGroup, label: str, parameter
            ):
                super().__init__(vt, max_vt, label, parameter)
                self.dot = dot
                self.chord = chord
                self.update_position()
                #self.add_updater(self.update_position)

            def update_position(self, _=None):
                pa = self.chord.dot_a.get_center()
                pb = self.chord.dot_b.get_center()
                p = self.dot.get_center()
                buff = 0.8*self.scale
                self.move_to(p + buff*normalize(rotate(pb - pa, PI/2)))


        class BasisVectors(VGroup):
            def __init__(self, vt: ValueTracker, max_vt, center, chord):
                self.vt = vt
                self.max_vt = max_vt
                self.center = center
                self.chord = chord
                self.u_paral = Vector(RIGHT)
                self.u_perp = Vector(UP)
                self.label_paral = LabelWithParameter(
                    vt, self.max_vt, r'\hat u_\parallel', 't')
                self.label_perp = LabelWithParameter(
                    vt, self.max_vt, r'\hat u_\perp', 't')
                self.right_angle = RightAngle(self.u_paral, self.u_perp)
                self.dot = Dot(center, color=GRAY, z_index=1)
                self.dash = DashedLine(center, center + RIGHT, stroke_width=1)
                self.dash.set_z_index(-1)
                self.dash_angle = Angle(self.dash, self.u_perp)
                self.angle_label = LabelWithParameter(
                    vt, self.max_vt, r'\theta', 't')
                self.update()
                super().__init__(
                    self.u_paral, self.u_perp, self.dot,
                    *self.label_paral, *self.label_perp, self.right_angle,
                    self.dash, self.dash_angle, *self.angle_label
                )
                self.add_updater(self.update)

            def update(self, _=None):
                pa = self.chord.dot_a.get_center()
                pb = self.chord.dot_b.get_center()
                v = np.array([1, 1j, 0])
                angle = np.angle((pb - pa)@v)
                self.u_paral.become(
                    Vector(
                        RIGHT, color=GRAY, stroke_width=2, tip_length=0.2
                    ).rotate(angle, about_point=ORIGIN).shift(self.center)
                )
                self.u_perp.become(
                    Vector(
                        UP, color=GRAY, stroke_width=2, tip_length=0.2
                    ).rotate(angle, about_point=ORIGIN).shift(self.center)
                )
                buff = 0.5
                self.label_paral.update_decimal()
                self.label_paral.move_to(
                    self.center + (1 + buff)*rotate(RIGHT, angle)
                )
                self.label_perp.update_decimal()
                self.label_perp.move_to(
                    self.center + (1 + buff)*rotate(UP, angle)
                )
                self.right_angle.become(
                    RightAngle(self.u_paral, self.u_perp, 0.2,
                               color=GRAY, stroke_width=2)
                )
                self.dot.move_to(self.center)
                self.dash.move_to(self.center + 0.5*RIGHT)

                if abs(angle) < 0.01:
                    self.dash_angle.become(Angle(self.dash, self.u_perp,
                                                 stroke_opacity=0))
                elif angle > 0:
                    self.dash_angle.become(Angle(self.dash, self.u_paral,
                                                 stroke_width=1, radius=0.3))
                else:
                    self.dash_angle.become(Angle(self.u_paral, self.dash,
                                                 stroke_width=1, radius=0.3))
                self.angle_label.update_decimal()
                self.angle_label.move_to(self.center + 0.3*RIGHT + 0.3*DOWN)


        A, B = 1.5, 3
        pa, pb = A*LEFT, B*RIGHT
        pc = ORIGIN
        time = 0
        vt = ValueTracker(time)
        max_vt = 6.55
        holditch = Holditch(vt, 'generic', a=A, b=B, center=DOWN,
                            scale=0.04, speed=100, stop_when_closed=True)
        chord_copy = holditch.chord.copy()
        holditch.chord.become(ChordVGroup(pa, pb, pc))
        chord = holditch.chord

        text = Tex("The standard proof uses Green's Theorem.")
        text.move_to(3*UP)
        self.play(Write(text))
        self.add(text)

        p_labels = []
        for dot, label in zip([chord.dot_a, chord.dot_b, chord.dot_c], 'ABC'):
            p_labels.append(
                ChordLabelWithParameter(vt, max_vt, dot, chord, label, None)
            )

        self.play(
            *[FadeIn(mob) for mob in chord],
            *[FadeIn(mob) for p_label in p_labels for mob in p_label],
            run_time=1
        )

        axis_origin = 3.5*LEFT + 2*DOWN
        plane = NumberPlane().move_to(axis_origin).set_z_index(-1)
        self.play(
            Unwrite(text),
            FadeIn(plane),
            run_time=1
        )
        vectors = [
            Vector(
                p - axis_origin, stroke_width=2, tip_length=0.2
            ).shift(axis_origin).set_z_index(6)
            for p in [pa, pb, pc]
        ]
        self.play(
            *itertools.chain(*(
                p_label.get_anims(fr'\vec{{{p_label.label}}}', None)
                for p_label in p_labels
            )),
            *[Create(vector) for vector in vectors],
            run_time=1
        )
        self.wait()
        self.play(
            *itertools.chain(*(
                p_label.get_anims(p_label.label, 't')
                for p_label in p_labels
            )),
            FadeOut(plane),
            *[FadeOut(vector) for vector in vectors],
            run_time=1
        )
        basis = BasisVectors(vt, max_vt, 4*RIGHT + 2*UP, chord)
        self.play(
            *[FadeIn(mob) for mob in basis],
            run_time=1
        )
        run_time = 2
        def rotation_update():
            angle = rate_functions.wiggle(vt.get_value()/run_time)
            pa = chord.dot_a.get_center()
            pb = chord.dot_b.get_center()
            v = np.array([1, 1j, 0])
            prev_angle =  np.angle((pb - pa)@v)
            chord.rotate(angle - prev_angle, about_point=(pa + pb)/2)
        chord_updater = lambda _: rotation_update()
        chord.add_updater(chord_updater)
        for p_label in p_labels:
            p_label.add_updater(p_label.update_position)
        self.add(chord, basis, *p_labels)
        self.play(
            vt.animate.set_value(run_time),
            run_time=run_time, rate_func=linear
        )
        chord.remove_updater(chord_updater)
        for p_label in p_labels:
            p_label.remove_updater(p_label.update_position)
        self.wait(0.5)

        v = np.array([1, 1j, 0])
        pa0 = chord_copy.dot_a.get_center()
        pb0 = chord_copy.dot_b.get_center()
        angle = np.angle((pb0 - pa0)@v)
        time = 0
        vt.set_value(time)
        run_time = 1.5
        homotopy = get_rotate_and_shift_homotopy(angle, pa, pa0 - pa)
        for p_label in p_labels:
            p_label.add_updater(p_label.update_position)
        self.play(
            FadeIn(holditch.curve),
            Homotopy(homotopy, chord),
            run_time=run_time
        )
        for p_label in p_labels:
            p_label.remove_updater(p_label.update_position)

        labels = [*p_labels, basis.label_paral, basis.label_perp,
                  basis.angle_label]
        self.add(*labels)
        self.play(
            *itertools.chain(*(
                label.get_anims(label.label, '0.00')
                for label in labels
            )),
            run_time=1
        )

        self.add(holditch.group)
        dt = max_vt
        for p_label in p_labels:
            p_label.add_updater(p_label.update_position)
        self.add(basis)
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        for p_label in p_labels:
            p_label.remove_updater(p_label.update_position)

        self.add(*labels)
        self.play(
            *itertools.chain(*(
                label.get_anims(label.label, 't')
                for label in labels
            )),
            run_time=1
        )

        text = Tex(r'$\partial \mathcal S_P$ is the path\\'
                   r'traced by $\vec P(t)$').move_to(3*UP)
        tex_Dab = MathTex(r'\partial \mathcal S_A', '\equiv',
                          r'\partial \mathcal S_B').move_to(5*LEFT + 1.5*UP)
        tex_Dab[0].set_color(COLOR_GREEN)
        tex_Dab[2].set_color(COLOR_GREEN)
        tex_Dc = MathTex(r'\partial \mathcal S_C').move_to(0.5*LEFT + 0.5*DOWN)
        tex_Dc.set_color(COLOR_RED)
        arrow_Dab = CurvedArrow(
            5*LEFT + UP, (3 + 2/3)*LEFT + (1/3)*UP,
            color=GRAY, radius=2, tip_length=0.2
        ).set_z_index(6)
        arrow_Dc = CurvedArrow(
            1.1*LEFT + 0.5*DOWN, 2.2*LEFT + 0.2*UP,
            color=GRAY, radius=-1.5, tip_length=0.2
        ).set_z_index(6)
        self.remove(*[label.decimal for label in labels])
        self.play(
            Write(text),
            Write(tex_Dab), Write(tex_Dc),
            Create(arrow_Dab), Create(arrow_Dc),
            FadeOut(holditch.strip),
            holditch.curve.animate.set_stroke(COLOR_GREEN),
            holditch.locus.animate.set_stroke(COLOR_RED),
            p_labels[0].text.animate.set_color(COLOR_GREEN),
            p_labels[1].text.animate.set_color(COLOR_GREEN),
            p_labels[2].text.animate.set_color(COLOR_RED),
            chord.dot_a.animate.set_fill(COLOR_GREEN),
            chord.dot_b.animate.set_fill(COLOR_GREEN),
            chord.dot_c.animate.set_fill(COLOR_RED),
            run_time=1
        )
        self.wait(2.5)

        self.play(
            Unwrite(text, reverse=False),
            Unwrite(tex_Dab), Unwrite(tex_Dc),
            FadeOut(arrow_Dab), FadeOut(arrow_Dc),
            p_labels[0].text.animate.set_color(WHITE),
            p_labels[1].text.animate.set_color(WHITE),
            p_labels[2].text.animate.set_color(WHITE),
            run_time=0.5
        )

        text = Tex(r'$\mathcal S_P$ is the signed area of the\\'
                   r'region enclosed by $\partial \mathcal S_P$').move_to(text)
        tex_Sab = MathTex(r'\mathcal S_A', '=', r'\mathcal S_B')
        tex_Sab.move_to(5*LEFT + 1.5*UP)
        tex_Sab[0].set_color(COLOR_GREEN)
        tex_Sab[2].set_color(COLOR_GREEN)
        tex_Sc = MathTex(r'\mathcal S_C').move_to(4.5*LEFT + 0.5*UP)
        tex_Sc.set_color(COLOR_RED)
        tex_Sd = MathTex(
            r'\Delta \mathcal S', '&=', r'\mathcal S_A', '-', r'\mathcal S_C',
            r'\\&=', r'\mathcal S_B', '-', r'\mathcal S_C'
        ).move_to(5*LEFT + 2*UP)
        tex_Sd[0].set_color(COLOR_BLUE)
        tex_Sd[2].set_color(COLOR_GREEN)
        tex_Sd[4].set_color(COLOR_RED)
        tex_Sd[6].set_color(COLOR_GREEN)
        tex_Sd[8].set_color(COLOR_RED)
        arrow_Sab = CurvedArrow(
            4.5*LEFT + UP, 3*LEFT + 0.5*UP,
            color=GRAY, radius=2.5, tip_length=0.2
        ).set_z_index(6)
        arrow_Sc = CurvedArrow(
            4.1*LEFT + 0.1*UP, (2 + 1/3)*LEFT + 0.5*DOWN,
            color=GRAY, radius=3, tip_length=0.2
        ).set_z_index(6)
        self.play(
            Write(text),
            Write(tex_Sab),
            Create(arrow_Sab),
            FadeOut(holditch.locus),
            holditch.curve.animate.set_fill(COLOR_DARK_GREEN, opacity=1),
            run_time=1
        )
        self.wait(2)

        arrow_Sab_copy = arrow_Sab.copy()
        self.play(
            Unwrite(tex_Sab), Write(tex_Sc),
            ReplacementTransform(arrow_Sab, arrow_Sc),
            holditch.curve.animate.set_fill(COLOR_DARK_GREEN, opacity=0),
            holditch.locus.animate.set_fill(COLOR_DARK_RED, opacity=1),
            run_time=1
        )
        self.wait(1.5)

        self.play(
            Unwrite(text),
            Unwrite(tex_Sc), Write(tex_Sd),
            ReplacementTransform(arrow_Sc, arrow_Sab_copy),
            holditch.curve.animate.set_fill(COLOR_DARK_GREEN, opacity=0),
            holditch.locus.animate.set_fill(COLOR_DARK_RED, opacity=0),
            FadeIn(holditch.strip),
            run_time=1
        )
        self.wait(2)

        basis.remove(*[mob for label in labels for mob in label])
        self.play(
            *[FadeOut(mob) for mob in chord],
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            FadeOut(holditch.strip),
            FadeOut(arrow_Sab_copy),
            *[FadeOut(label.text) for label in labels],
            *[FadeOut(mob) for mob in basis],
            Unwrite(tex_Sd),
            run_time=0.5
        )

        self.wait(0.1)


class Scene12(Scene):
    def construct(self):
        self.add(CC0)

        A, B = 1.5, 3
        center = ORIGIN
        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'generic', a=A, b=B, center=center,
                            scale=0.04, speed=100, stop_when_closed=True)
        axis_origin = 5*LEFT + 3*DOWN
        plane = NumberPlane(
            [-2, 12], [-1, 7],
            background_line_style={'stroke_opacity': 0}
        ).set_z_index(-2)
        self.play(
            FadeIn(plane),
            FadeIn(holditch.curve),
            run_time=1
        )

        p1 = holditch.curve_ps[0]
        arrow = Vector(
            p1 - axis_origin, stroke_width=4, tip_length=0.2, color=GRAY
        ).shift(axis_origin).set_z_index(6)
        def get_label_pos(p1, p2, r=0.5):
            p0 = (p1 + p2)/2
            dp0 = r*normalize(rotate(p2 - p1, PI/2))
            return p0 + dp0
        tex_p = MathTex(r'\vec P').move_to(get_label_pos(axis_origin, p1))
        tex_p.set_color(GRAY)
        self.play(
            Write(tex_p),
            Create(arrow),
            run_time=0.5
        )
        self.wait(0.5)

        max_k = 20
        dvs = []
        areas = []
        p2 = holditch.curve.point_from_proportion(1/max_k)
        dp = p2 - p1
        new_arrow = Vector(
            p2 - axis_origin, stroke_width=4, tip_length=0.2, color=GRAY
        ).shift(axis_origin)
        dv = Vector(
            p2 - p1, stroke_width=4, tip_length=0.2, color=COLOR_RED,
            stroke_opacity=1
        ).shift(p1).set_z_index(5)
        dvs.append(dv)
        area = Polygon(
            axis_origin, p1, p2,
            stroke_color=WHITE, stroke_opacity=0.75, stroke_width=1,
            fill_color=COLOR_DARK_RED, fill_opacity=0.75, z_index=-1
        )
        areas.append(area)
        tex_dp = MathTex(r'd\vec P').move_to(get_label_pos(p1, p1 + dp))
        self.play(
            Write(tex_dp),
            Transform(arrow, new_arrow),
            FadeIn(area),
            Create(dv),
            run_time=1
        )
        self.wait(0.5)

        tex_area = MathTex(
            r'\text{Signed Area}\\', r'=', r'\vec P', r'\wedge', r'd\vec P'
        ).move_to(1.5*LEFT + 1.5*UP)
        arrow_area = CurvedArrow(
            2.5*LEFT + 0.5*UP, 2.7*LEFT + DOWN,
            color=GRAY, radius=2, tip_length=0.2
        )
        area2 = Polygon(
            p1, p2, p1 + p2 - axis_origin,
            stroke_color=WHITE, stroke_opacity=0.75, stroke_width=1,
            fill_color=COLOR_DARK_RED, fill_opacity=0.75, z_index=-1
        )
        self.play(
            FadeIn(area2),
            run_time=0.5
        )
        self.play(
            Write(tex_area[:2]),
            *get_tex_map_anims((tex_p, tex_dp), ({0:0}, {0:2}),
                               tex_area[2:], (True, True)),
            Create(arrow_area),
            run_time=1
        )
        self.wait(1)

        text_wedge = Tex(r'wedge product\\(``2D–cross product")')
        text_wedge.move_to(RIGHT + 0.5*DOWN)
        arrow_wedge = CurvedArrow(
            text_wedge.get_corner(LEFT + UP) + 0.5*RIGHT,
            tex_area[3].get_edge_center(DOWN) + 0.1*DOWN,
            color=GRAY, radius=-1, tip_length=0.2
        )
        self.play(
            Write(text_wedge),
            Create(arrow_wedge),
            run_time=1
        )
        self.play(indicate_tex(tex_area[3]))

        arrow_area2 = CurvedArrow(
            2.5*LEFT + 0.5*UP, p1 + 0.1*RIGHT + 0.1*UP,
            color=GRAY, radius=3, tip_length=0.2
        )
        tex_dS = MathTex(
            'd', r'\mathcal S_P', r'=', r'\frac{1}{2}',
            r'\vec P', r'\wedge', r'd\vec P'
        ).move_to(tex_area[3])
        map_area_dS = {0:1, 1:2, 2:4, 3:5, 4:6}
        self.play(
            *get_tex_map_anims((tex_area,), (map_area_dS,), tex_dS),
            FadeOut(area2),
            FadeOut(arrow_wedge),
            Unwrite(text_wedge),
            Transform(arrow_area, arrow_area2),
            run_time=1
        )
        self.wait(1.5)

        text_red = Tex('RED', ' means negative area').move_to(0.5*DOWN)
        text_red[0].set_color(COLOR_RED)
        self.play(
            Write(text_red),
            FadeOut(arrow_area),
            run_time=1
        )
        self.wait(1)
        self.play(
            Unwrite(text_red),
            run_time=0.5
        )

        p1 = p2
        v = np.array([1, 1j, 0])
        angle1 = np.angle((p1 - axis_origin)@v)
        run_time = 4
        for k in range(2, max_k + 1):
            p2 = holditch.curve.point_from_proportion(k/max_k)
            angle2 = np.angle((p2 - axis_origin)@v)
            delta_angle = angle2 - angle1
            new_arrow = Vector(
                p2 - axis_origin, stroke_width=4, tip_length=0.2, color=GRAY
            ).shift(axis_origin)
            area_color = COLOR_DARK_BLUE if delta_angle > 0 else COLOR_DARK_RED
            vector_color = COLOR_BLUE if delta_angle > 0 else COLOR_RED
            z_index = -2 if delta_angle > 0 else -1
            dv = Vector(
                p2 - p1, stroke_width=4, tip_length=0.2, color=vector_color,
                stroke_opacity=1
            ).shift(p1).set_z_index(5)
            area = Polygon(
                axis_origin, p1, p2,
                stroke_color=vector_color, stroke_opacity=0.75, stroke_width=1,
                fill_color=area_color, fill_opacity=0.75, z_index=z_index
            )
            p1 = p2
            angle1 = angle2
            dvs.append(dv)
            areas.append(area)
            self.play(
                Transform(arrow, new_arrow),
                FadeIn(area),
                Create(dv),
                run_time=run_time/(max_k - 1)
            )
        self.wait(1)

        tex_int0 = MathTex(
            r'\iint', 'd', r'\mathcal S_P', r'=', r'\frac{1}{2}',
            r'\oint_{\partial \mathcal S_P}',
            r'\vec P', r'\wedge', r'd\vec P'
        ).move_to(0.5*LEFT)
        map_dS_int = {0:1, 1:2, 2:3, 3:4, 4:6, 5:7, 6:8}
        self.play(
            *get_tex_map_anims((tex_dS,), (map_dS_int,), tex_int0),
            run_time=1
        )
        self.wait(1)
        tex_int = MathTex(
            r'\mathcal S_P', r'=', r'\frac{1}{2}',
            r'\oint_{\partial \mathcal S_P}',
            r'\vec P', r'\wedge', r'd\vec P'
        ).move_to(tex_int0)
        map_int = {2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 8:6}
        self.play(
            *get_tex_map_anims((tex_int0,), (map_int,), tex_int),
            run_time=1
        )
        self.wait(0.5)
        self.play(
            tex_int.animate.move_to(2.5*UP),
            FadeOut(plane), FadeOut(holditch.curve),
            FadeOut(arrow),
            *[FadeOut(mob) for mob in areas],
            *[FadeOut(mob) for mob in dvs],
            Unwrite(tex_p), Unwrite(tex_dp),
            run_time=1
        )
        box = SurroundingRectangle(tex_int, color=TEAL)
        self.play(
            Create(box),
            tex_int.animate.set_color(TEAL)
        )

        text = Tex("This is Green's Theorem for:")
        text.move_to(tex_int, DOWN, MED_LARGE_BUFF)
        self.play(
            Write(text),
            run_time=1
        )
        tex_brace = MathTex(r'&L = -\frac{1}{2}y\\&M = \frac{1}{2}x')
        tex_brace.next_to(text, DOWN, MED_LARGE_BUFF)
        brace = Brace(tex_brace, LEFT)
        self.play(
            Write(tex_brace),
            Create(brace),
            run_time=1
        )
        tex_greens = MathTex(
            r'\iint_{\mathcal S_P} \left(\frac{\partial M}{\partial x}'
            r'- \frac{\partial L}{\partial y}\right)'
            r'= \oint_{\partial \mathcal S_P} \left(L dx + M dy\right)'
        ).next_to(tex_brace, DOWN, MED_LARGE_BUFF)
        self.play(
            Write(tex_greens),
            run_time=1
        )
        self.wait(3)

        self.play(
            FadeOut(brace),
            Unwrite(text), Unwrite(tex_brace), Unwrite(tex_greens),
            run_time=0.5
        )

        self.wait(0.1)

class Scene13(Scene):
    def construct(self):
        self.add(CC0)

        tex_int = MathTex(
            r'\mathcal S_P', r'=', r'\frac{1}{2}',
            r'\oint_{\partial \mathcal S_P}',
            r'\vec P', r'\wedge', r'd\vec P',
            color=TEAL
        ).move_to(2.5*UP)
        box = SurroundingRectangle(tex_int, color=TEAL)
        self.add(tex_int, box)
        self.play(
            tex_int.animate.shift(3*LEFT),
            box.animate.shift(3*LEFT),
            run_time=1
        )

        class BasisVectors(VGroup):
            def __init__(self, center, angle):
                self.center = center
                self.angle = angle

                self.u_paral = Vector(RIGHT, color=GRAY,
                                      stroke_width=2, tip_length=0.2)
                self.u_perp = Vector(UP, color=GRAY,
                                     stroke_width=2, tip_length=0.2)
                self.label_paral = MathTex(r'\hat u_\parallel',
                                           font_size=0.8*DEFAULT_FONT_SIZE)
                self.label_perp = MathTex(r'\hat u_\perp',
                                           font_size=0.8*DEFAULT_FONT_SIZE)

                self.u_paral.rotate(angle, about_point=ORIGIN).shift(center)
                self.u_perp.rotate(angle, about_point=ORIGIN).shift(center)
                buff = 0.3
                self.label_paral.move_to(
                    center + (1 + buff)*rotate(RIGHT, angle)
                )
                self.label_perp.move_to(
                    center + rotate(buff*RIGHT + UP, angle)
                )

                self.right_angle = RightAngle(self.u_paral, self.u_perp, 0.2,
                                              color=GRAY, stroke_width=2)
                self.dot = Dot(center, color=GRAY, z_index=1)
                self.dash = DashedLine(center, center + RIGHT, stroke_width=1)
                self.dash.move_to(center + 0.5*RIGHT).set_z_index(-1)
                self.dash_angle = Angle(self.dash, self.u_paral,
                                        stroke_width=1, radius=0.3)
                self.angle_label = MathTex(r'\theta',
                                           font_size=0.8*DEFAULT_FONT_SIZE)
                self.angle_label.move_to(center + 0.3*RIGHT + 0.3*DOWN)

                super().__init__(
                    self.u_paral, self.u_perp, self.dot,
                    self.label_paral, self.label_perp, self.right_angle,
                    self.dash, self.dash_angle, self.angle_label
                )

        A, B = 1, 2
        angle = 30*DEGREES
        pa = 1.5*RIGHT + 1*UP
        pb = pa + rotate((A + B)*RIGHT, angle)
        pc = pa + A*normalize(pb - pa)
        chord = ChordVGroup(pa, pb, pc)
        basis = BasisVectors(pa + 3.5*RIGHT, angle)
        fs = 0.8*DEFAULT_FONT_SIZE
        v_buff = 0.4*normalize(rotate(pb - pa, PI/2))
        labels = [
            MathTex(r'\vec A', font_size=fs).move_to(pa + v_buff),
            MathTex(r'\vec B', font_size=fs).move_to(pb + v_buff),
            MathTex(r'\vec C', font_size=fs).move_to(pc + v_buff),
            MathTex('a', font_size=fs).move_to((pa + pc)/2 - v_buff),
            MathTex('b', font_size=fs).move_to((pb + pc)/2 - v_buff),
        ]
        labels[0].set_color(BLUE)
        labels[1].set_color(GREEN)
        labels[3].set_color(BLUE)
        labels[4].set_color(GREEN)
        self.play(
            *[FadeIn(mob) for mob in chord],
            *[FadeIn(mob) for mob in basis],
            *[FadeIn(mob) for mob in labels],
            run_time=1
        )
        self.wait(1)

        tex_b = MathTex(
            r'\vec B', '=', r'\vec C', '+', 'b', r'\hat u_\parallel'
        ).next_to(tex_int, DOWN, MED_LARGE_BUFF).shift(2*LEFT)
        tex_b[0].set_color(GREEN)
        tex_b[4].set_color(GREEN)
        tex_bdot1 = MathTex(
            r'\dot{\vec B}', '=', r'\dot{\vec C}', '+', 'b',
            r'\dot{\hat u}_\parallel'
        ).next_to(tex_int, DOWN, MED_LARGE_BUFF).shift(2*RIGHT)
        tex_bdot1[0].set_color(GREEN)
        tex_bdot1[4].set_color(GREEN)
        tex_bdot2 = MathTex(
            r'\dot{\vec B}', '=', r'\dot{\vec C}', '+', 'b',
            r'\hat u_\perp', r'\dot\theta'
        ).move_to(tex_bdot1)
        tex_bdot2[0].set_color(GREEN)
        tex_bdot2[4].set_color(GREEN)
        self.play(
            Write(tex_b),
            run_time=1
        )
        self.wait(1)
        map_b_bdot1 = {k:k for k in range(6)}
        self.play(
            *get_tex_map_anims((tex_b,), (map_b_bdot1,), tex_bdot1, (True,)),
            run_time=1
        )
        self.wait(0.5)
        map_bdot12 = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 5:6}
        self.play(
            *get_tex_map_anims((tex_bdot1,), (map_bdot12,), tex_bdot2),
            run_time=1
        )
        self.wait(1)

        tex_intb1 = MathTex(
            r'\mathcal S_B', r'=', r'\frac{1}{2}',
            r'\oint_{\partial\mathcal S_B}', r'\vec B', r'\wedge', r'd\vec B'
        ).move_to(0.5*DOWN)
        tex_intb1[0].set_color(GREEN)
        tex_intb1[4].set_color(GREEN)
        tex_intb1[6].set_color(GREEN)
        tex_intb2 = MathTex(
            r'\mathcal S_B', r'=', r'\frac{1}{2}', r'\int_0^1', r'\vec B',
            r'\wedge', r'\dot{\vec B}', r'\;dt'
        ).move_to(tex_intb1)
        tex_intb2[0].set_color(GREEN)
        tex_intb2[4].set_color(GREEN)
        tex_intb2[6].set_color(GREEN)
        self.play(
            Write(tex_intb1),
            run_time=1
        )
        self.wait(0.5)
        map_intb12 = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 6:7}
        self.play(
            *get_tex_map_anims((tex_intb1,), (map_intb12,), tex_intb2),
            run_time=1
        )
        self.wait(1)

        tex_intc = MathTex(
            # 0                1     2               3
            r'\mathcal S_B', r'=', r'\frac{1}{2}', r'\int_0^1',
            # 4          5         6    7     8                    9
            r'\left(', r'\vec C', '+', 'b', r'\hat u_\parallel', r'\right)',
            # 10         11         12              13
            r'\wedge', r'\left(', r'\dot{\vec C}', '+',
            #14    15               16             17          18
            'b', r'\hat u_\perp', r'\dot\theta', r'\right)', r'\;dt'
        ).move_to(tex_intb2)
        tex_intc[0].set_color(GREEN)
        tex_intc[7].set_color(GREEN)
        tex_intc[14].set_color(GREEN)
        map_intc = {0:0, 1:1, 2:2, 3:3, 5:10, 7:18}
        map_b_int = {2:5, 3:6, 4:7, 5:8}
        map_bdot_int = {2:12, 3:13, 4:14, 5:15, 6:16}
        self.play(
            *get_tex_map_anims(
                (tex_intb2, tex_b, tex_bdot2),
                (map_intc, map_b_int, map_bdot_int),
                tex_intc,
                (False, True, True)
            ),
            run_time=1
        )
        self.wait(1)

        tex_exp1 = MathTex(
            # 0                1     2               3            4
            r'\mathcal S_B', r'=', r'\frac{1}{2}', r'\int_0^1', r'\left(',
            # 5          6          7                8
            r'\vec C', r'\wedge', r'\dot{\vec C}', r'\;+\;',
            # 9          10        11    12               13
            r'\vec C', r'\wedge', 'b', r'\hat u_\perp', r'\dot\theta',
            # 14       15    16                   17         18
            r'\;+\;', 'b', r'\hat u_\parallel', r'\wedge', r'\dot{\vec C}',
            # 19       20    21                   22
            r'\;+\;', 'b', r'\hat u_\parallel', r'\wedge',
            #23    25               25             26          27
            'b', r'\hat u_\perp', r'\dot\theta', r'\right)', r'\;dt',
            font_size=0.9*DEFAULT_FONT_SIZE
        ).move_to(tex_intc)
        tex_exp1[0].set_color(GREEN)
        tex_exp1[11].set_color(GREEN)
        tex_exp1[15].set_color(GREEN)
        tex_exp1[20].set_color(GREEN)
        tex_exp1[23].set_color(GREEN)
        tex_exp1[25].set_color(MAROON)
        map_exp1 = {
            0:0, 1:1, 2:2, 3:3, 4:4, 5:[5, 9], 6:8, 7:[15, 20], 8:[16, 21],
            10:[6, 10, 17, 22], 12:[7, 18], 13:19, 14:[11, 23], 15:[12, 24],
            16:[13, 25], 17:26, 18:27
        }
        self.play(
            *get_tex_map_anims((tex_intc,), (map_exp1,), tex_exp1),
            run_time=1
        )
        self.wait(1)

        brace = BraceLabel(tex_exp1[20:25], r'\equiv b^2')
        self.play(
            Write(brace),
            run_time=1
        )
        self.wait(1)
        tex_exp2 = MathTex(
            # 0                1     2               3            4
            r'\mathcal S_B', r'=', r'\frac{1}{2}', r'\int_0^1', r'\left[',
            # 5          6          7                8        9     10
            r'\vec C', r'\wedge', r'\dot{\vec C}', r'\;+\;', 'b', r'\left(',
            # 11         12         13               14             15
            r'\vec C', r'\wedge', r'\hat u_\perp', r'\dot\theta', r'\;+\;',
            # 16                   17         18               19
            r'\hat u_\parallel', r'\wedge', r'\dot{\vec C}', r'\right)',
            # 20       21      22             23          24
            r'\;+\;', 'b^2', r'\dot\theta', r'\right]', r'\;dt',
            font_size=0.9*DEFAULT_FONT_SIZE
        ).move_to(tex_exp1)
        tex_exp2[0].set_color(GREEN)
        tex_exp2[9].set_color(GREEN)
        tex_exp2[21].set_color(GREEN)
        tex_exp2[5:8].set_color(RED)
        tex_exp2[10:20].set_color(GOLD)
        tex_exp2[22].set_color(MAROON)
        map_exp12 = {
            0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:11, 10:12, 11:9,
            12:13, 13:14, 14:15, 15:9, 16:16, 17:17, 18:18, 19:20, 20:21,
            23:21, 25:22, 26:23, 27:24 
        }
        self.play(
            FadeOut(brace),
            *get_tex_map_anims((tex_exp1,), (map_exp12,), tex_exp2),
            run_time=1
        )
        self.wait(1)

        tex_exp3 = MathTex(
            # 0                1     2               3            4
            r'\mathcal S_B', r'=', r'\frac{1}{2}', r'\int_0^1', r'\vec C',
            # 5          6                7        8        9     10
            r'\wedge', r'\dot{\vec C}', r'\;dt', r'\;+\;', 'b', r'\cdot',
            # 11              12           13         14         15
            r'\frac{1}{2}', r'\int_0^1', r'\left(', r'\vec C', r'\wedge',
            # 16               17             18        19
            r'\hat u_\perp', r'\dot\theta', r'\;+\;', r'\hat u_\parallel',
            # 20         21               22          23       24       25
            r'\wedge', r'\dot{\vec C}', r'\right)', r'\;dt', r'\;+\;', 'b^2',
            # 26        27              28           29             30
            r'\cdot', r'\frac{1}{2}', r'\int_0^1', r'\dot\theta', r'\;dt',
            font_size=0.8*DEFAULT_FONT_SIZE
        ).move_to(tex_exp2)
        tex_exp3[0].set_color(GREEN)
        tex_exp3[9].set_color(GREEN)
        tex_exp3[25].set_color(GREEN)
        tex_exp3[2:8].set_color(RED)
        tex_exp3[11:24].set_color(GOLD)
        tex_exp3[28:].set_color(MAROON)
        map_exp23 = {
            0:0, 1:1, 2:[2, 11, 27], 3:[3, 12, 28], 5:4, 6:5, 7:6, 8:8, 9:9,
            10:13, 11:14, 12:15, 13:16, 14:17, 15:18, 16:19, 17:20, 18:21,
            19:22, 20:24, 21:25, 22:29, 24:[7, 23, 30]
        }
        self.remove(*tex_exp2)
        self.play(
            *get_tex_map_anims((tex_exp2,), (map_exp23,), tex_exp3),
            run_time=1
        )
        self.wait(1)

        brace1 = BraceLabel(tex_exp3[2:8], r'\mathcal S_C')
        brace1.set_color(RED)
        brace2 = BraceText(tex_exp3[11:24],
                           r'only depends on $\vec C$ and $\theta$')
        brace2.set_color(GOLD)
        self.play(
            Write(brace1),
            run_time=1
        )
        self.wait(1)
        self.play(
            Write(brace2),
            run_time=1
        )
        self.wait(2)

        tex_simp1 = MathTex(
            # 0                1     2               3    4     5
            r'\mathcal S_B', r'=', r'\mathcal S_C', '+', 'b', r'\mathcal I_C',
            #6    7       8         9
            '+', 'b^2', r'\cdot', r'\frac{1}{2}',
            # 10
            r'\left[\theta(1) - \theta(0)\right]',
            font_size=0.9*DEFAULT_FONT_SIZE
        ).move_to(tex_exp3)
        tex_simp1[0].set_color(GREEN)
        tex_simp1[4].set_color(GREEN)
        tex_simp1[7].set_color(GREEN)
        tex_simp1[2].set_color(RED)
        tex_simp1[5].set_color(GOLD)
        tex_simp1[10].set_color(MAROON)
        map_simp1 = {
            0:0, 1:1, 2:2, 3:2, 4:2, 5:2, 6:2, 7:2, 8:3, 9:4, 11:5,
            12:5, 13:5, 14:5, 15:5, 16:5, 17:5, 18:5, 19:5, 20:5, 21:5, 22:5,
            23:5, 24:6, 25:7, 26:8, 27:9, 28:10, 29:10, 30:10
        }
        self.play(
            FadeOut(brace1), FadeOut(brace2), Uncreate(box),
            tex_int.animate.set_color(WHITE),
            *get_tex_map_anims((tex_exp3,), (map_simp1,), tex_simp1),
            run_time=1
        )
        self.wait(1)

        tex_simp2 = MathTex(
            r'\mathcal S_B', r'=', r'\mathcal S_C', '+', 'b', r'\mathcal I_C',
            '+', 'b^2', r'\cdot', r'\frac{1}{2}', r'\cdot', '2', r'\pi',
            r'\left(\text{number of turns}\right)',
            font_size=0.9*DEFAULT_FONT_SIZE
        ).move_to(tex_simp1)
        tex_simp2[0].set_color(GREEN)
        tex_simp2[4].set_color(GREEN)
        tex_simp2[7].set_color(GREEN)
        tex_simp2[2].set_color(RED)
        tex_simp2[5].set_color(GOLD)
        tex_simp2[11:].set_color(MAROON)
        map_simp2 = {
            0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:[11, 12, 13]
        }
        self.remove(*tex_simp1)
        self.play(
            *get_tex_map_anims((tex_simp1,), (map_simp2,), tex_simp2),
            run_time=1
        )
        self.wait(1)

        tex_simp3 = MathTex(
            r'\mathcal S_B', r'=', r'\mathcal S_C', '+', 'b', r'\mathcal I_C',
            '+', 'b^2', r'\cdot', r'\frac{1}{2}', r'\cdot', '2', r'\pi', 'n',
            font_size=0.9*DEFAULT_FONT_SIZE
        ).move_to(tex_simp2)
        tex_simp3[0].set_color(GREEN)
        tex_simp3[4].set_color(GREEN)
        tex_simp3[7].set_color(GREEN)
        tex_simp3[2].set_color(RED)
        tex_simp3[5].set_color(GOLD)
        tex_simp3[11:].set_color(MAROON)
        map_simp3 = {k:k for k in range(13)}
        self.play(
            *get_tex_map_anims((tex_simp2,), (map_simp3,), tex_simp3),
            run_time=1
        )
        self.wait(1)

        tex_simp4 = MathTex(
            r'\mathcal S_B', '-', r'\mathcal S_C', r'=', 'b', r'\mathcal I_C',
            '+', 'b^2', r'\pi', 'n'
        ).move_to(tex_simp3)
        tex_simp4[0].set_color(GREEN)
        tex_simp4[4].set_color(GREEN)
        tex_simp4[7].set_color(GREEN)
        tex_simp4[2].set_color(RED)
        tex_simp4[5].set_color(GOLD)
        tex_simp4[8:].set_color(MAROON)
        map_simp4 = {0:0, 1:3, 2:2, 4:4, 5:5, 6:6, 7:7, 12:8, 13:9}
        self.play(
            *get_tex_map_anims((tex_simp3,), (map_simp4,), tex_simp4),
            run_time=1
        )
        self.wait(1)

        tex_dSb = MathTex(
            r'\Delta\mathcal S', r'=', 'b', r'\mathcal I_C',
            '+', 'b^2', r'\pi', 'n'
        ).move_to(tex_simp4)
        tex_dSb[2].set_color(GREEN)
        tex_dSb[5].set_color(GREEN)
        tex_dSb[3].set_color(GOLD)
        tex_dSb[6:].set_color(MAROON)
        map_simp5 = {0:0, 1:0, 2:0, 3:1, 4:2, 5:3, 6:4, 7:5, 8:6, 9:7}
        self.play(
            *get_tex_map_anims((tex_simp4,), (map_simp5,), tex_dSb),
            run_time=1
        )
        self.wait(1)

        tex_a = MathTex(
            r'\vec A', '=', r'\vec C', '-', 'a', r'\hat u_\parallel'
        ).next_to(tex_b, DOWN, MED_LARGE_BUFF)
        tex_a[0].set_color(BLUE)
        tex_a[3:5].set_color(BLUE)
        tex_bdot1 = MathTex(
            r'\dot{\vec B}', '=', r'\dot{\vec C}', '+', 'b',
            r'\dot{\hat u}_\parallel'
        ).move_to(2*RIGHT + 0.5*UP)
        tex_bdot1[0].set_color(GREEN)
        tex_bdot1[4].set_color(GREEN)
        tex_adot2 = MathTex(
            r'\dot{\vec A}', '=', r'\dot{\vec C}', '-', 'a',
            r'\hat u_\perp', r'\dot\theta'
        ).next_to(tex_bdot2, DOWN, MED_LARGE_BUFF)
        tex_adot2[0].set_color(BLUE)
        tex_adot2[3:5].set_color(BLUE)
        map_ba = {k:k for k in range(6)}
        map_badot = {k:k for k in range(7)}
        self.play(
            *get_tex_map_anims((tex_b,), (map_ba,), tex_a, (True,)),
            *get_tex_map_anims((tex_bdot2,), (map_badot,), tex_adot2, (True,)),
            tex_dSb.animate.shift(tex_a.get_center() - tex_b.get_center()),
            run_time=1
        )
        self.wait(1)

        tex_dSa = MathTex(
            r'\Delta\mathcal S', r'=', '-', 'a', r'\mathcal I_C',
            '+', 'a^2', r'\pi', 'n'
        ).next_to(tex_dSb, DOWN, MED_LARGE_BUFF)
        tex_dSa[2:4].set_color(BLUE)
        tex_dSa[6].set_color(BLUE)
        tex_dSa[4].set_color(GOLD)
        tex_dSa[7:].set_color(MAROON)
        map_dSba = {0:0, 1:1, 2:[2, 3], 3:4, 4:5, 5:6, 6:7, 7:8}
        self.play(
            *get_tex_map_anims((tex_dSb,), (map_dSba,), tex_dSa, (True,)),
            run_time=1
        )
        self.wait(1)

        tex_dSb2 = MathTex(
            'a', r'\Delta\mathcal S', r'=', 'a', 'b', r'\mathcal I_C',
            '+', 'a', 'b^2', r'\pi', 'n'
        ).move_to(tex_dSb)
        tex_dSb2[0].set_color(BLUE)
        tex_dSb2[3].set_color(BLUE)
        tex_dSb2[7].set_color(BLUE)
        tex_dSb2[4].set_color(GREEN)
        tex_dSb2[8].set_color(GREEN)
        tex_dSb2[5].set_color(GOLD)
        tex_dSb2[9:].set_color(MAROON)
        tex_dSa2 = MathTex(
            'b', r'\Delta\mathcal S', r'=', '-', 'b', 'a', r'\mathcal I_C',
            '+', 'b', 'a^2', r'\pi', 'n'
        ).move_to(tex_dSa)
        tex_dSa2[5].set_color(BLUE)
        tex_dSa2[9].set_color(BLUE)
        tex_dSa2[0].set_color(GREEN)
        tex_dSa2[4].set_color(GREEN)
        tex_dSa2[8].set_color(GREEN)
        tex_dSa2[6].set_color(GOLD)
        tex_dSa2[10:].set_color(MAROON)
        map_dSb12 = {0:1, 1:2, 2:4, 3:5, 4:6, 5:8, 6:9, 7:10}
        map_dSa12 = {0:1, 1:2, 2:3, 3:5, 4:6, 5:7, 6:9, 7:10, 8:11}
        self.play(
            *get_tex_map_anims((tex_dSb,), (map_dSb12,), tex_dSb2),
            *get_tex_map_anims((tex_dSa,), (map_dSa12,), tex_dSa2),
            run_time=1
        )
        self.wait(1)

        tex_dSb3 = MathTex(
            'a', r'\Delta\mathcal S', r'=', r'\mathcal I_C',
            '+', 'b', r'\pi', 'n'
        ).move_to(tex_dSb2)
        tex_dSb3[0].set_color(BLUE)
        tex_dSb3[5].set_color(GREEN)
        tex_dSb3[3].set_color(GOLD)
        tex_dSb3[6:].set_color(MAROON)
        tex_dSa3 = MathTex(
            'b', r'\Delta\mathcal S', r'=', '-', r'\mathcal I_C',
            '+', 'a', r'\pi', 'n'
        ).move_to(tex_dSa2)
        tex_dSa3[0].set_color(GREEN)
        tex_dSa3[6].set_color(BLUE)
        tex_dSa3[4].set_color(GOLD)
        tex_dSa3[7:].set_color(MAROON)
        map_dSb23 = {0:0, 1:1, 2:2, 5:3, 6:4, 8:5, 9:6, 10:7}
        map_dSa23 = {0:0, 1:1, 2:2, 3:3, 6:4, 7:5, 9:6, 10:7, 11:8}
        self.remove(*tex_dSb2)
        self.play(
            *get_tex_map_anims((tex_dSb2,), (map_dSb23,), tex_dSb3),
            *get_tex_map_anims((tex_dSa2,), (map_dSa23,), tex_dSa3),
            run_time=1
        )
        self.wait(1)

        tex_dS1 = MathTex(
            '(', 'b', '+', 'a', ')', r'\Delta\mathcal S', r'=',
            '(', 'b', '+', 'a', ')', r'\pi', 'n'
        ).move_to((tex_dSb3.get_center() + tex_dSa3.get_center())/2)
        tex_dS1[3].set_color(BLUE)
        tex_dS1[10].set_color(BLUE)
        tex_dS1[1].set_color(GREEN)
        tex_dS1[8].set_color(GREEN)
        tex_dS1[12:].set_color(MAROON)
        map_dSa1 = {0:3, 1:5, 2:6, 5:8, 6:12, 7:13}
        map_dSb1 = {0:1, 1:5, 2:6, 6:10, 7:12, 8:13}
        self.play(
            *get_tex_map_anims(
                (tex_dSb3, tex_dSa3), (map_dSa1, map_dSb1), tex_dS1
            ),
            run_time=1
        )
        self.wait(1)

        tex_dS2 = MathTex(
            r'\Delta\mathcal S', r'=', r'\pi', 'n'
        ).move_to(tex_dS1)
        tex_dS2[2:].set_color(MAROON)
        map_dS2 = {5:0, 6:1, 12:2, 13:3}
        self.remove(*tex_dS1)
        self.play(
            *get_tex_map_anims((tex_dS1,), (map_dS2,), tex_dS2),
            run_time=1
        )
        box = SurroundingRectangle(tex_dS2, buff=MED_SMALL_BUFF, color=TEAL)
        self.play(
            Create(box),
            tex_dS2.animate.set_color(TEAL),
            run_time=1
        )
        self.wait(1)

        self.play(
            Unwrite(tex_int),
            Unwrite(tex_b), Unwrite(tex_bdot2),
            Unwrite(tex_a), Unwrite(tex_adot2), 
            Uncreate(box), Unwrite(tex_dS2),
            *[FadeOut(mob) for mob in chord],
            *[FadeOut(mob) for mob in basis],
            *[FadeOut(mob) for mob in labels],
            run_time=0.5
        )
        self.wait(0.1)
