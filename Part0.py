# DO NOT USE!!!
# Poor quality code

import os

from collections import defaultdict

from manim import *
import numpy as np

IMAGES_DIR = 'Images'
_CC0_PATH = os.path.join(IMAGES_DIR, 'CC0', 'cc-zero.svg')
CC0 = SVGMobject(_CC0_PATH, height=0.3).to_edge(DR, buff=0.05)

COLOR_BLUE = '#0000cd'
COLOR_ROYAL = '#4169e1'
COLOR_RED = '#a52a2a'
COLOR_FILL = '#0d0040'


norm = np.linalg.norm


def rotate(v, angle):
    m = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    if len(v) == 2:
        return m[:2,:2]@v
    return m@v


def get_angle(v):
    return np.arctan2(v[1], v[0])


def unitary(v):
    return v/norm(v)


template = TexTemplate(preamble='\\usepackage{amsmath,cancel}')


def get_combined_tex_anims(tex1, tex2, target_tex, map1, map2):
    fixed = set(range(len(target_tex)))
    combined = defaultdict(list)
    for tex_, map_ in ((tex1, map1), (tex2, map2)):
        for k, targets in map_.items():
            if isinstance(targets, int):
                targets = [targets]
            fixed = fixed.difference(targets)
            for kt in targets:
                combined[kt].append(tex_[k])

    anims = []
    v12 = tex2.get_center() - tex1.get_center()
    unused1 = set(range(len(tex1))) - set(map1.keys())
    unused2 = set(range(len(tex2))) - set(map2.keys())
    anims.extend(FadeOut(tex1[k].copy(),
                         target_position=tex1[k].get_center() + v12/2)
                 for k in unused1)
    anims.extend(FadeOut(tex2[k].copy(),
                         target_position=tex2[k].get_center() - v12/2)
                 for k in unused2)

    anims.extend(FadeIn(target_tex[k]) for k in fixed)
    for kt, tex_list in combined.items():
        anims.append(TransformFromCopy(VGroup(*tex_list), target_tex[kt]))
    return anims


def indicate_line(group, color, revert_color=True):
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


def indicate_tex(group, color, revert_color=True):
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


def indicate_group(group, color, revert_color=True):
    lines = []
    texs = []
    for part in group.submobjects:
        if isinstance(part, MathTex):
            texs.append(part)
        else:
            lines.append(part)
    return AnimationGroup(indicate_line(lines, color, revert_color),
                          indicate_tex(texs, color, revert_color))


class Scene01(Scene):
    def fastplay(self, *args, **kwargs):
        kwargs['run_time'] = 0.7
        self.play(*args, **kwargs)

    def construct(self):
        self.add(CC0)

        a, b = 1, 2

        pa = Dot(point=LEFT*(a+b)/2, stroke_width=2,
                 fill_color=COLOR_RED, z_index=2)
        pb = Dot(point=RIGHT*(a+b)/2, stroke_width=2,
                 fill_color=COLOR_RED, z_index=2)
        pc = Dot(point=LEFT*(a+b)/2 + RIGHT*a, stroke_width=2,
                 fill_color=COLOR_BLUE, z_index=2)
        line_ac = Line(start=[LEFT*(a+b)/2], end=[RIGHT*(a-b)/2],
                       color=RED, z_index=1)
        line_bc = Line(start=[RIGHT*(a+b)/2], end=[RIGHT*(a-b)/2],
                       color=RED, z_index=1)
        text_A = Tex('$A$')
        text_B = Tex('$B$')
        text_C = Tex('$C$')
        text_A.next_to(pa, UP)
        text_B.next_to(pb, UP)
        text_C.next_to(pc, UP)
        self.fastplay(Create(pa), Create(text_A),
                      Create(pb), Create(text_B),
                      Create(line_ac), Create(line_bc))

        brace_a = BraceLabel(Line(pa.get_center(), pc.get_center()), 'a')
        brace_b = BraceLabel(Line(pc.get_center(), pb.get_center()), 'b')
        self.fastplay(FadeIn(pc), Create(text_C),
                      FadeIn(brace_a), FadeIn(brace_b))

        group = Group(line_ac, line_bc, pa, pb, pc,
                      text_A, text_B, text_C,
                      brace_a, brace_b)
        self.fastplay(group.animate.shift(DOWN))

        text = Tex(r'Normalize to $ab = 1$').shift(UP)
        self.fastplay(Create(text))
        self.wait(1.5)


        ### First circle ###
        r = 2.5
        c = [0,0,0]
        path = Circle(radius=r, color=WHITE, z_index=1)
        path.move_to(c)
        slider = VGroup(line_ac, line_bc, pa, pb, pc)
        slider.set_z_index(2)
        remove_group = Group(text, text_A, text_B, text_C, brace_a, brace_b)
        pa_center = pa.get_center()
        slider_angle = np.arcsin((a + b)/(2*r))
        pa_start = path.point_at_angle(-TAU/4)
        self.play(Create(path),
                  slider.animate.rotate(
                      angle=slider_angle, about_point=pa_center
                  ).shift(pa_start - pa_center),
                  FadeOut(remove_group))
        self.wait()

        angle = ValueTracker(0)
        v = pc.get_center() - c
        rv = norm(v)
        start_angle = get_angle(v)
        locus = Arc(start_angle=start_angle, radius=rv,
                    color=COLOR_ROYAL, z_index=1)
        locus.move_to(c)
        def update_arc(arc):
            arc.angle = angle.get_value()
            arc.generate_points()
        locus.add_updater(update_arc)
        self.add(angle, locus)
        self.play(Rotate(slider, angle=TAU, about_point=c),
                  angle.animate.set_value(TAU),
                  run_time=2,
                  rate_func=linear)

        locus.clear_updaters()
        ring = Annulus(inner_radius=rv, outer_radius=r,
                       color=COLOR_FILL, z_index=0)
        ring.move_to(c)
        self.fastplay(FadeIn(ring))

        text = Tex(r'Area = $\pi$')
        text.move_to(c)
        arrow = CurvedArrow(text.get_corner(RIGHT) + 0.3*RIGHT,
                            c + RIGHT*((rv + r)/2),
                            radius=-3)
        arrow.set_z_index(3)
        self.fastplay(Create(text), FadeIn(arrow))
        self.wait()

        self.fastplay(FadeOut(locus), FadeOut(ring),
                      FadeOut(text), FadeOut(arrow))



        ### Second circle ###
        old_path = path
        r = 3.5
        path = Circle(radius=r, color=WHITE, z_index=1)
        path.move_to(c)
        pa_center = pa.get_center()
        slider_angle = np.arcsin((a + b)/(2*r)) - slider_angle
        pa_start = path.point_at_angle(-TAU/4)
        self.play(ReplacementTransform(old_path, path),
                  slider.animate.rotate(
                      angle=slider_angle, about_point=pa_center
                      ).shift(pa_start - pa_center))

        angle = ValueTracker(0)
        v = pc.get_center() - c
        rv = norm(v)
        start_angle = get_angle(v)
        locus = Arc(start_angle=start_angle, radius=rv,
                    color=COLOR_ROYAL, z_index=1)
        locus.move_to(c)
        def update_arc(arc):
            arc.angle = angle.get_value()
            arc.generate_points()
        locus.add_updater(update_arc)
        self.add(angle, locus)
        self.play(Rotate(slider, angle=TAU, about_point=c),
                  angle.animate.set_value(TAU),
                  run_time=2,
                  rate_func=linear)

        locus.clear_updaters()
        ring = Annulus(inner_radius=rv, outer_radius=r,
                       color=COLOR_FILL, z_index=0)
        ring.move_to(c)
        self.fastplay(FadeIn(ring))

        text = Tex(r'Area = $\pi$')
        text.move_to(c)
        arrow = CurvedArrow(text.get_corner(RIGHT) + 0.3*RIGHT,
                            c + RIGHT*((rv + r)/2),
                            radius=-3)
        arrow.set_z_index(3)
        self.fastplay(Create(text), FadeIn(arrow))
        self.wait()

        text2 = Tex(r'It does not depend\\on the radius!')
        text2.move_to(text.get_corner(DOWN) + DOWN)
        self.fastplay(Create(text2))
        self.wait(1.5)

        self.fastplay(FadeOut(text), FadeOut(arrow),
                      FadeOut(text2))

        ### Proof for circles ###
        r = 2.5
        c = 4.2*LEFT + UP
        old_path = path
        old_locus = locus
        old_ring = ring
        path = Circle(radius=r, color=WHITE, z_index=1)
        path.move_to(c)
        ab_angle = 2*np.arcsin((a + b)/(2*r))
        ppa = c + r*DOWN
        ppb = c + rotate(r*DOWN, ab_angle)
        ppc = ppa + a*unitary(ppb - ppa)

        v = ppc - c
        rv = norm(v)
        start_angle = get_angle(v)
        pa_center = pa.get_center()
        vab = pb.get_center() - pa_center
        old_angle = get_angle(vab)
        slider_angle = ab_angle/2 - old_angle
        locus = Arc(start_angle=start_angle, radius=rv, angle=TAU,
                    color=COLOR_ROYAL, z_index=1)
        locus.move_to(c)
        ring = Annulus(inner_radius=rv, outer_radius=r,
                       color=COLOR_FILL, z_index=0)
        ring.move_to(c)
        slider.set_z_index(3)
        self.play(ReplacementTransform(old_path, path),
                  ReplacementTransform(old_locus, locus),
                  ReplacementTransform(old_ring, ring),
                  slider.animate.rotate(
                      angle=slider_angle, about_point=pa_center
                      ).shift(ppa - pa_center))
        center_dot = Dot(point=c)
        line_a = Line(c, pa.get_center(), z_index=2)
        line_b = Line(c, pb.get_center(), z_index=2)
        line_c = Line(c, pc.get_center(), z_index=2)
        angle_a = Angle(line_ac, line_a, radius=0.3, quadrant=(1, -1))
        angle_b = Angle(line_b, line_bc, radius=0.3, quadrant=(-1, 1))
        text_a = MathTex('R').next_to(line_a.get_center(), LEFT)
        text_b = MathTex('R').next_to(line_b.get_center(), UP)
        text_c = MathTex('r').next_to(line_c.get_center(), RIGHT)
        text_ac = MathTex('a').next_to(line_ac.get_center(), DR, buff=0.24)
        text_bc = MathTex('b').next_to(line_bc.get_center(), DR, buff=0.06)
        text_ta = MathTex(r'\theta').move_to(pa.get_center() + (.3, .6, 0))
        text_tb = MathTex(r'\theta').move_to(pb.get_center() + (-.6, -.1, 0))
        for elem in [text_a, text_b, text_c,
                     text_ac, text_bc, text_ta, text_tb]:
            elem.set_z_index(5)
        self.fastplay(FadeIn(center_dot),
                      FadeIn(line_a), FadeIn(line_b), FadeIn(line_c),
                      FadeIn(text_a), FadeIn(text_b), FadeIn(text_c),
                      FadeIn(text_ac), FadeIn(text_bc))
        self.wait()

        self.play(FadeIn(angle_a), FadeIn(angle_b),
                  FadeIn(text_ta), FadeIn(text_tb))
        self.wait()

        header = Tex('Law of cosines:').move_to(2.5*UP + 3.1*RIGHT)
        eqs = [
            'r ^2 = R ^2 + a ^2 -2 R a \\cos \\theta',
            'r ^2 = R ^2 + b ^2 -2 R b \\cos \\theta',
            'r^2 = R^2 + a ^2 -2R a \\cos\\theta',
            'r^2 = R^2 + b ^2 -2R b \\cos\\theta',
            'b r^2 = b R^2 + a ^2 b -2R a b \\cos\\theta',
            'a r^2 = a R^2 + a b ^2 -2R a b \\cos\\theta',
            '( b - a ) r^2 = ( b - a ) R^2 + ( a - b ) a b',
            '( b - a ) r^2 = ( b - a ) R^2 + ( a - b ) ab',
            'r^2 = R^2 - ab',
            '\\pi R^2 - \\pi r^2  = \\pi ab',
            '\\text{Area} = \\pi',
        ]
        eqs = [MathTex(*eq.split(' ')) for eq in eqs]
        eqs[0].next_to(header, DOWN, buff=0.5)
        eqs[1].next_to(eqs[0], DOWN)
        eqs[2].move_to(eqs[0])
        eqs[3].move_to(eqs[1])
        eqs[4].move_to(eqs[0])
        eqs[5].move_to(eqs[1])
        eqs[6].next_to(eqs[5], DOWN)
        eqs[7].move_to(eqs[6])
        eqs[8].next_to(eqs[7], DOWN)
        eqs[9].next_to(eqs[8], DOWN)
        eqs[10].next_to(eqs[9], DOWN, buff=0.5)

        self.play(Create(header))

        triangle_a = VGroup(line_a, line_c, line_ac,
                            text_a, text_c, text_ac,
                            angle_a, text_ta)
        pa.set_z_index(4)
        pc.set_z_index(4)
        pb.set_z_index(4)
        center_dot.set_z_index(4)
        self.play(indicate_group(triangle_a, TEAL, revert_color=False))
        self.play(TransformMatchingTex(
            Group(text_a, text_c, text_ac, text_ta).copy(),
            eqs[0]
        ), run_time=2)
        self.wait()

        triangle_b = VGroup(line_b, line_c, line_bc,
                            text_b, text_c, text_bc,
                            angle_b, text_tb)
        self.play(FadeToColor(triangle_a - line_c - line_ac, WHITE),
                  FadeToColor(line_ac, COLOR_RED),
                  indicate_group(triangle_b, TEAL, revert_color=False))
        self.play(TransformMatchingTex(
            Group(text_b, text_c, text_bc, text_tb).copy(),
            eqs[1]
        ), run_time=2)
        self.wait()

        times_b = MathTex('(\\times', 'b', ')').next_to(eqs[0], LEFT, buff=0.5)
        times_a = MathTex('(\\times', 'a', ')').next_to(times_b, DOWN)
        self.play(FadeToColor(triangle_b - line_bc, WHITE),
                  FadeToColor(line_bc, COLOR_RED),
                  Create(times_b), Create(times_a))
        self.wait()

        self.remove(eqs[0], eqs[1])
        self.add(eqs[2], eqs[3])
        self.play(FadeOut(times_a), FadeOut(times_b),
                  TransformMatchingTex(eqs[2], eqs[4]),
                  TransformMatchingTex(eqs[3], eqs[5]))
        self.wait()

        brace = BraceLabel(Group(eqs[4], eqs[5]), '-', LEFT)
        self.play(FadeIn(brace))
        self.wait()

        # b r^2 = b R^2 + a ^2 b -2R a b  \\cos\\theta
        # 0 1   2 3 4   5 6 7  8 9  10 11 12
        #
        # a r^2 = a R^2 + a b ^2 -2R a b  \\cos\\theta
        # 0 1   2 3 4   5 6 7  8 9  10 11 12
        #
        # ( b - a ) r^2 = ( b - a  )  R^2 +  (  a  -  b  )  a  b
        # 0 1 2 3 4 5   6 7 8 9 10 11 12  13 14 15 16 17 18 19 20
        #
        map1 = {0:1, 1:5, 2:6, 3:8, 4:12, 5:13, 6:[15,19], 8:20}
        map2 = {0:3, 1:5, 2:6, 3:10, 4:12, 5:13, 6:19, 7:[17,20]}
        self.play(FadeOut(brace),
                  *get_combined_tex_anims(eqs[4], eqs[5], eqs[6], map1, map2))
        self.wait()

        self.remove(eqs[6])
        self.play(TransformMatchingTex(eqs[7].copy(), eqs[8]))
        self.wait()
        self.play(TransformMatchingTex(eqs[8].copy(), eqs[9]))
        self.wait()
        eqs[10][0].set_color(COLOR_ROYAL)
        self.play(FadeToColor(eqs[9][:5], COLOR_ROYAL),
                  Create(eqs[10]))
        self.wait()

        mobs = [path, locus, ring, slider, center_dot,
                line_a, line_b, line_c]
        texts = [header, angle_a, angle_b, text_a, text_b, text_c,
                 text_ac, text_bc, text_ta, text_tb]
        texts.extend(eqs[4:7] + eqs[8:])
        self.play(
            *[FadeOut(mob) for mob in mobs],
            *[Unwrite(text, reverse=False) for text in texts],
            run_time=1
        )
        self.wait(0.1)
