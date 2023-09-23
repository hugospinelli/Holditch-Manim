import numpy as np

from manim import *


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
