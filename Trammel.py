import numpy as np

from manim import *

from HolditchMobjects import ChordVGroup

norm = np.linalg.norm


COLOR_FILL = '#0d0040'
COLOR_LOCUS = '#4169e1'


class Trammel:
    def __init__(
            self, vt: ValueTracker, center: np.ndarray, angle: float,
            a: float, b: float, v: float, h: float = 0
    ):
        self.vt = vt
        self.time = vt.get_value()
        self.center = center
        self.angle = angle
        self.a = a
        self.b = b
        self.v = v
        self.h = h
        L = self.a + self.b
        self.x = -L
        self.y = 0
        self._target_chord_angle: float | None = None
        self._stopped = False

        self.line_x = Line(1.1*L*LEFT, 1.1*L*RIGHT).set_z_index(1)
        self.line_y = Line(1.1*L*LEFT, 1.1*L*RIGHT).set_z_index(1)
        self.line_x.move_to(center)
        self.line_y.rotate(angle).move_to(center)
        self.chord = ChordVGroup(
            *self.get_ps(), ab_visible=self.h!=0).set_z_index(2)
        self._ellipse_inputs = 4*(None,)  # (angle, a, b, h)
        self.ellipse = self.get_ellipse()

    def stop_at_chord_angle(self, angle: float | None):
        self._target_chord_angle = angle

    def pause(self):
        self._stopped = True

    def resume(self):
        self._stopped = False

    def _get_lims(self):
        L = self.a + self.b
        c = np.cos(self.angle)
        xpp = L/np.sqrt(2*(1 + c))
        xpm = L/np.sqrt(2*(1 - c))
        xmp = -xpp
        xmm = -xpm
        return xpp, xpm, xmp, xmm

    @property
    def T(self):
        xpp, xpm, _, _ = self._get_lims()
        return 4*(xpp + xpm)/self.v

    def _get_kt(self, x, y):
        xpp, xpm, xmp, xmm = self._get_lims()
        if x <= y < -x:
            k = 0
            t = (y - xmm)/self.v
        elif -y <= x < y:
            k = 1
            t = (x - xmp)/self.v
        elif -x < y <= x:
            k = 2
            t = (xpm - y)/self.v
        else:
            k = 3
            t = (xpp - x)/self.v
        return k, t

    def _get_xy(self, k, t):
        k = int(round(k))
        L = self.a + self.b
        xpp, xpm, xmp, xmm = self._get_lims()
        s, c = np.sin(self.angle), np.cos(self.angle)

        def f(x0, y):
            d = np.sqrt(L**2 - (y*s)**2)
            x1 = y*c - d
            x2 = y*c + d
            if abs(x0 - x1) < abs(x0 - x2):
                return x1
            return x2

        if k == 0:
            y = xmm + self.v*t
            x = f(self.x, y)
        elif k == 1:
            x = xmp + self.v*t
            y = f(self.y, x)
        elif k == 2:
            y = xpm - self.v*t
            x = f(self.x, y)
        else:  # k == 3
            x = xpp - self.v*t
            y = f(self.y, x)
        return x, y

    def update_xy(self, dt, silent=False):
        if self._stopped:
            return
        k, t0 = self._get_kt(self.x, self.y)
        dk, t = divmod(t0 + dt, self.T/4)
        k = (k + dk) % 4
        x, y = self._get_xy(k, t)
        if self._target_chord_angle is None:
            self.x, self.y = x, y
            return
        a1 = self.get_chord_angle()
        a2 = self._get_chord_angle(x, y)
        da1 = (a1 - self._target_chord_angle + PI) % TAU - PI
        da2 = (a2 - self._target_chord_angle + PI) % TAU - PI
        if da1 == 0 or not (-PI*PI/4 < da1*da2 <= 0):
            self.x, self.y = x, y
            return
        if not silent:
            print(f'\nStopped Trammel chord at {self.vt.get_value() = }')
        s, c = np.sin(self.angle), np.cos(self.angle)
        s2 = np.sin(self._target_chord_angle)
        c2 = np.cos(self._target_chord_angle)
        L = self.a + self.b
        self.y = L*s2/s
        self.x = self.y*c - L*c2
        self._stopped = True

    def _get_chord_angle(self, x, y):
        return np.arctan2(y*np.sin(self.angle), y*np.cos(self.angle) - x)

    def get_chord_angle(self):
        return self._get_chord_angle(self.x, self.y)

    def get_ps(self):
        s, c = np.sin(self.angle), np.cos(self.angle)
        pa = self.x*RIGHT + self.center
        pb = self.y*c*RIGHT + self.y*s*UP + self.center
        ab = pb - pa
        pc = pa + self.a*normalize(ab) + self.h*get_unit_normal(ab, IN)
        return pa, pb, pc

    def get_ellipse_params(self):
        L = self.a + self.b
        s, c = np.sin(self.angle), np.cos(self.angle)
        k = c/s
        A = 1/self.b**2
        B = (k*L/(self.a*self.b))**2
        C = 1/self.a**2
        D = (A + B + C)/2
        E = (A - B - C)/2
        F = np.sqrt(A*B + E**2)
        X = D - F
        Y = D + F
        Z = E/F
        a = np.sqrt(1/X)
        b = np.sqrt(1/Y)
        beta = np.arcsin(np.sqrt((1 + Z)/2))
        if k < 0:
            beta = np.pi - beta
        return a, b, beta

    def get_ellipse(self, min_length=0.05):
        inputs = (self.angle, self.a, self.b, self.h)
        if self._ellipse_inputs == inputs:
            return self.ellipse
        if self.h == 0:
            a, b, beta = self.get_ellipse_params()
            ellipse = Ellipse(
                2*a, 2*b,
                fill_color=COLOR_FILL,
                fill_opacity=0,
                stroke_color=COLOR_LOCUS,
                stroke_opacity=1,
            ).set_z_index(0)
            return ellipse.rotate(beta).move_to(self.center)
        self._ellipse_inputs = inputs
        stopped = self._stopped
        self._stopped = False
        target_chord_angle = self._target_chord_angle
        self._target_chord_angle = self.get_chord_angle()
        pcs = []
        dt = min_length/self.v
        while not self._stopped:
            self.update_xy(dt, silent=True)
            _, _, pc = self.get_ps()
            pcs.append(pc)
        self._stopped = stopped
        self._target_chord_angle = target_chord_angle
        ellipse = Polygon(
            *pcs,
            fill_color=COLOR_FILL,
            fill_opacity=0,
            stroke_color=COLOR_LOCUS,
            stroke_opacity=1,
        ).set_z_index(0)
        return ellipse

    def update(self):
        time = self.vt.get_value()
        dt = time - self.time
        self.time = time
        self.update_xy(dt)

    def update_line_y(self):
        L = self.a + self.b
        line_y = Line(1.1*L*LEFT, 1.1*L*RIGHT).set_z_index(1)
        line_y = line_y.rotate(self.angle).move_to(self.center)
        self.line_y.become(line_y)

    def update_chord(self):
        self.chord.become(ChordVGroup(
            *self.get_ps(), ab_visible=self.h!=0).set_z_index(2))

    def update_ellipse(self):
        self.ellipse.become(self.get_ellipse())

    def get_tex_formulas(self):
        a2_b2 = (
            r"a',\,b' = \frac{\sqrt2 a b \sin(\alpha)}{\sqrt{a^2 + b^2 "
            r'+ 2ab\cos^2(\alpha) \mp \cos^2(\alpha)\sqrt{4 a^2 \left(a '
            r'+ b\right)^2 \tan^2(\alpha) + \left(\left(a + b\right)^2 '
            r'+ \left(b^2 - a^2\right) \tan^2(\alpha) \right)^2}}}'
        )
        beta = (
            r'\beta = \arcsin\left(\frac{\sqrt2}{2}\cdot\sqrt{1 + '
            r'\frac{a - b - 2a\cos^2(\alpha)}{\sqrt{\left(a + b\right)^2 '
            r'- 4ab\sin^2(\alpha)}}}\right)'
        )
        return beta, a2_b2


class Test(Scene):
    def construct(self):
        A, B = 1.5, 2.5
        v = 4.0
        time = 0
        vt = ValueTracker(time)
        trammel = Trammel(vt, 0.5*LEFT + 0.3*DOWN, 90*DEGREES, A, B, v)
        vt.add_updater(lambda _: trammel.update())
        trammel.line_y.add_updater(lambda _: trammel.update_line_y())
        trammel.chord.add_updater(lambda _: trammel.update_chord())
        trammel.ellipse.add_updater(lambda _: trammel.update_ellipse())

        def update_angle():
            angle = (90 + 40*np.sin(TAU*vt.get_value()/10))*DEGREES
            trammel.angle = angle
        vt.add_updater(lambda _: update_angle())

        self.play(FadeIn(trammel.line_x),
                  FadeIn(trammel.line_y),
                  FadeIn(trammel.chord),
                  FadeIn(trammel.ellipse))

        dt = 10
        self.play(vt.animate.set_value(time := time + dt),
                  run_time=dt, rate_func=linear)
        self.wait()
