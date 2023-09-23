import os

from manim import *
import numpy as np

from HolditchMobjects import Holditch


IMAGES_DIR = 'Images'
_CC0_PATH = os.path.join(IMAGES_DIR, 'CC0', 'cc-zero.svg')
CC0 = SVGMobject(_CC0_PATH, height=0.3, z_index=100).to_edge(DR, buff=0.05)


norm = np.linalg.norm


class Scene14(Scene):
    def construct(self):
        self.add(CC0)

        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'heart', center=DOWN, scale=0.04, speed=100)
        self.play(
            *[FadeIn(mob) for mob in holditch.chord],
            FadeIn(holditch.curve),
            run_time=0.5
        )
        self.add(holditch.group)

        text = Tex('We did not assume that the curve was convex.')
        text.move_to(3*UP)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 4.5
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 0.5
        self.play(
            Unwrite(text, reverse=False),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        text = Tex('The red area is negative.').move_to(text)
        dt = 1
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 1.5
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 0.5
        self.play(
            Unwrite(text, reverse=False),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        text = Tex('The total area is still $\pi$.').move_to(text)
        dt = 1
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 2
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(12.5, 13)
        holditch.group.add_updater(chord_opacity_updater)
        dt = 0.5
        self.play(
            vt.animate.set_value(time := time + dt),
            Unwrite(text, reverse=False),
            FadeOut(holditch.strip),
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch.group)
        self.wait(0.1)


class Scene15(Scene):
    def construct(self):
        self.add(CC0)

        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'triangle', center=0.5*DOWN,
                            scale=0.035, speed=150)
        self.play(
            *[FadeIn(mob) for mob in holditch.chord],
            FadeIn(holditch.curve),
            run_time=0.5
        )
        self.add(holditch.group)

        text = Tex('If the areas overlap, they must be added.')
        text.move_to(3*UP)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 6.5
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(8, 8.5)
        holditch.group.add_updater(chord_opacity_updater)
        dt = 0.5
        self.play(
            vt.animate.set_value(time := time + dt),
            Unwrite(text, reverse=False),
            FadeOut(holditch.strip),
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch.group)
        self.wait(0.1)


class Scene16(Scene):
    def construct(self):
        self.add(CC0)

        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'double_loop', center=0.5*DOWN,
                            scale=0.035, speed=150)
        self.play(
            *[FadeIn(mob) for mob in holditch.chord],
            FadeIn(holditch.curve),
            run_time=0.5
        )
        self.add(holditch.group)

        text = Tex(r'If the chord spins twice, the area will be $2\pi$.')
        text.move_to(3*UP)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 6.5
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(8, 8.5)
        holditch.group.add_updater(chord_opacity_updater)
        dt = 0.5
        self.play(
            vt.animate.set_value(time := time + dt),
            Unwrite(text, reverse=False),
            FadeOut(holditch.strip),
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch.group)
        self.wait(0.1)


class Scene17(Scene):
    def construct(self):
        self.add(CC0)

        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'lemniscate', center=0.5*DOWN,
                            scale=0.05, speed=150)
        self.play(
            *[FadeIn(mob) for mob in holditch.chord],
            FadeIn(holditch.curve),
            run_time=0.5
        )
        self.add(holditch.group)

        text = Tex(r'If it spins back without completing a\\'
                   'full turn, the area will be $0$.')
        text.move_to(3*UP)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 5.5
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(7, 7.5)
        holditch.group.add_updater(chord_opacity_updater)
        dt = 0.5
        self.play(
            vt.animate.set_value(time := time + dt),
            Unwrite(text, reverse=False),
            FadeOut(holditch.strip),
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch.group)
        self.wait(0.1)


class Scene18(Scene):
    def construct(self):
        self.add(CC0)

        A, B = 1.6, 2.4

        time = 0
        vt = ValueTracker(time)

        t = np.linspace(-(2/3)*2*np.pi, (1/3)*2*np.pi, num=300, endpoint=False)
        m = (1 - 1e-9)*(A + B)/4
        xs = -2*m*np.sin(t) + m*np.sin(2*t)
        ys = 2*m*np.cos(t) + m*np.cos(2*t)
        ps = np.array([xs, ys, 0*t]).T
        holditch1 = Holditch(
            vt, ps, center=3.5*LEFT + DOWN, a=A, b=B, speed=16*m/3,
            starting_index=0
        )
        holditch2 = Holditch(
            vt, ps, center=3.5*RIGHT + DOWN, a=A, b=B, speed=16*m/3,
            starting_index=0, b_index=1, shortcut=True
        )

        self.play(
            *[FadeIn(mob) for mob in holditch1.chord],
            FadeIn(holditch1.curve),
            *[FadeIn(mob) for mob in holditch2.chord],
            FadeIn(holditch2.curve),
            run_time=0.5
        )
        self.add(holditch1.group, holditch2.group)

        text = Tex('The path might not be unique.')
        text.move_to(3*UP)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        dt = 6.5
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch1.chord.opacity = 1 - (t - t1)/(t2 - t1)
                holditch2.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(8, 8.5)
        holditch1.group.add_updater(chord_opacity_updater)
        holditch2.group.add_updater(chord_opacity_updater)
        dt = 0.5
        self.play(
            vt.animate.set_value(time := time + dt),
            Unwrite(text, reverse=False),
            FadeOut(holditch1.strip),
            FadeOut(holditch1.curve),
            FadeOut(holditch1.locus),
            FadeOut(holditch2.strip),
            FadeOut(holditch2.curve),
            FadeOut(holditch2.locus),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch1.group, *holditch2.group)
        self.wait(0.1)


class Scene19(Scene):
    def construct(self):
        self.add(CC0)

        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'dragon', center=0.5*DOWN,
                            scale=0.035, speed=120)
        self.play(
            *[FadeIn(mob) for mob in holditch.chord],
            FadeIn(holditch.curve),
            run_time=0.5
        )
        self.add(holditch.group)

        text = Tex('But the curve can be arbitrarily complex.')
        text.move_to(3*UP)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 7
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 0.5
        self.play(
            Unwrite(text, reverse=False),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        text = Tex(r'It only needs to allow for\\'
                   'the chord to complete a full loop.').move_to(text)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 7
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(17.5, 18)
        holditch.group.add_updater(chord_opacity_updater)
        dt = 0.5
        self.play(
            vt.animate.set_value(time := time + dt),
            Unwrite(text, reverse=False),
            FadeOut(holditch.strip),
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch.group)
        self.wait(0.1)


class Scene20(Scene):
    def construct(self):
        self.add(CC0)

        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'beluga', ORIGIN,
                            scale=0.04, speed=100, build_path=False)
        text = Tex('Can you guess what path the chord will take?')
        text.move_to(3*UP)
        self.play(
            Write(text),
            *[FadeIn(mob) for mob in holditch.chord],
            FadeIn(holditch.curve),
        )
        self.add(holditch.group)
        self.wait(2)

        self.play(
            Unwrite(text, reverse=False),
            run_time=0.5
        )

        text = Tex("Here's a hint:").move_to(text)
        holditch.chord.toggle_lines_visibility()
        dt = 1
        self.play(
            Write(text),
            run_time=dt
        )
        loop_time = 12.8
        dt = loop_time
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        holditch.chord.toggle_lines_visibility()
        holditch.build_path = True
        dt = 0.5
        self.play(
            Unwrite(text, reverse=False),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = loop_time + 1.5
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        dt = 0.5
        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(time - dt, time)
        holditch.group.add_updater(chord_opacity_updater)
        self.play(
            vt.animate.set_value(time := time + dt),
            Unwrite(text, reverse=False),
            FadeOut(holditch.strip),
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch.group)
        self.wait(0.1)


class Scene21(Scene):
    def construct(self):
        self.add(CC0)

        time = 0
        vt = ValueTracker(time)
        holditch = Holditch(vt, 'pi', center=0.5*DOWN,
                            scale=0.035, speed=120)
        self.play(
            *[FadeIn(mob) for mob in holditch.chord],
            FadeIn(holditch.curve),
            run_time=0.5
        )
        self.add(holditch.group)

        text = Tex(r'For some curves, finding a way around\\'
                   'can be quite a challenge!')
        text.move_to(3.25*UP)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        dt = 4
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 0.5
        self.play(
            Unwrite(text, reverse=False),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        text = Tex(r'But the total area is still $\pi$.').move_to(text)
        dt = 1.5
        self.play(
            Write(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 4
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 0.5
        self.play(
            Unwrite(text),
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )
        dt = 30
        self.play(
            vt.animate.set_value(time := time + dt),
            run_time=dt, rate_func=linear
        )

        dt = 2
        def update_chord_opacity(t1, t2):
            t = vt.get_value()
            if t1 < t <= t2:
                holditch.chord.opacity = 1 - (t - t1)/(t2 - t1)
        chord_opacity_updater = lambda _: update_chord_opacity(time - dt, time)
        holditch.group.add_updater(chord_opacity_updater)
        self.play(
            vt.animate.set_value(time := time + dt),
            FadeOut(holditch.strip),
            FadeOut(holditch.curve),
            FadeOut(holditch.locus),
            run_time=dt, rate_func=linear
        )
        self.remove(*holditch.group)
        self.wait(0.1)


class Scene22(Scene):
    def construct(self):
        self.add(CC0)

        tex_template = TexTemplate()
        tex_template.add_to_preamble(r'\usepackage{url}')
        tex_template.add_to_preamble(r'\def\UrlBreaks{\do\/\do-}')
        tex_template.add_to_preamble(r'\setlength{\textwidth}{260pt}')

        text = Tex(r'The original article from 1858\\'
                   'by Rev. Hamnet Holditch:').move_to(2.5*UP)
        bib = Tex(
            r'''
            \renewcommand*{\refname}{}
            \begin{thebibliography}{99}

            \bibitem{holditch1858}
              Holditch, H. (1858).
              Geometrical theorem.
              \textit{The Quarterly Journal of Pure and Applied Mathematics},
              \textit{2}, 38.
              \url{http://resolver.sub.uni-goettingen.de/purl?PPN600494829_0002}

            \end{thebibliography}
            \bibliographystyle{plain}
            \bibliography{thebibliography}''',
            tex_template=tex_template
        ).move_to(0.5*DOWN)
        self.play(Write(text))
        self.play(Write(bib))

        self.wait(1.5)
        self.play(
            Unwrite(text, reverse=False),
            Unwrite(bib, reverse=False),
            run_time=0.5
        )

        text = Tex(
            r"For a review on van Schooten's Locus Problem:"
        ).move_to(text)
        tex_template = TexTemplate()
        tex_template.add_to_preamble(r'\usepackage{url}')
        tex_template.add_to_preamble(r'\def\UrlBreaks{\do\/\do-}')
        tex_template.add_to_preamble(r'\setlength{\textwidth}{280pt}')
        bib = Tex(
            r'''
            \renewcommand*{\refname}{}
            \begin{thebibliography}{99}
            \makeatletter
            \addtocounter{\@listctr}{1}
            \makeatother

            \bibitem{wetzel2010}
              Wetzel, J. E. (2010).
              An ancient elliptic locus.
              \textit{The American Mathematical Monthly},
              \textit{117}(2),
              161–167.
              \url{https://doi.org/10.4169/000298910X476068}

            \end{thebibliography}
            \bibliographystyle{plain}
            \bibliography{thebibliography}''',
            tex_template=tex_template
        ).move_to(bib)
        self.play(Write(text))
        self.play(Write(bib))
        self.wait(1.5)
        self.play(
            Unwrite(text, reverse=False),
            Unwrite(bib, reverse=False),
            run_time=0.5
        )

        text = Tex(r"For Monterde \& Rochera's full proof,\\"
                   'including many details omitted here:').move_to(3*UP)
        bib = Tex(
            r'''
            \renewcommand*{\refname}{}
            \begin{thebibliography}{99}
            \makeatletter
            \addtocounter{\@listctr}{2}
            \makeatother

            \bibitem{monterde2017}
              Monterde, J., \& Rochera, D. (2017).
              Holditch's Ellipse Unveiled.
              \textit{The American Mathematical Monthly},
              \textit{124}(5),
              403–421.
              \url{https://doi.org/10.4169/amer.math.monthly.124.5.403}

            \bibitem{plata2019}
              Rochera, D. (2019).
              \textit{On Holditch's theorem and related kinematics}
              [Doctoral dissertation, Universitat de València].
              \url{https://roderic.uv.es/handle/10550/72339}

            \end{thebibliography}
            \bibliographystyle{plain}
            \bibliography{thebibliography}''',
            tex_template=tex_template
        ).move_to(DOWN)
        self.play(Write(text))
        self.play(Write(bib))
        self.wait(2.5)
        self.play(
            Unwrite(text),
            Unwrite(bib),
            run_time=1
        )
        self.wait(0.1)
