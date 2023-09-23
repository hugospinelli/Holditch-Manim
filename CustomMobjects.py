from collections.abc import Sequence
from manim import *

class PolygonalChain(Polygram):
    def __init__(self, *vertices: Sequence[float], color=BLUE, **kwargs):
        type(self).mro()[2].__init__(self, color=color, **kwargs)

        vertices = np.array(vertices)
        self.start_new_path(vertices[0])
        for vertex in vertices[1:]:
            self.add_line_to(vertex)
