from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from PIL import Image


class Triangle:
    def __init__(self):
        self.create_triangle()

    def create_triangle(self):
        self.points = []
        self.edges = []
        self.angles = []

        for i in range(3):
            self.points.append((random.uniform(0, 500), random.uniform(0, 500)))

        self.get_edges(self.points)
        self.get_angles(self.edges)

        if not self.check_triangle():
            self.create_triangle()

    def get_distance(self, p0, p1):
        return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)

    def get_edges(self, points):
        for i in points:
            for j in points:
                dist = self.get_distance(i, j)
                if dist != 0 and dist not in self.edges:
                    self.edges.append(dist)

    def get_angles(self, edges):
        a, b, c = edges[0], edges[1], edges[2]
        self.angles.append(math.degrees(math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))))
        self.angles.append(math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))))
        self.angles.append(math.degrees(math.acos((c ** 2 + a ** 2 - b ** 2) / (2 * c * a))))

    def check_triangle(self):
        if len(self.edges) == 3 and len(self.angles) == 3:
            for edge in self.edges:
                if edge < 100: return False
            for angle in self.angles:
                if angle < 30: return False
            return True
        else:
            return False

    def draw(self):
        white = 255
        img = np.zeros([500, 500, 3], dtype=np.uint8)
        img.fill(0)

        draw_line(img, self.points[0], self.points[1], white)
        draw_line(img, self.points[0], self.points[2], white)
        draw_line(img, self.points[1], self.points[2], white)

        plt.imshow(img)
        plt.show()


def _fpart(x):
    return x - int(x)


def _rfpart(x):
    return 1 - _fpart(x)


def putpixel(img, xy, color, alpha=1):
    """Paints color over the background at the point xy in img.
    Use alpha for blending. alpha=1 means a completely opaque foreground.
    """
    img[xy[0]][xy[1]] = color


def draw_line(img, p1, p2, color):
    """Draws an anti-aliased line in img from p1 to p2 with the given color."""
    x1, y1, x2, y2 = p1 + p2
    dx, dy = x2 - x1, y2 - y1
    steep = abs(dx) < abs(dy)
    p = lambda px, py: ((px, py), (py, px))[steep]

    if steep:
        x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx
    if x2 < x1:
        x1, x2, y1, y2 = x2, x1, y2, y1

    grad = dy / dx
    intery = y1 + _rfpart(x1) * grad

    def draw_endpoint(pt):
        x, y = pt
        xend = round(x)
        yend = y + grad * (xend - x)
        xgap = _rfpart(x + 0.5)
        px, py = int(xend), int(yend)
        putpixel(img, (px, py), color, _rfpart(yend) * xgap)
        putpixel(img, (px, py + 1), color, _fpart(yend) * xgap)
        return px

    xstart = draw_endpoint(p(*p1)) + 1
    xend = draw_endpoint(p(*p2))

    for x in range(xstart, xend):
        y = int(intery)
        putpixel(img, p(x, y), color, _rfpart(intery))
        putpixel(img, p(x, y + 1), color, _fpart(intery))
        intery += grad


if __name__ == '__main__':
    # tr = Triangle()
    # tr.draw()

    import random

    img = np.zeros([500,500,3],dtype=np.uint8)
    img.fill(0) # or img[:] = 255
    P = 0.1
    for i in range(500):
        for j in range(500):
            x = random.random()
            if x <= P:
                color = random.randint(0, 255)
                img[i][j] = color

    plt.imshow(img)
    plt.show()

    with open('1.pgm', 'w') as f:
        pgmHeader = 'P5' + '\n' + str(500) + '  ' + str(500) + '  ' + str(255) + '\n'
        f.write(pgmHeader)
        img.tofile(f)




