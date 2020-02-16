import math
import random
import colorsys

def get_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

def generate_color():
    # Get bright colors only.
    h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
    r, g, b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
    return r, g, b

def get_square(x1, y1, z1, x2, y2, z2):
    return (x2 - x1) * (y2 - y1) * (z2 - z1)

def get_intersection(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    x5 = max(x1, x3)
    y5 = max(y1, y3)
    z5 = max(z1, z3)
    x6 = min(x2, x4)
    y6 = min(y2, y4)
    z6 = min(z2, z4)

    if x5 < x6 and y5 < y6 and z5 < z6:
        return get_square(x5, y5, z5, x6, y6, z6)
    else:
        return 0.0

def get_union(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    first = get_square(x1, y1, z1, x2, y2, z2)
    second = get_square(x3, y3, z3, x4, y4, z4)
    intersection = get_intersection(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)
    return first + second - intersection

def get_iou(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4):
    intersection = get_intersection(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)
    union = get_union(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4)
    return intersection / union