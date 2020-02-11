from .utils import get_distance

class BoundingBox:
    def __init__(self, class_id, x, y, z, l, w, h, x_2d, y_2d, w_2d, h_2d, rot_y, score):
        self.class_id = class_id

        self.x = x
        self.y = y
        self.z = z

        self.l = l
        self.w = w
        self.h = h

        self.x_2d = x_2d
        self.y_2d = y_2d

        self.w_2d = w_2d
        self.h_2d = h_2d

        self.rot_y = rot_y
        self.score = score

    def distance(self, bbox):
        return get_distance(self.x, self.y, self.z, bbox.x, bbox.y, bbox.z)