from .utils import get_distance, get_iou

class BoundingBox:
    def __init__(self, class_id, x, y, z, l, w, h, rot_y, score):
        self.class_id = class_id

        self.x = x
        self.y = y
        self.z = z

        self.l = l
        self.w = w
        self.h = h

        self.rot_y = rot_y
        self.score = score

    def distance(self, bbox):
        return get_distance(self.x, self.y, self.z, bbox.x, bbox.y, bbox.z)

    def iou(self, bbox):
        return get_iou(self.x - self.l, self.y - self.w, self.z - self.h,
            self.x + self.l, self.y + self.w, self.z + self.h,
            bbox.x - bbox.l, bbox.y - bbox.w, bbox.z - bbox.h,
            bbox.x + bbox.l, bbox.y + bbox.w, bbox.z + bbox.h)

    def copy(self):
        return BoundingBox(self.class_id, self.x, self.y, self.z,
            self.l, self.w, self.h, self.rot_y, self.score)

    def move(self, x_speed, y_speed, z_speed):
        self.x += x_speed
        self.y += y_speed
        self.z += z_speed