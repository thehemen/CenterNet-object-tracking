from .utils import get_distance

class TrackObject:
    def __init__(self, track_id, class_id, x, y, z, l, w, h, rot_y, score):
        self.track_id = track_id
        self.class_id = class_id

        self.x = x
        self.y = y
        self.z = z

        self.l = l
        self.w = w
        self.h = h

        self.rot_y = rot_y
        self.score = score

        # Needs to check if a new overlapping object is found or not.
        self.__is_updated = True

    def distance(self, x, y, z):
        return get_distance(self.x, self.y, self.z, x, y, z)

    def reset(self):
        self.__is_updated = False

    def update(self, x, y, z, l, w, h):
        self.x = x
        self.y = y
        self.z = z

        self.l = l
        self.w = w
        self.h = h

        self.__is_updated = True

    def is_alive(self):
        return self.__is_updated
