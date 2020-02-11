from .utils import get_distance

class TrackObject:
    def __init__(self, track_id, class_id, x, y, z, l, w, h, x_2d, y_2d, w_2d, h_2d, rot_y, score, ttl):
        self.track_id = track_id
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

        # Needs to check if a new overlapping object is found or not.
        self.__is_updated = True

        self.__is_alive = True
        self.__ttl = ttl

    def distance(self, x, y, z):
        return get_distance(self.x, self.y, self.z, x, y, z)

    def reset(self):
        self.__is_updated = False

    def update(self, x, y, z, l, w, h, x_2d, y_2d, w_2d, h_2d, ttl):
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

        self.__is_updated = True
        self.__ttl = ttl

    def dec_ttl(self):
        if self.__is_alive:
            if self.__ttl > 0:
                self.__ttl -= 1
            else:
                self.__is_alive = False

    def is_updated(self):
        return self.__is_updated

    def is_alive(self):
        return self.__is_alive
