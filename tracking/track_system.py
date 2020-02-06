from .utils import generate_color
from .track_object import TrackObject

class TrackSystem:
    def __init__(self, dist_threshold):
        self.__dist_threshold = dist_threshold

        self.__track_objects = []
        self.__saved_ids = []
        self.__colors = []

    def reset(self):
        for i in range(len(self.__track_objects)):
            self.__track_objects[i].reset()
        self.__saved_ids.clear()

    def add_object(self, class_id, x, y, z, l, w, h, rot_y, score):
        min_object_id = -1
        min_dist = self.__dist_threshold

        for i, track_object in enumerate(self.__track_objects):
            if track_object.is_alive():
                continue

            if not class_id == track_object.class_id:
                continue

            dist = track_object.distance(x, y, z)

            if dist < min_dist:
                min_object_id = i
                min_dist = dist

        if min_object_id != -1:
            track_id = self.__track_objects[min_object_id].track_id
            self.__track_objects[min_object_id].update(x, y, z, l, w, h)
        else:
            track_id = len(self.__track_objects)
            new_track_object = TrackObject(track_id, class_id, x, y, z, l, w, h, rot_y, score)
            self.__track_objects.append(new_track_object)
            self.__colors.append(generate_color())

        self.__saved_ids.append(track_id)

    def get_object_ids(self):
        return self.__saved_ids

    def get_object(self, index):
        return self.__track_objects[index]

    def get_color(self, index):
        return self.__colors[index]
