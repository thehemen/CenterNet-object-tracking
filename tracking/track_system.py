import numpy as np
from .utils import *
from .track_object import TrackObject

class TrackSystem:
    def __init__(self, dist_threshold, ttl):
        self.__dist_threshold = dist_threshold
        self.__ttl = ttl

        self.__track_objects = []
        self.__new_objects = []
        self.__saved_ids = []
        self.__colors = []

    def reset(self):
        for i in range(len(self.__track_objects)):
            self.__track_objects[i].reset()
        self.__new_objects.clear()
        self.__saved_ids.clear()

    def add_object(self, class_id, x, y, z, l, w, h, x_2d, y_2d, w_2d, h_2d, rot_y, score):
        object_id = len(self.__new_objects)
        new_object = TrackObject(object_id, class_id, x, y, z, l, w, h, x_2d, y_2d, w_2d, h_2d, rot_y, score, self.__ttl)
        self.__new_objects.append(new_object)
        self.__saved_ids.append(-1)

    def update(self):
        pairs = []

        for new_object_id, new_object in enumerate(self.__new_objects):
            new_object = self.__new_objects[new_object_id]
            class_id = new_object.class_id
            x, y, z = new_object.x, new_object.y, new_object.z

            for track_id, track_object in enumerate(self.__track_objects):
                if not track_object.is_alive():
                    continue

                if not class_id == track_object.class_id:
                    continue

                distance = track_object.distance(x, y, z)

                if distance < self.__dist_threshold:
                    pairs.append(tuple((new_object_id, track_id, distance)))

        pairs.sort(key=lambda x: x[2])

        for new_object_id, track_id, distance in pairs:
            if self.__saved_ids[new_object_id] != -1:
                continue

            if self.__track_objects[track_id].is_updated():
                continue

            new_object = self.__new_objects[new_object_id]
            x, y, z = new_object.x, new_object.y, new_object.z
            l, w, h = new_object.l, new_object.w, new_object.h
            x_2d, y_2d, w_2d, h_2d = new_object.x_2d, new_object.y_2d, new_object.w_2d, new_object.h_2d
            self.__track_objects[track_id].update(x, y, z, l, w, h, x_2d, y_2d, w_2d, h_2d, self.__ttl)
            self.__saved_ids[new_object_id] = track_id

        for new_object_id in range(len(self.__saved_ids)):
            if self.__saved_ids[new_object_id] != -1:
                continue

            track_id = len(self.__track_objects)
            new_object = self.__new_objects[new_object_id]
            new_object.track_id = track_id
            self.__track_objects.append(new_object)
            self.__saved_ids[new_object_id] = track_id
            self.__colors.append(generate_color())

    def update_ttl(self):
        for i, track_object in enumerate(self.__track_objects):
            if track_object.is_alive() and not track_object.is_updated():
                self.__track_objects[i].dec_ttl()

    def get_object_ids(self):
        return self.__saved_ids

    def get_object(self, index):
        return self.__track_objects[index]

    def get_color(self, index):
        return self.__colors[index]
