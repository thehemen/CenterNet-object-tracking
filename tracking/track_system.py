import numpy as np
from .utils import generate_color
from .bounding_box import BoundingBox
from .track_object import TrackObject

class TrackSystem:
    def __init__(self, dist_threshold, iou_threshold, depth_threshold, ttl):
        self.__dist_threshold = dist_threshold
        self.__iou_threshold = iou_threshold
        self.__depth_threshold = depth_threshold
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

    def add_object(self, class_id, x, y, z, l, w, h, rot_y, score):
        object_id = len(self.__new_objects)
        bbox = BoundingBox(class_id, x, y, z, l, w, h, rot_y, score)
        new_object = TrackObject(object_id, bbox, self.__ttl)
        self.__new_objects.append(new_object)
        self.__saved_ids.append(-1)

    def update(self):
        pairs = []

        for new_object_id, new_object in enumerate(self.__new_objects):
            bbox = self.__new_objects[new_object_id].bbox

            for track_id, track_object in enumerate(self.__track_objects):
                if not track_object.is_alive():
                    continue

                if not track_object.bbox.class_id == bbox.class_id:
                    continue

                distance = track_object.bbox.distance(bbox)

                if distance < self.__dist_threshold:
                    pairs.append(tuple((new_object_id, track_id, distance)))

        pairs.sort(key=lambda x: x[2])

        while True:
            best_intersect_count = 0
            best_iou_acc = 0.0
            best_iou_pairs = {}
            best_x_speed, best_y_speed, best_z_speed = 0.0, 0.0, 0.0

            for pair_object_id, pair_track_id, distance in pairs:
                if self.__saved_ids[pair_object_id] != -1:
                    continue

                if pair_track_id in self.__saved_ids:
                    continue

                pair_track_bbox = self.__track_objects[pair_track_id].bbox
                pair_new_bbox = self.__new_objects[pair_object_id].bbox

                x_speed = pair_track_bbox.x - pair_new_bbox.x
                y_speed = pair_track_bbox.y - pair_new_bbox.y
                z_speed = pair_track_bbox.z - pair_new_bbox.z

                intersect_count = 0
                iou_acc = 0.0
                iou_pairs = {}

                for new_object_id, new_object in enumerate(self.__new_objects):
                    if self.__saved_ids[new_object_id] != -1:
                        continue

                    # Only the same class bboxes comes together.
                    if not pair_new_bbox.class_id == new_object.bbox.class_id:
                        continue

                    new_bbox = new_object.bbox.copy()
                    new_bbox.move(x_speed, y_speed, z_speed)

                    best_iou = 0.0
                    best_track_id = -1

                    for track_id, track_object in enumerate(self.__track_objects):
                        if not track_object.is_alive():
                            continue

                        if not track_object.bbox.class_id == new_bbox.class_id:
                            continue

                        if track_id in iou_pairs.values():
                            continue

                        if track_id in self.__saved_ids:
                            continue

                        iou = new_bbox.iou(track_object.bbox)

                        if iou > self.__iou_threshold:
                            if iou > best_iou:
                                best_iou = iou
                                best_track_id = track_id

                    if best_track_id != -1:
                        intersect_count += 1
                        iou_acc += best_iou
                        iou_pairs[new_object_id] = best_track_id

                if intersect_count >= best_intersect_count:
                    if iou_acc > best_iou_acc:
                        best_iou_acc = iou_acc
                        best_intersect_count = intersect_count
                        best_iou_pairs = dict(iou_pairs)
                        best_x_speed, best_y_speed, best_z_speed = x_speed, y_speed, z_speed

            if len(best_iou_pairs.items()) != 0:
                neighbour_ids = best_iou_pairs.values()
                for new_object_id, track_id in best_iou_pairs.items():
                    bbox = self.__new_objects[new_object_id].bbox
                    self.__track_objects[track_id].update(bbox, best_x_speed, best_y_speed, best_z_speed,
                        neighbour_ids, self.__ttl)
                    self.__saved_ids[new_object_id] = track_id
            else:
                break

        for track_id, track_object in enumerate(self.__track_objects):
            if track_object.is_alive() and not track_object.is_updated():
                neighbour_ids = track_object.neighbour_ids
                neighbour_count = len(neighbour_ids)
                speeds = None

                if neighbour_count > 0:
                    x_speed_acc = 0.0
                    y_speed_acc = 0.0
                    z_speed_acc = 0.0

                    for neighbour_id in neighbour_ids:
                        if neighbour_id == track_id:
                            continue
                            
                        if self.__track_objects[neighbour_id].is_alive():
                            x_speed_acc += self.__track_objects[neighbour_id].x_speed
                            y_speed_acc += self.__track_objects[neighbour_id].y_speed
                            z_speed_acc += self.__track_objects[neighbour_id].z_speed

                    x_speed_acc /= neighbour_count
                    y_speed_acc /= neighbour_count
                    z_speed_acc /= neighbour_count
                    speeds = x_speed_acc, y_speed_acc, z_speed_acc

                self.__track_objects[track_id].try_kill(self.__depth_threshold, speeds)

        for new_object_id in range(len(self.__saved_ids)):
            if self.__saved_ids[new_object_id] != -1:
                continue

            track_id = len(self.__track_objects)
            new_object = self.__new_objects[new_object_id]
            new_object.track_id = track_id
            self.__track_objects.append(new_object)
            self.__saved_ids[new_object_id] = track_id
            self.__colors.append(generate_color())

    def get_object_ids(self):
        return self.__saved_ids

    def get_object(self, index):
        return self.__track_objects[index]

    def get_color(self, index):
        return self.__colors[index]
