class TrackObject:
    def __init__(self, track_id, bbox, ttl):
        self.track_id = track_id
        self.bbox = bbox

        # To track an object if any bbox not found
        self.x_speed = 0.0
        self.y_speed = 0.0
        self.z_speed = 0.0
        self.neighbour_ids = []

        # Needs to check if a new overlapping object is found or not.
        self.__is_updated = True

        self.__is_alive = True
        self.__ttl = ttl

    def reset(self):
        self.__is_updated = False

    def update(self, bbox, x_speed, y_speed, z_speed, neighbour_ids, ttl):
        self.bbox = bbox
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.z_speed = z_speed
        self.neighbour_ids = neighbour_ids

        self.__ttl = ttl
        self.__is_updated = True

    def try_kill(self, depth_threshold, speeds):
        if self.__is_alive:
            if self.bbox.z > depth_threshold:
                if self.__ttl > 0:
                    self.__ttl -= 1

                    if speeds is not None:
                        x_speed, y_speed, z_speed = speeds[0], speeds[1], speeds[2]
                    else:
                        x_speed, y_speed, z_speed = self.x_speed, self.y_speed, self.z_speed

                    self.bbox.move(x_speed, y_speed, z_speed)
                else:
                    self.__is_alive = False
            else:
                self.__is_alive = False

    def is_updated(self):
        return self.__is_updated

    def is_alive(self):
        return self.__is_alive
