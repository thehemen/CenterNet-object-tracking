class TrackObject:
    def __init__(self, track_id, bbox, ttl):
        self.track_id = track_id
        self.bbox = bbox

        # Needs to check if a new overlapping object is found or not.
        self.__is_updated = True

        self.__is_alive = True
        self.__ttl = ttl

    def reset(self):
        self.__is_updated = False

    def update(self, bbox, ttl):
        self.bbox = bbox
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
