class TrackChecking:
    def __init__(self, width, height, z_min_threshold, dim_ratio):
        self.__width = width
        self.__height = height
        self.__z_min_threshold = z_min_threshold
        self.__dim_ratio = dim_ratio

        self.__bboxes_3d = {}

    def check_bbox_2d(self, bbox_2d):
        x1, y1, x2, y2 = bbox_2d

        if x1 < 0.0:
            x1 = 0.0

        if y1 < 0.0:
            y1 = 0.0

        if x2 >= self.__width:
            x2 = self.__width - 1.0

        if y2 >= self.__height:
            y2 = self.__height - 1.0

        return x1, y1, x2, y2

    def check_bbox_3d(self, track_id, bbox_3d):
        if bbox_3d.z - bbox_3d.h < self.__z_min_threshold:
            if track_id in self.__bboxes_3d.keys():
                bbox_3d = self.__bboxes_3d[track_id]

                bbox_3d.l *= self.__dim_ratio
                bbox_3d.w *= self.__dim_ratio
                bbox_3d.h *= self.__dim_ratio
            else:
                self.__bboxes_3d[track_id] = bbox_3d

        return bbox_3d