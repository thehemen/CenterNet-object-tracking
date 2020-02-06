import os
import sys

CENTERNET_PATH = 'CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

import cv2
import glob
import argparse

from detectors.detector_factory import detector_factory
from opts import opts

from tracking.track_system import TrackSystem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict 3D bounding boxes by CenterNet.')
    parser.add_argument('--model_name', default='data/ddd_3dop.pth', help='the pretrained model\'s location')
    parser.add_argument('--image_dir', default='../kitti/training/image_02/0001/*.png', help='the images\' location')
    args = parser.parse_args()
    classes = ['__background__', 'Pedestrian', 'Car', 'Cyclist']

    opt = opts().init('{} --load_model {}'.format('ddd', args.model_name).split(' '))
    detector = detector_factory[opt.task](opt)

    trackSystem = TrackSystem(dist_threshold=15.0)

    image_names = sorted(glob.glob(args.image_dir))
    for frame_id, image_name in enumerate(image_names):
        ret = detector.run(image_name)['results']
        trackSystem.reset()

        for class_id, val in ret.items():
            for i in range(len(val)):
                l, w, h = val[i][5: 8]
                x, y, z  = val[i][8: 11]
                rot_y = val[i][11]
                score = val[i][-1]
                trackSystem.add_object(class_id, x, y, z, l, w, h, rot_y, score)

        colors = {}
        texts = {}

        print('\nFrame: {}'.format(frame_id))
        for track_id in trackSystem.get_object_ids():
            obj = trackSystem.get_object(track_id)

            print('\n\t{}'.format(classes[obj.class_id]))
            print('\tTrack ID: {}'.format(track_id))
            print('\tDimensions: {:.6f}, {:.6f}, {:.6f}.'.format(obj.l, obj.w, obj.h))
            print('\tLocation: {:.6f}, {:.6f}, {:.6f}.'.format(obj.x, obj.y, obj.z))
            print('\tRotY: {:.6f}.'.format(obj.rot_y))
            print('\tScore: {:.6f}.'.format(obj.score))

            if obj.class_id not in colors.keys():
                colors[obj.class_id] = []
                texts[obj.class_id] = []

            color = trackSystem.get_color(track_id)
            colors[obj.class_id].append(color)
            texts[obj.class_id].append('{}'.format(track_id))

        imgs = detector.get_drawn_detections(ret, colors, texts)

        for i, v in imgs.items():
            cv2.imshow('{}'.format(i), v)

        cv2.waitKey(1)
    cv2.destroyAllWindows()
