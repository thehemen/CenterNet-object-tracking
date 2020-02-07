import os
import sys

CENTERNET_PATH = 'CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

import cv2
import glob
import tqdm
import argparse

from detectors.detector_factory import detector_factory
from opts import opts

from tracking.track_system import TrackSystem

if __name__ == '__main__':
    image_dir = '../kitti/training/image_02/{}/*.png'
    seqmap_in_file = '../kitti/devkit_tracking/devkit/python/data/tracking/evaluate_tracking.seqmap.training'
    seqmap_out_file = '../kitti/devkit_tracking/devkit/python/data/tracking/evaluate_tracking.seqmap'
    out_dir = '../kitti/devkit_tracking/devkit/python/results/result_sha/data/{}.txt'

    parser = argparse.ArgumentParser(description='Predict 3D bounding boxes by CenterNet.')
    parser.add_argument('--model_name', default='data/ddd_3dop.pth', help='The pretrained model\'s location.')
    parser.add_argument('--score_threshold', default=0.5, help='To object score threshold.')
    parser.add_argument('--dist_threshold', default=15.0, help='To nearest object distances threshold.')
    parser.add_argument('--ttl', type=int, default=5, help='The objects time-to-live.')
    parser.add_argument('--begin_index', type=int, default=1, help='The begin index of frame sets.')
    parser.add_argument('--end_index', type=int, default=1, help='The end index of frame sets.')
    parser.add_argument('--is_debug', type=bool, default=True, help='To show debug info or not.')
    args = parser.parse_args()
    classes = ['__background__', 'Pedestrian', 'Car', 'Cyclist']

    with open(seqmap_in_file, 'r') as seq_in:
        with open(seqmap_out_file, 'w') as seq_out:
            lines = seq_in.readlines()
            for i in range(args.begin_index, args.end_index + 1):
                seq_out.write(lines[i])

    opt = opts().init('{} --load_model {} --vis_thresh {}'.format('ddd', args.model_name,
        args.score_threshold).split(' '))
    detector = detector_factory[opt.task](opt)

    for imageset_index in range(args.begin_index, args.end_index + 1):
        trackSystem = TrackSystem(dist_threshold=args.dist_threshold, ttl=args.ttl)

        imageset_index_str = str(imageset_index).zfill(4)
        image_names = sorted(glob.glob(image_dir.format(imageset_index_str)))

        f = open(out_dir.format(imageset_index_str), 'w')

        for frame_id in tqdm.tqdm(range(len(image_names)), disable=args.is_debug):
            ret = detector.run(image_names[frame_id])['results']
            trackSystem.reset()

            for class_id, val in ret.items():
                for i in range(len(val)):
                    l, w, h = val[i][5: 8]
                    x, y, z  = val[i][8: 11]
                    rot_y = val[i][11]
                    score = val[i][-1]

                    if score <= args.score_threshold:
                        continue

                    trackSystem.add_object(class_id, x, y, z, l, w, h, rot_y, score)

            colors = {}
            texts = {}

            for track_id in trackSystem.get_object_ids():
                obj = trackSystem.get_object(track_id)

                if obj.class_id not in colors.keys():
                    colors[obj.class_id] = []
                    texts[obj.class_id] = []

                color = trackSystem.get_color(track_id)
                colors[obj.class_id].append(color)
                texts[obj.class_id].append('{}'.format(track_id))

            trackSystem.update()
            imgs, bboxes = detector.get_drawn_detections(ret, colors, texts)

            if args.is_debug:
                print('\nFrame: {}'.format(frame_id))

            for track_id, bbox in zip(trackSystem.get_object_ids(), bboxes):
                obj = trackSystem.get_object(track_id)
                class_name = classes[obj.class_id]
                score = obj.score
                x1, y1, x2, y2 = bbox

                if args.is_debug:
                    print('\n\t{}'.format(class_name))
                    print('\tTrack ID: {}'.format(track_id))
                    print('\tDimensions: {:.6f}, {:.6f}, {:.6f}.'.format(obj.l, obj.w, obj.h))
                    print('\tLocation: {:.6f}, {:.6f}, {:.6f}.'.format(obj.x, obj.y, obj.z))
                    print('\tRotY: {:.6f}.'.format(obj.rot_y))
                    print('\tScore: {:.6f}.'.format(score))

                f.write('{} {} {} 0 0 0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {}\n'.format(frame_id,
                    track_id, class_name, x1, y1, x2, y2, score))

            if args.is_debug:
                for i, v in imgs.items():
                    cv2.imshow('{}'.format(i), v)
                cv2.waitKey(1)

        if args.is_debug:
            cv2.destroyAllWindows()
        f.close()
