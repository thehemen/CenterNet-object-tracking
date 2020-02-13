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
    parser.add_argument('--score_threshold', type=float, default=0.5, help='The object score threshold.')
    parser.add_argument('--dist_threshold', type=float, default=5.0, help='The nearest object distances threshold.')
    parser.add_argument('--ttl', type=int, default=3, help='The objects time-to-live.')
    parser.add_argument('--begin_index', type=int, default=1, help='The begin index of frame sets.')
    parser.add_argument('--end_index', type=int, default=1, help='The end index of frame sets.')
    parser.add_argument('--show_frames', dest='is_shown', help='To show debug information.', action='store_true')
    parser.add_argument('--no_show_frames', dest='is_shown', help='Not to show debug information.', action='store_false')
    parser.add_argument('--verbose', dest='is_verbose', help='To print out debug information.', action='store_true')
    parser.add_argument('--no_verbose', dest='is_verbose', help='Not to print out debug information.', action='store_false')
    parser.add_argument('--with_keys', dest='continue_by_key', help='To continue frames processing only by a pressed key (with exit by \'q\').', action='store_true')
    parser.add_argument('--with_no_keys', dest='continue_by_key', help='To output frames without waiting for a key pressed.', action='store_false')
    parser.set_defaults(is_shown=True)
    parser.set_defaults(is_verbose=True)
    parser.set_defaults(continue_by_key=False)
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
        print('\nImageset Index: {}'.format(imageset_index))
        trackSystem = TrackSystem(dist_threshold=args.dist_threshold, ttl=args.ttl)

        imageset_index_str = str(imageset_index).zfill(4)
        image_names = sorted(glob.glob(image_dir.format(imageset_index_str)))

        f = open(out_dir.format(imageset_index_str), 'w')

        for frame_id in tqdm.tqdm(range(len(image_names)), disable=args.is_verbose):
            outputs = detector.run(image_names[frame_id])
            ret, dets = outputs['results'], outputs['dets'][0]
            trackSystem.reset()

            index = 0
            for class_id, val in ret.items():
                for i in range(len(val)):
                    l, w, h = val[i][5: 8]
                    x, y, z  = val[i][8: 11]
                    rot_y = val[i][11]
                    score = val[i][-1]

                    if score <= args.score_threshold:
                        continue

                    while dets[index, 2] <= args.score_threshold:
                        index += 1

                    w_2d, h_2d = dets[index, -3], dets[index, -2]
                    x_2d, y_2d = dets[index, 0], dets[index, 1]
                    trackSystem.add_object(class_id, x, y, z, l, w, h, x_2d, y_2d, w_2d, h_2d, rot_y, score)
                    index += 1

            trackSystem.update()

            bboxes_3d = []
            colors = []
            texts = []

            for track_id in trackSystem.get_object_ids():
                bbox = trackSystem.get_object(track_id).bbox
                bboxes_3d.append(bbox)

                color = trackSystem.get_color(track_id)
                colors.append(color)
                texts.append('{}'.format(track_id))

            trackSystem.update_ttl()
            imgs, bboxes_2d = detector.get_drawn_results(bboxes_3d, colors, texts)

            if args.is_verbose:
                print('\nFrame: {}'.format(frame_id))

            for track_id, bbox_2d in zip(trackSystem.get_object_ids(), bboxes_2d):
                bbox = trackSystem.get_object(track_id).bbox
                class_name = classes[bbox.class_id]
                score = bbox.score
                x1, y1, x2, y2 = bbox_2d

                if args.is_verbose:
                    print('\t{}. ID: {}. {:.6f}, {:.6f}, {:.6f}. ({:.6f} x {:.6f} x {:.6f}). RotY: {:.6f}. Score: {:.3f}.'.format(class_name,
                        track_id, bbox.x, bbox.y, bbox.z, bbox.l, bbox.w, bbox.h, bbox.rot_y, score))

                f.write('{} {} {} 0 0 0 {} {} {} {} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 {}\n'.format(frame_id,
                    track_id, class_name, x1, y1, x2, y2, score))

            if args.is_shown:
                for i, v in imgs.items():
                    cv2.imshow('{}'.format(i), v)

                if args.continue_by_key:
                    k = cv2.waitKey(0)
                    if k == 27:
                        break
                else:
                    cv2.waitKey(1)

        if args.is_shown:
            cv2.destroyAllWindows()
        f.close()
