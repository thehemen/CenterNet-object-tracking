import cv2
import glob
import argparse
from tracking.utils import generate_color

class BoundingBox2D:
    def __init__(self, class_id, x1, y1, x2, y2, track_id, score):
        self.class_id = class_id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.track_id = track_id
        self.score = score

if __name__ == '__main__':
    image_dir = '../kitti/training/image_02/{}/*.png'
    gt_dir = '../kitti/training/label_02/{}.txt'
    pred_dir = '../kitti/devkit_tracking/devkit/python/results/result_sha/data/{}.txt'
    classes = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']

    parser = argparse.ArgumentParser(description='Parse the KITTI ground truth or predictions.')
    parser.add_argument('--index', type=int, default=1, help='The index of ground truth/predictions.')
    parser.add_argument('--is_gt', type=bool, default=False, help='To use ground truth or predictions.')
    args = parser.parse_args()

    if args.is_gt:
        label_dir = gt_dir
    else:
        label_dir = pred_dir

    frame_names = sorted(glob.glob(image_dir.format(str(args.index).zfill(4))))
    bboxes = {}
    colors = {}

    for frame_id in range(len(frame_names)):
        bboxes[frame_id] = []

    with open(label_dir.format(str(args.index).zfill(4)), 'r') as f:
        lines = f.readlines()

        for line in lines:
            values = line.split(' ')
            frame_id = int(values[0])
            track_id = int(values[1])
            class_name = values[2]

            if class_name == 'DontCare':
                continue

            if class_name == 'Person':
                class_name = 'Pedestrian'

            class_id = classes.index(class_name)
            x1 = int(float(values[6]))
            y1 = int(float(values[7]))
            x2 = int(float(values[8]))
            y2 = int(float(values[9]))

            if not args.is_gt:
                score = float(values[17])
            else:
                score = 1.0

            bboxes[frame_id].append(BoundingBox2D(class_id, x1, y1, x2, y2, track_id, score))

            if track_id not in colors.keys():
                colors[track_id] = generate_color()

    header_name = '{}'.format(args.index)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for frame_id, frame_name in enumerate(frame_names):
        img = cv2.imread(frame_name)
        print('\nFrame: {}'.format(frame_id))

        for bbox in bboxes[frame_id]:
            x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
            class_id = bbox.class_id
            track_id = bbox.track_id
            score = bbox.score
            txt = '{} {:.2f} {}'.format(classes[class_id], score, track_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[track_id], 1)
            cv2.rectangle(img, (x1, y1 - 16), (x1 + len(txt) * 12, y1), colors[track_id], -1)
            cv2.putText(img, txt, (x1, y1 - 4), font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            print('\t{}. ID: {}. {}, {}, {}, {}. Score: {:.3f}.'.format(classes[class_id],
                track_id, x1, y1, x2, y2, score))

        cv2.imshow(header_name, img)
        k = cv2.waitKey(0)
        if k == 27:
            break

    cv2.destroyAllWindows()
