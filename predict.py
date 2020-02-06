import os
import sys

CENTERNET_PATH = 'CenterNet/src/lib/'
sys.path.insert(0, CENTERNET_PATH)

import cv2
import random
import colorsys
import argparse

from detectors.detector_factory import detector_factory
from opts import opts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict 3D bounding boxes by CenterNet.')
    parser.add_argument('--model_name', default='data/ddd_3dop.pth', help='the pretrained model\'s location')
    parser.add_argument('--image_name', default='data/example.png', help='the image\'s location')
    args = parser.parse_args()

    opt = opts().init('{} --load_model {}'.format('ddd', args.model_name).split(' '))
    detector = detector_factory[opt.task](opt)
    ret = detector.run(args.image_name)['results']

    classes = ['__background__', 'Pedestrian', 'Car', 'Cyclist']
    for key, val in ret.items():
        for i in range(len(val)):
            dim = val[i][5: 8]
            loc  = val[i][8: 11]
            rot_y = val[i][11]
            score = val[i][-1]
            print('\n{}'.format(classes[key]))
            print('Track ID: {}'.format(i))
            print('Dimensions: {:.6f}, {:.6f}, {:.6f}.'.format(dim[0], dim[1], dim[2]))
            print('Location: {:.6f}, {:.6f}, {:.6f}.'.format(loc[0], loc[1], loc[2]))
            print('RotY: {:.6f}.'.format(rot_y))
            print('Score: {:.6f}.'.format(score))

    colors = []
    texts = []
    for key, val in ret.items():
        for i in range(len(val)):
            # Get bright colors only.
            h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
            r,g,b = [int(256 * i) for i in colorsys.hls_to_rgb(h, l, s)]
            colors.append(tuple((r, g, b)))
            texts.append('{}'.format(i))

    imgs = detector.get_drawn_detections(ret, colors, texts)

    for i, v in imgs.items():
        cv2.imshow('{}'.format(i), v)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
