# CenterNet Object Tracking
This project is used to implement the KITTI object detection and tracking system using a pretrained [CenterNet](https://github.com/xingyizhou/CenterNet) model.

# How to run
Firstly, download [KITTI left images and labels](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) to evaluate the model.

Secondly, download the pretrained [ddd_3dop.pth](https://drive.google.com/open?id=1znsM6E-aVTkATreDuUVxoU0ajL1az8rz) model.

To predict and track objects from KITTI, use:
~~~
python3 predict.py [--dataset_type] [--model_name] [--score_threshold] [--dist_threshold] [--iou_threshold] [--depth_threshold] [--check_zmin] [--check_dim_ratio] [--ttl] [--begin_index] [--end_index] [--show_frames/no_show_frames] [--verbose/no_verbose] [--with_keys/with_no_keys]
~~~
To show already predicted objects, use:
~~~
python3 parse.py [--dataset_type] [--index] [--is_gt]
~~~
# Results
Unfortunately, this model can't be evaluated on the testing KITTI dataset due to its policy. So, only training dataset's results are published.
|Class|MOTA|MOTP|MT|ML|IDS|FRAG|
|-----|----|----|--|--|---|----|
|Car|79.90%|80.22%|70.92%|6.91%|165|539|
|Pedestrian|52.26%|69.23%|39.52%|11.97%|490|915|