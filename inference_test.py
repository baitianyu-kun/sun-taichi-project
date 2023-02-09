import mmcv
from mmpose.apis import inference_top_down_pose_model, inference_pose_lifter_model
from main import *
from matplotlib import pyplot as plt
import os.path as osp
from mpl_toolkits.mplot3d import Axes3D
from mmpose.core.camera import SimpleCamera

config_file = 'model/associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = 'model/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
img = cv.imread('./imgs/test_pics/quanshen.jpg')
pose_results, _ = inference_bottom_up_pose_model(pose_model, img)
print(pose_results[0]['keypoints'][:,:2].astype(int))