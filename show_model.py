from torchsummary import summary
from main import *


config_file = 'model/associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = 'model/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'


summary(pose_model, input_size=(3, 480, 640))
