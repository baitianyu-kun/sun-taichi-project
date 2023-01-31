from mmpose.apis import inference_top_down_pose_model

from main import *

config_file = 'model/mspn50_coco_256x192.py'
checkpoint_file = 'model/mspn50_coco_256x192-8fbfb5d0_20201123.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
img = cv.imread('./imgs/test_pics/quanshen.jpg')
pose_results, _ = inference_top_down_pose_model(pose_model, img)
xys = np.array(pose_results[0]['keypoints'])[:, :2].astype(int)
thickness = 5
point_size = 1
for xy in xys:
    cv.circle(img, tuple(xy), point_size, (0, 0, 255), thickness)
cv.imshow('img',img)
cv.waitKey(0)
cv.destroyAllWindows()
