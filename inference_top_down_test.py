import mmcv
from mmpose.apis import inference_top_down_pose_model, inference_pose_lifter_model
from main import *
from matplotlib import pyplot as plt
import os.path as osp
from mpl_toolkits.mplot3d import Axes3D
from mmpose.core.camera import SimpleCamera


def _keypoint_camera_to_world(keypoints,
                              camera_params,
                              image_name=None,
                              dataset='Body3DH36MDataset'):
    """Project 3D keypoints from the camera space to the world space.

    Args:
        keypoints (np.ndarray): 3D keypoints in shape [..., 3]
        camera_params (dict): Parameters for all cameras.
        image_name (str): The image name to specify the camera.
        dataset (str): The dataset type, e.g. Body3DH36MDataset.
    """
    cam_key = None
    if dataset == 'Body3DH36MDataset':
        subj, rest = osp.basename(image_name).split('_', 1)
        _, rest = rest.split('.', 1)
        camera, rest = rest.split('_', 1)
        cam_key = (subj, camera)
    else:
        raise NotImplementedError

    camera = SimpleCamera(camera_params[cam_key])
    keypoints_world = keypoints.copy()
    keypoints_world[..., :3] = camera.camera_to_world(keypoints[..., :3])

    return keypoints_world


config_file = 'model/associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = 'model/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
pose_lift_config_file = './model/simplebaseline3d_h36m.py'
pose_lift_ckpt_file = './model/simple3Dbaseline_h36m-f0ad73a4_20210419.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
print(pose_model)
pose_lift_model = init_pose_model(pose_lift_config_file, pose_lift_ckpt_file, device='cuda:0')

img = cv.imread('./imgs/test_pics/quanshen.jpg')
pose_results, _ = inference_top_down_pose_model(pose_model, img)
print(pose_results)
# pose_results_3d = inference_pose_lifter_model(
#     pose_lift_model,
#     [pose_results],
#     dataset='Body3DH36MDataset',
#     with_track_id=False
# )
# keypoints_3d = pose_results_3d[0]['keypoints_3d']
# # keypoints_3d=_keypoint_camera_to_world(keypoints_3d,
# #                                        camera_params=mmcv.load('./model/cameras.pkl'),
# #                                        image_name='tests/data/h36m\S1_Directions_1.54138969_000001.jpg',
# #                                        dataset='Body3DH36MDataset')
# xyz = keypoints_3d[:, :3]
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# x = xyz[:, 0]
# y = xyz[:, 1]
# z = xyz[:, 2]
# ax.scatter(x, y, z, c='g', marker='o')
# ax.set_xlabel('x axis')
# ax.set_ylabel('y axis')
# ax.set_zlabel('z axis')
# ax.view_init(elev=20,    # 仰角
#              azim=-100    # 方位角
#             )
# plt.show()

# xys = np.array(pose_results[0]['keypoints'])[:, :2].astype(int)
# thickness = 5
# point_size = 1
# for xy in xys:
#     cv.circle(img, tuple(xy), point_size, (0, 0, 255), thickness)
# cv.imshow('img',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
