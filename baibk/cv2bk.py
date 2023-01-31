# 画点
# xy = np.array(pose_results[0]['keypoints'])[:, :2]
# # [[414.62841796875, 288.29150390625], [418.77783203125, 274.06494140625],
# xy = xy.tolist()
# # [(414, 288), (418, 274), 在opencv中展示
# xy = [(int(x), int(y)) for x, y in xy]
#
# img = np.zeros((607, 1080, 3), np.uint8)  # 生成一个空灰度图像
# point_size = 1
# point_color = (0, 0, 255)  # BGR
# thickness = 5  # 可以为 0 、4、8
#
# for point in xy:
#     cv.circle(img, point, point_size, point_color, thickness)
#     cv.imshow('image', img)
#     cv.waitKey(2)  # 显示 10000 ms 即 10s 后消失
#     time.sleep(2)

# 显示颜色
# from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)
# import numpy as np
# import cv2 as cv
#
# config_file = 'config.py'
# checkpoint_file = 'weight.pth'
# pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
#
# image_path = '1.jpg'
# # test a single image
# pose_results, _ = inference_bottom_up_pose_model(pose_model, image_path)
# xy = np.array(pose_results[0]['keypoints'])[:, :2]
# # [[414.62841796875, 288.29150390625], [418.77783203125, 274.06494140625],
# xy = xy.tolist()
# # [(414, 288), (418, 274), 在opencv中展示
# xy = [(int(x), int(y)) for x, y in xy]
#
# img = cv.imread('1.jpg')
# point_size = 1
# palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
#                     [230, 230, 0], [255, 153, 255], [153, 204, 255],
#                     [255, 102, 255], [255, 51, 255], [102, 178, 255],
#                     [51, 153, 255], [255, 153, 153], [255, 102, 102],
#                     [255, 51, 51], [153, 255, 153], [102, 255, 102],
#                     [51, 255, 51], [0, 255, 0], [0, 0, 255],
#                     [255, 0, 0], [255, 255, 255]])
# # 17个点
# # 前5个: 脸部(1(鼻子),2(右眼睛),3(左眼睛),4(右耳朵),5(左耳朵))
# # 之后6个: 胳膊(2(左肩膀),4(左肘),6(左手)，1(右肩膀),3(右肘),5(右手))
# # 在之后6个: 两条腿(2(左屁股),4(左膝盖),6(左脚)，1(右屁股),3(右膝盖),5(右脚))
# pose_kpt_color = palette[[
#     16, 16, 16, 17, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
# ]]
# thickness = 5  # 可以为 0 、4、8
# for point_index in range(len(xy)):
#     cv.circle(img, xy[point_index], point_size, tuple(pose_kpt_color[point_index].tolist()), thickness)
# cv.imshow('image', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# show the results
# vis_pose_result(pose_model, image_path, pose_results, out_file='vis_persons.jpg', thickness=3)




import math
import time

from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)
import numpy as np
import cv2 as cv

config_file = '../model/associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = '../model/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
image_path = '../imgs/test_pics/quanshen.jpg'
# test a single image
pose_results, _ = inference_bottom_up_pose_model(pose_model, image_path)
xy = np.array(pose_results[0]['keypoints'])[:, :2].tolist()
body_parts = {'nose': xy[0], 'right_eye': xy[1], 'left_eye': xy[2], 'right_ear': xy[3], 'left_ear': xy[4],
              'right_shoulder': xy[5], 'left_shoulder': xy[6], 'right_elbow': xy[7], 'left_elbow': xy[8],
              'right_wrist': xy[9], 'left_wrist': xy[10], 'right_ass': xy[11], 'left_ass': xy[12],
              'right_knee': xy[13], 'left_knee': xy[14], 'right_jiaohuai': xy[15], 'left_jiaohuai': xy[16]}
# 17个点
# 前5个: 脸部(1(鼻子),2(右眼睛),3(左眼睛),4(右耳朵),5(左耳朵))
# 之后6个: 胳膊(2(左肩膀),4(左肘),6(左手腕)，1(右肩膀),3(右肘),5(右手腕))
# 在之后6个: 两条腿(2(左屁股),4(左膝盖),6(左脚踝)，1(右屁股),3(右膝盖),5(右脚踝))
img = np.zeros((969, 801, 3), np.uint8) #生成一个空灰度图像
img.fill(255)
thickness = 5
point_size = 1
for key in body_parts.keys():
    x, y = body_parts[key]
    point = (int(x), int(y))
    if 'right' in key:
        color = (0, 0, 255)
    elif 'left' in key:
        color = (0, 255, 0)
    else:
        color = (255, 0, 0)
    cv.circle(img, point, point_size, color, thickness)
cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()

