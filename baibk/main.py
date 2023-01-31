import glob
import math
import os
import time

from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)
import numpy as np
import cv2 as cv


def cos_theta(array1, array2):
    norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array1))))
    norm2 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array2))))
    return sum([array1[i] * array2[i] for i in range(0, len(array1))]) / (norm1 * norm2)


def cos2degree(coss):
    return math.degrees(math.acos(coss))


def caluculate_cos(body_parts):
    # 1. left_shoulder-left_elbow-left_eye
    # 2. left_elbow-left_wrist-left_eye
    # 3. left_shoulder-left_elbow-left_wrist
    # 4. left_ass-left_knee-left_jiaohuai
    # 5. left_elbow-left_ass-left_jiaohuai
    # 6. left_jiaohuai-nose-right_jiaohuai
    cos_left_1 = cos_theta(body_parts['left_shoulder'] - body_parts['left_elbow'],
                           body_parts['left_eye'] - body_parts['left_elbow'])
    cos_left_2 = cos_theta(body_parts['left_elbow'] - body_parts['left_wrist'],
                           body_parts['left_eye'] - body_parts['left_wrist'])
    cos_left_3 = cos_theta(body_parts['left_shoulder'] - body_parts['left_elbow'],
                           body_parts['left_wrist'] - body_parts['left_elbow'])
    cos_left_4 = cos_theta(body_parts['left_ass'] - body_parts['left_knee'],
                           body_parts['left_jiaohuai'] - body_parts['left_knee'])
    cos_left_5 = cos_theta(body_parts['left_elbow'] - body_parts['left_ass'],
                           body_parts['left_jiaohuai'] - body_parts['left_ass'])
    cos_right_1 = cos_theta(body_parts['right_shoulder'] - body_parts['right_elbow'],
                            body_parts['right_eye'] - body_parts['right_elbow'])
    cos_right_2 = cos_theta(body_parts['right_elbow'] - body_parts['right_wrist'],
                            body_parts['right_eye'] - body_parts['right_wrist'])
    cos_right_3 = cos_theta(body_parts['right_shoulder'] - body_parts['right_elbow'],
                            body_parts['right_wrist'] - body_parts['right_elbow'])
    cos_right_4 = cos_theta(body_parts['right_ass'] - body_parts['right_knee'],
                            body_parts['right_jiaohuai'] - body_parts['right_knee'])
    cos_right_5 = cos_theta(body_parts['right_elbow'] - body_parts['right_ass'],
                            body_parts['right_jiaohuai'] - body_parts['right_ass'])
    cos_6 = cos_theta(body_parts['left_jiaohuai'] - body_parts['nose'],
                      body_parts['right_jiaohuai'] - body_parts['nose'])
    return [cos_left_1, cos_left_2, cos_left_3, cos_left_4, cos_left_5,
            cos_right_1, cos_right_2, cos_right_3, cos_right_4, cos_right_5, cos_6]


def calculate_ratio(body_parts):
    # left_shoulder_elbow_wrist
    # left_ass_knee_jiaohuai
    # left_ear_shoulder_elbow
    # right_shoulder_elbow_wrist
    # right_ass_knee_jiaohuai
    # right_ear_shoulder_elbow
    left_shoulder_elbow_wrist_ratio = normalvaluesratio(body_parts['left_shoulder'] - body_parts['left_elbow'],
                                                        body_parts['left_wrist'] - body_parts['left_elbow'])
    left_ass_knee_jiaohuai_ratio = normalvaluesratio(body_parts['left_ass'] - body_parts['left_knee'],
                                                     body_parts['left_jiaohuai'] - body_parts['left_knee'])
    left_ear_shoulder_elbow_ratio = normalvaluesratio(body_parts['left_ear'] - body_parts['left_shoulder'],
                                                      body_parts['left_elbow'] - body_parts['left_shoulder'])
    right_shoulder_elbow_wrist_ratio = normalvaluesratio(body_parts['right_shoulder'] - body_parts['right_elbow'],
                                                         body_parts['right_wrist'] - body_parts['right_elbow'])
    right_ass_knee_jiaohuai_ratio = normalvaluesratio(body_parts['right_ass'] - body_parts['right_knee'],
                                                      body_parts['right_jiaohuai'] - body_parts['right_knee'])
    right_ear_shoulder_elbow_ratio = normalvaluesratio(body_parts['right_ear'] - body_parts['right_shoulder'],
                                                       body_parts['right_elbow'] - body_parts['right_shoulder'])
    return [left_shoulder_elbow_wrist_ratio, left_ass_knee_jiaohuai_ratio, left_ear_shoulder_elbow_ratio,
            right_shoulder_elbow_wrist_ratio, right_ass_knee_jiaohuai_ratio, right_ear_shoulder_elbow_ratio]


def normalvaluesratio(array1, array2):
    norm1 = math.sqrt(sum(list(map(lambda x: math.pow(x, 2), array1))))
    return sum([array1[i] * array2[i] for i in range(0, len(array1))]) / norm1


def opencvshow(img_path, array1, array2, array3):
    def convertxy(xyarray):
        x, y = xyarray
        return (int(x), int(y))

    array1 = convertxy(array1)
    array2 = convertxy(array2)
    array3 = convertxy(array3)
    img = cv.imread(img_path)
    color = (0, 0, 255)
    thickness = 5
    point_size = 1
    cv.circle(img, array1, point_size, color, thickness)
    cv.circle(img, array2, point_size, color, thickness)
    cv.circle(img, array3, point_size, color, thickness)
    cv.line(img, array1, array2, color, thickness, 4)
    cv.line(img, array3, array2, color, thickness, 4)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def opencvwrite(img, color, array1, array2, array3):
    def convertxy(xyarray):
        x, y = xyarray
        return (int(x), int(y))

    array1 = convertxy(array1)
    array2 = convertxy(array2)
    array3 = convertxy(array3)
    thickness = 5
    point_size = 1
    cv.circle(img, array1, point_size, color, thickness)
    cv.circle(img, array2, point_size, color, thickness)
    cv.circle(img, array3, point_size, color, thickness)
    cv.line(img, array1, array2, color, thickness, 4)
    cv.line(img, array3, array2, color, thickness, 4)
    return img


def opencvwrite2(img, color, array1, array2, array3):
    def convertxy(xyarray):
        x, y = xyarray
        return (int(x), int(y))

    array1 = convertxy(array1)
    array2 = convertxy(array2)
    array3 = convertxy(array3)
    thickness = 3
    point_size = 5
    cv.circle(img, array1, point_size, color, thickness)
    cv.circle(img, array2, point_size, color, thickness)
    cv.circle(img, array3, point_size, color, thickness)
    cv.line(img, array1, array2, color, thickness, 4)
    cv.line(img, array3, array2, color, thickness, 4)


if __name__ == '__main__':
    pics_folder_path = '../imgs/pics_sun_taichi'
    pics_save_folder_path = '../imgs/process_pics_sun_taichi'
    config_file = '../model/associative_embedding_hrnet_w32_coco_512x512.py'
    checkpoint_file = '../model/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
    device = 'cuda:0'
    gesture_list = ['更鸡独立', '右蹬腿', '手挥琵琶', '白鹤亮翅', '倒撵猴', '高探马', '上步搬拦捶', '单鞭', '右通背',
                    '玉女穿梭']
    gesture_save_value_path = '../imgs/process_pics_sun_taichi/gesture_value.txt'
    gesture_file = open(os.path.join(pics_folder_path, 'gesture.txt'))
    gesutre = gesture_file.readlines()
    # {'1.jpg': '0', '2.jpg': '1', '3.jpg': '2',
    gesutre = {x.strip().split(' ')[0]: x.strip().split(' ')[1] for x in gesutre}
    gesture_file.close()
    for pic_file in os.listdir(pics_folder_path):
        if pic_file[-3:] != 'jpg':
            continue
        image_path = os.path.join(pics_folder_path, pic_file)
        save_angle_path = os.path.join(pics_save_folder_path, 'save_angle_' + pic_file)
        save_weight_path = os.path.join(pics_save_folder_path, 'save_weight_' + pic_file)
        pose_model = init_pose_model(config_file, checkpoint_file, device=device)
        pose_results, _ = inference_bottom_up_pose_model(pose_model, image_path)
        xy = np.array(pose_results[0]['keypoints'])[:, :2]
        # 17个点
        # 前5个: 脸部(1(鼻子),2(右眼睛),3(左眼睛),4(右耳朵),5(左耳朵))
        # 之后6个: 胳膊(2(左肩膀),4(左肘),6(左手腕)，1(右肩膀),3(右肘),5(右手腕))
        # 在之后6个: 两条腿(2(左屁股),4(左膝盖),6(左脚踝)，1(右屁股),3(右膝盖),5(右脚踝))
        body_parts = {'nose': xy[0], 'right_eye': xy[1], 'left_eye': xy[2], 'right_ear': xy[3], 'left_ear': xy[4],
                      'right_shoulder': xy[5], 'left_shoulder': xy[6], 'right_elbow': xy[7], 'left_elbow': xy[8],
                      'right_wrist': xy[9], 'left_wrist': xy[10], 'right_ass': xy[11], 'left_ass': xy[12],
                      'right_knee': xy[13], 'left_knee': xy[14], 'right_jiaohuai': xy[15], 'left_jiaohuai': xy[16]}
        degrees = [cos2degree(x) for x in caluculate_cos(body_parts)]
        # 1.jpg -> 0 = gesture_value
        gesture_value = gesutre[pic_file]
        with open(gesture_save_value_path, 'a+') as f:
            f.write(gesture_value + '|' + str(degrees) + '\n')
        # save angle
        # 1. left_shoulder-left_elbow-left_eye
        # 2. left_elbow-left_wrist-left_eye
        # 3. left_shoulder-left_elbow-left_wrist
        # 4. left_ass-left_knee-left_jiaohuai
        # 5. left_elbow-left_ass-left_jiaohuai
        # 6. left_jiaohuai-nose-right_jiaohuai
        img = cv.imread(image_path)
        opencvwrite2(img, (0, 0, 255), body_parts['left_shoulder'], body_parts['left_elbow'], body_parts['left_eye'])
        opencvwrite2(img, (0, 255, 0), body_parts['left_elbow'], body_parts['left_wrist'], body_parts['left_eye'])
        opencvwrite2(img, (255, 0, 0), body_parts['left_shoulder'], body_parts['left_elbow'], body_parts['left_wrist'])
        opencvwrite2(img, (255, 255, 0), body_parts['left_ass'], body_parts['left_knee'], body_parts['left_jiaohuai'])
        opencvwrite2(img, (0, 255, 255), body_parts['left_elbow'], body_parts['left_ass'], body_parts['left_jiaohuai'])
        opencvwrite2(img, (0, 0, 255), body_parts['right_shoulder'], body_parts['right_elbow'],
                     body_parts['right_eye'])
        opencvwrite2(img, (0, 255, 0), body_parts['right_elbow'], body_parts['right_wrist'], body_parts['right_eye'])
        opencvwrite2(img, (255, 0, 0), body_parts['right_shoulder'], body_parts['right_elbow'],
                     body_parts['right_wrist'])
        opencvwrite2(img, (255, 255, 0), body_parts['right_ass'], body_parts['right_knee'],
                     body_parts['right_jiaohuai'])
        opencvwrite2(img, (0, 255, 255), body_parts['right_elbow'], body_parts['right_ass'],
                     body_parts['right_jiaohuai'])
        opencvwrite2(img, (255, 0, 255), body_parts['left_jiaohuai'], body_parts['nose'], body_parts['right_jiaohuai'])
        cv.imwrite(save_angle_path, img)

        # save weight
        # left_shoulder_elbow_wrist
        # left_ass_knee_jiaohuai
        # left_ear_shoulder_elbow
        # right_shoulder_elbow_wrist
        # right_ass_knee_jiaohuai
        # right_ear_shoulder_elbow
        img1 = cv.imread(image_path)
        opencvwrite2(img1, (0, 0, 255), body_parts['left_shoulder'], body_parts['left_elbow'],
                     body_parts['left_wrist'])
        opencvwrite2(img1, (0, 255, 0), body_parts['left_ass'], body_parts['left_knee'],
                     body_parts['left_jiaohuai'])
        opencvwrite2(img1, (255, 0, 0), body_parts['left_ear'], body_parts['left_shoulder'],
                     body_parts['left_elbow'])
        opencvwrite2(img1, (0, 0, 255), body_parts['right_shoulder'], body_parts['right_elbow'],
                     body_parts['right_wrist'])
        opencvwrite2(img1, (0, 255, 0), body_parts['right_ass'], body_parts['right_knee'],
                     body_parts['right_jiaohuai'])
        opencvwrite2(img1, (255, 0, 0), body_parts['right_ear'], body_parts['right_shoulder'],
                     body_parts['right_elbow'])
        cv.imwrite(save_weight_path, img1)
