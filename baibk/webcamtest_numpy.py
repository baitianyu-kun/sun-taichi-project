import cv2 as cv
import numpy as np
from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)
import threading
from main import *
import ast
from collections import Counter


class detect_thread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, threadID, name, img):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.img = img

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        global is_run
        is_run = True
        global pose_results
        pose_results, _ = inference_bottom_up_pose_model(pose_model, self.img)
        is_run = False


pose_results = None
is_run = False
config_file = '../model/associative_embedding_hrnet_w32_coco_512x512.py'
checkpoint_file = '../model/hrnet_w32_coco_512x512-bcb8c247_20200816.pth'
gesture_file_save_path = '../imgs/process_pics_sun_taichi/gesture_value.txt'
pose_model = init_pose_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
cap = cv.VideoCapture(0)
cv.namedWindow("video", 0)  # 0 即 cv.WINDOW_NORMAL，表示可以自己调整窗口大小。注意：此“winname”参数应与后面的inshow()中一致。
cv.resizeWindow("video", 960, 720)  # 修改窗口大小为960X720
gesture_list = ['更鸡独立', '右蹬腿', '手挥琵琶', '白鹤亮翅', '倒撵猴', '高探马', '上步搬拦捶', '单鞭', '右通背',
                '玉女穿梭', '全身']
gesture_file_data = np.loadtxt(gesture_file_save_path, delimiter=',')
# [[ 38.3051958  105.21101468  61.44069537....],[]...]
gesture_degrees = gesture_file_data[:, 0:11]
# [ 0.  9.  1.  2.  3.  4.  5.  6.  7.  8. 10.]
gesture_targets = gesture_file_data[:, 11].astype(int)

while (1):
    ret, img = cap.read()
    if not is_run:
        thread1 = detect_thread(1, "detect_thread", img)
        thread1.start()
    try:
        xy = np.array(pose_results[0]['keypoints'])[:, :2]
    except:
        continue
    body_parts = get_body_parts(xy)
    try:
        detect_degrees = np.asarray([cos2degree(x) for x in caluculate_cos(body_parts)])
        for gesture_index in range(len(gesture_degrees)):
            results = np.abs(gesture_degrees[gesture_index] - detect_degrees)
            results = np.linalg.norm(results)
            print(results)
            if results < 70:
                print(gesture_list[gesture_targets[gesture_index]])
                break
    except:
        continue
    draw_angles(img, body_parts)
    # show a frame
    cv.imshow("video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
