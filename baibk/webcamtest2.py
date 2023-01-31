import cv2 as cv
import numpy as np
from mmpose.apis import (init_pose_model, inference_bottom_up_pose_model, vis_pose_result)
import threading
from main import opencvwrite, opencvwrite2, caluculate_cos, cos2degree
import ast


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
gesture_file = open(gesture_file_save_path, 'r')
# gesture: {'0' : [14.992716975632574, 105.2110146802188... ,}
gesture = {x.strip().split('|')[0]: ast.literal_eval(x.strip().split('|')[1]) for x in gesture_file}

while (1):
    ret, img = cap.read()
    if not is_run:
        thread1 = detect_thread(1, "detect_thread", img)
        thread1.start()
    try:
        xy = np.array(pose_results[0]['keypoints'])[:, :2]
    except:
        continue
    body_parts = {'nose': xy[0], 'right_eye': xy[1], 'left_eye': xy[2], 'right_ear': xy[3], 'left_ear': xy[4],
                  'right_shoulder': xy[5], 'left_shoulder': xy[6], 'right_elbow': xy[7], 'left_elbow': xy[8],
                  'right_wrist': xy[9], 'left_wrist': xy[10], 'right_ass': xy[11], 'left_ass': xy[12],
                  'right_knee': xy[13], 'left_knee': xy[14], 'right_jiaohuai': xy[15], 'left_jiaohuai': xy[16]}
    try:
        detect_degrees = [cos2degree(x) for x in caluculate_cos(body_parts)]
        for gesture_key, gesture_value in gesture.items():
            results = [True if abs(detect_degrees[i] - gesture_value[i]) <= 20 else False for i in
                       range(len(detect_degrees))]
            count = 0
            for result in results:
                if result == True:
                    count += 1
            if count >= 10:
                print(gesture_list[int(gesture_key)])
                break
    except:
        continue
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
    # show a frame
    cv.imshow("video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
