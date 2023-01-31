import threading

from mmpose.apis import inference_top_down_pose_model

from main import *
from sklearn import svm


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
        pose_results, _ = inference_top_down_pose_model(pose_model, self.img)
        is_run = False


pose_results = None
is_run = False
config_file = 'model/mspn50_coco_256x192.py'
checkpoint_file = 'model/mspn50_coco_256x192-8fbfb5d0_20201123.pth'
gesture_file_save_path = './classify/many_gesture_value.txt'
device='cpu'
pose_model = init_pose_model(config_file, checkpoint_file, device=device)  # or device='cuda:0'
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
# LogisticRegression 分类
# LR = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
# LR.fit(gesture_degrees, gesture_targets)

SV = svm.SVC(probability=True)
SV.fit(gesture_degrees, gesture_targets)

while (1):
    ret, img = cap.read()
    if not is_run:
        thread1 = detect_thread(1, "detect_thread", img)
        thread1.start()
    try:
        xy = np.array(pose_results[0]['keypoints'])[:, :2]
        # score = pose_results[0]['score']
        # # 关键点分数小于0.3则不予显示
        # if score < 0.3:
        #     cv.imshow("video", img)
        # else:
        body_parts = get_body_parts(xy)
        detect_degrees = np.asarray([[cos2degree(x) for x in caluculate_cos(body_parts)]])
        # 数组中不包含nan才预测
        if not (True in np.isnan(detect_degrees)):
            detect_probability = SV.predict_proba(detect_degrees)
            if np.max(detect_probability, axis=1) > 0.9:
                print("SVM Result:", gesture_list[int(np.argmax(detect_probability, axis=1))])
        draw_angles(img, body_parts)
        cv.imshow("video", img)
    except:
        cv.imshow("video", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
