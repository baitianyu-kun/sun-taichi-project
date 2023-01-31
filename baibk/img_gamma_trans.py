import os

import cv2 as cv
import numpy as np
import math


def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def read_get_pic(file,save_file,guiyihua_xiangsu):
    img_gray = cv.imread(file, 0)  # 灰度图读取，用于计算gamma值
    img = cv.imread(file)  # 原图读取A
    mean = np.mean(img_gray)
    gamma_val = math.log10(guiyihua_xiangsu) / math.log10(mean / 255)  # 公式计算gamma
    image_gamma_correct = gamma_trans(img, gamma_val)  # gamma变换
    cv.imwrite(save_file, image_gamma_correct)
    cv.waitKey(0)

file= '../imgs/test_pics/gang1.jpg'
save_file='gang1_gamma.jpg'
read_get_pic(file,save_file,0.69)



