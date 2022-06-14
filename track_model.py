from PIL import Image, ImageTk
import tkinter
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
import tkinter.font as tkfont
import cv2
from sourcece.KCF import KCFTracker
from sourcece import kcftracker, code
import time
from tkinter import messagebox
import os
import shutil
import tkinter.filedialog
from AIDetector_pytorch import Detector
from sourcece.kalman_yolobox import KalmanYolo
import numpy as np
import math

det = Detector()


def yolo_track(frameEach, miss_flag, middle_miss_flag, yolodetect_list, yolopredict_list, w_list, h_list,
               miss_count):
    global det
    print('yolo track model')
    im, result = det.detect(frameEach)

    if len(result) > 0:
        # 目标从边界消失后返回视野，以yolo检测结果为准进行tracker初始化
        if (miss_flag == True) and (middle_miss_flag == True):
            Begin_train_flag = True
            miss_flag = False
            middle_miss_flag = False
            yolocount = 0
            ix = result[0][0]
            iy = result[0][1]
            cx = result[0][2] - result[0][0]
            cy = result[0][3] - result[0][1]


        print('yolo track model--success')
        miss_flag = False
        yolodetect_list.append([result[0][0], result[0][1]])
        # distance_min = 0
        x = result[0][0]
        y = result[0][1]
        w = result[0][2] - result[0][0]
        h = result[0][3] - result[0][1]
        w_list.append(w)
        h_list.append(h)
        cp_pre, current_prediction = KalmanYolo(yolodetect_list[-1], yolopredict_list[-1])
        #  如果yolo预测值和检测值差距过大，认为yolo检测失败
        if (math.fabs(cp_pre[0] - x) > 300) or (math.fabs(cp_pre[1] - y)
                                                > 300):
            print('yolo track model--fail')
            del yolodetect_list[-1]
            del w_list[-1]
            del h_list[-1]

            miss_count += 1
            if miss_count >= 3:
                miss_flag = True

            # 目标不是飞出视野，而是在视野内某个位置被遮挡而丢失，
            # 则不断对yolo进行卡尔曼更新
            if (cp_pre[0] + w_list[-1] < im.shape[1] - 200) & (cp_pre[0]
                                                               > 200) & (cp_pre[1] + h_list[-1] < im.shape[0] - 200) & (
                    cp_pre[1] > 200):
                yolopredict_list.append(cp_pre)
                yolodetect_list.append(cp_pre)
            # 目标飞出视野外丢失，yolo相关列表全部清空
            elif miss_flag:
                middle_miss_flag = True
                yolodetect_list = []
                yolopredic_list = []
                w_list = []
                h_list = []

        # for i in range(len(result)):    #选取离上一个yolo框最近的yolo框
        #     dx_temp = result[i][0] - yolodetect_list[-1][0]
        #     dy_temp = result[i][1] - yolodetect_list[i][1]
        #     distance_temp = dx_temp*dx_temp + dy_temp*dy_temp
        #     if distance_temp < distance_min:
        #         yolodetect_list[-1] = [result[i][0],result[i][1]]
        #         w = result[i][2] - result[i][0]
        #         h = result[i][3] - result[i][1]
        #         test = [0, 1, result[i][0], result[i][1], w, h]

        if w >= 20:  # 若宽度超过20，下帧开始进入kcf模式
            yolo_flag = False
            tracker.init([x, y, w, h], im)

        t1 = time.time()
        if w != 0 and h != 0:
            sub_frame = frameEach[y: y + h, x: x + w]
        cv2.rectangle(frameEach, (x, y), (x + w, y + h), (0, 255, 0), 2)
        duration = 0.01 * duration + 0.99 * (t1 - t0)
        KCFGroundList.append(test)
        cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255), 2)
    else:
        print('yolo track model--fail')
        miss_count += 1
        if miss_count >= 3:
            miss_flag = True
        # 目标消失在视野外
        if middle_miss_flag == False and miss_flag == True:
            t1 = time.time()
            duration = 0.01 * duration + 0.99 * (t1 - t0)
            KCFGroundList.append(test)
            cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255), 2)
        else:
            cp_pre, current_prediction = KalmanYolo(yolodetect_list[-1], yolopredict_list[-1])
            if (cp_pre[0] + w_list[-1] < im.shape[1] - 200) & (cp_pre[0]
                                                               > 200) & (
                    cp_pre[1] + h_list[-1] < im.shape[0] - 200) & (
                    cp_pre[1] > 200):
                yolopredict_list.append(cp_pre)
                yolodetect_list.append(cp_pre)
            # 目标飞出视野外丢失，yolo相关列表全部清空
            elif miss_flag:
                middle_miss_flag = True
                yolodetect_list = []
                yolopredic_list = []
                w_list = []
                h_list = []