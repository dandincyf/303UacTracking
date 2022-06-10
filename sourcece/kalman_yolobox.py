import cv2
import numpy as np

'''
该模块主要用来判断YOLO的结果是否可靠，若不可靠，则相信kcf。
判断依据：YOLO当前测量值相比于YOLO卡尔曼先验估计值变化较大，说明YOLO检测错误，此时尽管YOLO检测框距离kcf框较远，但仍以kcf框为准。
'''

def KalmanYolo(cm,lp): # c:current, m:measurenment, l:last, p:prediction

    #初始化测量值、预测值
    # last_measurement = current_measurement = np.zeros((2, 1), np.float32)
    # last_prediction = current_prediction = np.zeros((4, 1), np.float32)

    current_measurement = np.array([[cm[0]],[cm[1]]], np.float32)
    # print('current measurement == ',current_measurement)

    last_prediction = np.array([[lp[0]],[lp[1]],[lp[2]],[lp[3]]], np.float32)
    current_prediction = np.zeros((4, 1), np.float32)

    # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）     2：观测量，能看到的是坐标值
    kalman = cv2.KalmanFilter(4, 2)

    # 系统测量矩阵 H
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)

    # 状态转移矩阵 A(F)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    # 系统过程噪声协方差 Q ,比较相信预测模型，故Q取值较小，0.0001
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                      np.float32) * 0.01

    # 设置先验误差协方差矩阵 P_pre
    kalman.errorCovPre = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    # 测量噪声协方差矩阵 R
    kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.0001

    kalman.correct(current_measurement)
    current_prediction = kalman.predict()
    # print('kalman: current prediction==',current_prediction)
    current_prediction = current_prediction.reshape(-1).tolist()

    #状态量的先验估计值
    cp_pre = np.dot(kalman.transitionMatrix, last_prediction)
    cp_pre = cp_pre.reshape(-1).tolist()

    return cp_pre, current_prediction

