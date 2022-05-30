import cv2
import time

from sourcece.KCF import KCFTracker
import numpy as np

last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)
kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差
fpslist=[]
def move(frame,x, y):
    global  current_measurement, measurements, last_measurement, current_prediction, last_prediction
    last_prediction = current_prediction # 把当前预测存储为上一次预测
    last_measurement = current_measurement # 把当前测量存储为上一次测量
    current_measurement = np.array([[np.float32(x)], [np.float32(y)]]) # 当前测量
    kalman.correct(current_measurement) # 用当前测量来校正卡尔曼滤波器
    current_prediction = kalman.predict() # 计算卡尔曼预测值，作为当前预测

    lmx, lmy = last_measurement[0], last_measurement[1] # 上一次测量坐标
    cmx, cmy = current_measurement[0], current_measurement[1] # 当前测量坐标
    lpx, lpy = last_prediction[0], last_prediction[1] # 上一次预测坐标
    cpx, cpy = current_prediction[0], current_prediction[1] # 当前预测坐标

    # 绘制从上一次测量到当前测量以及从上一次预测到当前预测的两条线
    cv2.line(frame, (lmx, lmy), (cmx, cmy), (255, 0, 0)) # 蓝色线为测量值
    cv2.line(frame, (lpx, lpy), (cpx, cpy), (255, 0, 255)) # 粉色线为预测值
    icxy=[cpx,cpy]
    return icxy
def ExeTracking():
    # global Img2
    cap = cv2.VideoCapture("B:\Document\Code\\586\KCFTest\\testVideo\dog.avi")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    the_first_frame_flag, the_first_frame = cap.read()
    cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
    pathIn = 'get_the_first_frame.jpg'
    pathOut = './Outputframes/frame_000001.jpg'
    # 狗
    test = [0, 0, 74,86,56,48]
    Box = [(test[2], test[3])]
    Box.append(((test[2] + test[4]), test[3] + test[5]))
    frameZoom = cv2.imread(pathIn)
    frameZoom2 = cv2.imread(pathOut)
    # print(str(Box[1]))
    cv2.rectangle(frameZoom, Box[0], Box[1], (0, 0, 205), 2)
    frame_count = 1
    frame_flag = True
    train_flag = True
    detect_flag = True
    tracker = KCFTracker(feature_type='raw', sub_feature_type='gray')
    count = 20
    framenum=1
    while (frame_flag):
        frame_flag, frameEach = cap.read()
        if(frame_flag):
            frame_clone = frameEach.copy()
            if train_flag:
                TrackerBox = [Box[0][0], Box[0][1], Box[1][0] - Box[0][0], Box[1][1] - Box[0][1]]
                start_time = time.time()
                tracker.train(frameEach, TrackerBox)
                total_time = time.time() - start_time
                train_flag = False
            else:
                start_time = time.time()
                res = tracker.detect(frame_clone)
                total_time = time.time() - start_time
                TrackingStr2 = ("Frames-per-second:" + str(1. / total_time) + "\n")
                print(TrackingStr2)
                fpslist.append((1./total_time))
                cv2.rectangle(frameEach, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])),
                              (0, 0, 205), 2)
                count = count - 1
                TrackingFrame_Pathout = ("./Outputframes/frame_00000" + str(framenum) + ".jpg")
                framenum += 1
                cv2.imwrite(TrackingFrame_Pathout, frameEach)
                move(frameEach,(res[0]+(res[2]/2)), (res[1]+(res[3]/2)))
            cv2.imshow("frame", frameEach)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break
ExeTracking()
