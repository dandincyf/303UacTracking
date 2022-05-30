from sourcece import KCF2
import cv2
import time
import math
import numpy as np
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)
kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差
def move( x, y):
    global frame, current_measurement, measurements, last_measurement, current_prediction, last_prediction
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
def  Coreerroranalysis(corekcftracker,corekalman):
    DValuex=abs(corekcftracker[0]-corekalman[0]) ** 2
    DValuey=abs(corekcftracker[1]-corekalman[1]) ** 2
    SQUValue=math.sqrt(DValuex+DValuey)
    print("kkk")
    print(SQUValue)
    print("kkk")
    return SQUValue

# 计算中心位置函数
def calc_coordinate_Core(Box):
    Xcore=int((Box[1][0]+Box[0][0])/2)
    Ycore=int((Box[1][1]+Box[0][1])/2)
    CORE=[Xcore,Ycore]
    return CORE

'''
pathIn='./Inputframes/frame_000001.jpg'
pathOut='./Outputframes/frame_000001.jpg'
test=code.yolo_detect(pathIn,pathOut)
# print(str(test[0])+"/"+str(test[1])+"/"+str(test[2])+"/"+str(test[3]))
Box=[(test[0],test[1])]
Box.append(((test[0]+test[2]),test[1]+test[3]))
frameZoom=cv2.imread("./Inputframes/frame_000001.jpg")
# print(str(Box[1]))
cv2.rectangle(frameZoom, Box[0],Box[1], (0,0,205), 2)
cap=cv2.VideoCapture("./test2.mp4")
frame_count=1
frame_flag=True
train_flag=True
detect_flag=True
tracker = KCFTracker(feature_type='raw', sub_feature_type='gray')'''
pathIn='./Inputframes/frame_000001.jpg'
pathOut='./Outputframes/frame_000001.jpg'
# test=code.yolo_detect(pathIn,pathOut)
# print(str(test[0])+"/"+str(test[1])+"/"+str(test[2])+"/"+str(test[3]))
# 74	86	56	48
test=[0,0,127, 226, 26, 19]
Box=[(test[2],test[3])]
Box.append(((test[2]+test[4]),test[3]+test[5]))
# frameZoom=cv2.imread("./Inputframes/frame_000001.jpg")
# print(str(Box[1]))
# cv2.rectangle(frame, Box[0],Box[1], (0,0,205), 2)
tracker = KCF2.KCFTracker()
video = cv2.VideoCapture('B:\Document\Code\\586\KCFTest\\testVideo\\areoplane_of_tinyandShelter.mp4')
ok, frame = video.read()

# bbox = cv2.selectROI(frame, False)
bbox=(test[2],test[3],test[4],test[5])
frames = 0
print(bbox)
#print(frame)
start = time.time()
tracker.init(bbox,frame)
kalmancounter=0
count=50
while True:
    _,frame = video.read()
    if(_):
        kalmancounter += 1
        count-=1
        peak_value, item = tracker.update(frame)
        frames += 1
        # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        #print(item)
        x = int(item[0])
        y = int(item[1])
        w = int(item[2])
        h = int(item[3])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        icxy = move(int(item[0] + item[0] + item[2]) / 2,
                    int(item[1] + item[1] + item[3]) / 2)

        corekcftracker = [int((item[0] + item[2] + item[0]) / 2),
                          int((item[1] + item[1] + item[3]) / 2)]
        corekalman = [int((icxy[0] + icxy[0] + item[2]) / 2), int((icxy[1] + icxy[1] + item[3]) / 2)]
        # print("kalman")
        # print(icxy[0])
        # print(boundingbox[2])
        SVALUEXY = Coreerroranalysis(corekcftracker, corekalman)
        print(SVALUEXY)
        if (SVALUEXY > 100 and kalmancounter >= 30):
            ix = int(icxy[0]) - int(item[1] / 2)
            iy = int(icxy[1]) - int(item[3] / 2)
            cx = int(icxy[0]) + int(item[1] / 2)
            cy = int(icxy[1]) + int(item[3] / 2)
            Begin_train_flag = True
        cv2.imshow("track",frame)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break