import cv2
import queue
from sourcece import kcftracker
import time
import numpy as np

onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0
duration = 0.01
fpslist=[]

# 创建一个空帧，定义(700, 700, 3)画图区域
frame = np.zeros((700, 700, 3), np.uint8)

# 初始化测量坐标和鼠标运动预测的数组
last_measurement = current_measurement = np.array((2, 1), np.float32)
last_prediction = current_prediction = np.zeros((2, 1), np.float32)
kalman = cv2.KalmanFilter(4, 2) # 4：状态数，包括（x，y，dx，dy）坐标及速度（每次移动的距离）；2：观测量，能看到的是坐标值
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32) # 系统测量矩阵
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) # 状态转移矩阵
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)*0.03 # 系统过程噪声协方差
print("fsdfdsgdf")
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
    cv2.line(frame, (lmx, lmy), (cmx, cmy), (255, 0, 0)) # 蓝色线为测量值
    cv2.line(frame, (lpx, lpy), (cpx, cpy), (255, 0, 255)) # 粉色线为预测值

    # print(cpy)
    cylist=[]
    cylist.append(str(cpx)+","+str(cpy))
    return cylist
# 获取检测框参数
# 雷达飞机
# test=[0,0,127, 226, 26, 19]
test=[0,0,74,86,56,48]
# test=[0,0,58,100,28,23]
Box=[(test[2],test[3])]

Box.append(((test[4]+test[2]),test[5]+test[3]))
# 打开视频流
cap = cv2.VideoCapture("B:\Document\Code\\586\KCFTest\\testVideo\\dog.mp4")
ok, frame = cap.read()
Begin_train_flag=True
# 勾勒初帧跟踪框位置
cv2.rectangle(frame, Box[0],Box[1], (0,0,205), 2)
tracker = kcftracker.KCFTracker(True, True, True)
ix=Box[0][0]
iy=Box[0][1]
cx=Box[1][0]
cy=Box[1][1]
history_queue_frame=queue.Queue()
# CORE=calc_coordinate_Core(Box)
# history_queue_frame.put(CORE)
# print(history_queue_frame.qsize())
kalmancounter=0
xylist=[]
while(cap.isOpened()):
    kalmancounter+=1
    ret, frame = cap.read()
    if not ret:
        break
    if(Begin_train_flag):
        cv2.rectangle(frame, (ix, iy), (cx, cy), (0, 0, 255), 2)
        # print([ix, iy, w, h])
        tracker.init([ix, iy, cx-ix, cy-iy], frame)
        Begin_train_flag=False
        onTracking = True
    elif(onTracking):
        # 如果卡尔曼和kcf差距不大则采用卡尔曼修改kcf方式，否则，采用kcf，这是因为前期需要测量
        t0 = time.time()
        boundingbox = tracker.update(frame)
        t1 = time.time()
        boundingbox = list(map(int, boundingbox))
        cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 0, 255), 2)
        duration = 0.8 * duration + 0.2 * (t1 - t0)
        corex=boundingbox[0]+boundingbox[2]/2
        corey=boundingbox[1]+boundingbox[3]/2

        # cv2.putText(frame, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        xypredict=move(frame,(boundingbox[0] + boundingbox[2]/2),(boundingbox[1] + boundingbox[3]/2))

        xylist.append(xypredict)
        # print(xypredict[0])
        duration = t1 - t0
    cv2.imshow('tracking', frame)
    c = cv2.waitKey(27) & 0xFF
    if c == 27 or c == ord('q'):
        break
for i in range(len(xylist)):
    print(xylist[i])
file_write_obj = open("B:\Document\Code\\586\KCFTest\\testVideo\ohjesus.txt", 'w')
for i in range(len(xylist)):
    file_write_obj.writelines(str(xylist[i]))
    file_write_obj.write('\n')
    print(xylist[i])