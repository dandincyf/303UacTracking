# from PIL import Image,ImageTk
# from siamrpn import TrackerSiamRPN
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
test = []
TrackingBoundingBoxPath = ""
fps = 20
first_detect_times = 1


# yolodetect_list = [] #装yolo检测值
# yolopredic_list = [[0,0,0,0]] #装yolo的卡尔曼最佳估计值,初始化

Windows = tkinter.Tk()

Windows.title("ODTS")
# 主界面窗口大小，建议不要全屏放大窗口
Windows.geometry("1301x650")
Algorithm_name = ""
Test_Video_Name = ""
Test_RecGroundTruth_Name = ""

'''
按钮算法连接
'''


def ExeDetection():
    global test
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.insert(tkinter.END, "*********开始执行检测识别**********\n")
    Text_TrackingLog.update()
    global Img
    global first_detect_times
    # yolo检100次直到检出目标
    for i in range(1, 100):
        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        the_first_frame_flag, the_first_frame = cap.read()
        cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
        pathIn = './get_the_first_frame.jpg'
        im, result = det.detect(the_first_frame)
        if len(result) > 0:
            first_detect_times = i
            break
    x1=result[0][0]
    y1=result[0][1]
    x2=result[0][2]
    y2=result[0][3]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('C:/dataset/KCF_Test_results/Outputframes_temp/frame_000001.jpg', im)
    pathOut = 'C:/dataset/KCF_Test_results/Outputframes_temp/frame_000001.jpg'


    w = result[0][2] - result[0][0]
    h = result[0][3] - result[0][1]
    test = [0, 1, result[0][0], result[0][1], w, h]

    Text_DetectLog.config(state="normal")
    Text_DetectLog.delete('1.0', 'end')
    Text_DetectLog.config(state='disabled')
    Img_FirstFrame = Image.open(pathOut)
    Img_FirstFrame = Img_FirstFrame.resize((480, 320), Image.ANTIALIAS)
    Img = ImageTk.PhotoImage(Img_FirstFrame)
    Label_FirstFrameShow.config(image=Img)
    Label_FirstFrameShow.image = Img
    Label_FirstFrameShow.update()


'''***********************************************************************************************'''
'''
跟踪算法
'''


def ExeTracking():
    print(Algorithm_name)
    print(Test_Video_Name)
    print(Test_RecGroundTruth_Name)
    global TrackingBoundingBoxPath
    global test
    global first_detect_times
    yolo_flag = False
    yololist = []
    misscount = 0
    jumptimes = 0
    jumplist = [] # 放jump时yolo结果
    jump_delta = [] # 放jump时相邻帧yolo结果差值
    jump_same_flag = False # jump时，yolo检测的是否是目标
    miss_flag = False
    edgemiss_flag = False # 边缘miss，在满足miss_flag后，还需满足在边缘的条件
    if (Algorithm_name == "KCF with adaptive frames" and Test_RecGroundTruth_Name != ""):
        TrackingBoundingBoxPath = "./sourcece/KCF_withAdaptiveFrames.txt"
        selectingObject = False
        initTracking = False
        onTracking = False
        ix, iy, cx, cy = -1, -1, -1, -1
        w, h = 0, 0
        duration = 0.01
        Performancelistx = []
        Performancelisty = []
        KCFGroundList = []
        RecGroundList = []
        IOUlistx = []
        IOUlisty = []
        dogmp4IOUfilename1 = Test_RecGroundTruth_Name
        file = open(dogmp4IOUfilename1, "r")
        line = file.readline()
        while line:
            # print(line)
            line = file.readline()
            line = line.rstrip("\n")
            RecGroundList.append(line)
        file.close()

        def draw_boundingbox(event, x, y, flags, param):
            global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
            if event == cv2.EVENT_LBUTTONDOWN:
                selectingObject = True
                onTracking = False
                ix, iy = x, y
                cx, cy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                cx, cy = x, y
            elif event == cv2.EVENT_LBUTTONUP:

                selectingObject = False
                if (abs(x - ix) > 10 and abs(y - iy) > 10):
                    w, h = abs(x - ix), abs(y - iy)
                    ix, iy = min(x, ix), min(y, iy)
                    initTracking = True
                else:
                    onTracking = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                onTracking = False
                if (w > 0):
                    ix, iy = x - w / 2, y - h / 2
                    initTracking = True

        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        the_first_frame_flag, the_first_frame = cap.read()
        cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
        pathIn = 'get_the_first_frame.jpg'
        pathOut = './Outputframes/frame_000001.jpg'
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        # 即Box=[(x1,y1),(x2,y2)]
        tracker = kcftracker.KCFTracker(True, True, True)
        ix = Box[0][0]
        iy = Box[0][1]
        cx = Box[1][0]
        cy = Box[1][1]
        Begin_train_flag = True
        framenum = 0
        print(framenum)
        Text_TrackingLog.insert(tkinter.END, '***********执行跟踪程序*************\n')
        Missflag = False
        count = 120
        while (cap.isOpened()):
            Text_TrackingLog.config(state='normal')
            sub_frame = []
            if (count == 0):
                TrackingStr2 = ("**************自动重检**************\n")
                test = code.yolo_detect(pathIn, pathOut)
                print("test:", test)
                Box = [(test[2], test[3])]
                Box.append((test[2] + test[4], test[3] + test[5]))
                tracker = kcftracker.KCFTracker(True, True, True)
                ix = Box[0][0]
                iy = Box[0][1]
                cx = Box[1][0]
                cy = Box[1][1]
                Begin_train_flag = True
                Text_TrackingLog.insert(tkinter.END, TrackingStr2)
                Text_TrackingLog.update()
                count = 120
            Text_TrackingLog.update()
            ret, frameEach = cap.read()
            if not ret:
                break
            if (Begin_train_flag):
                framenum += 1
                if w == 0 or h == 0:
                    sub_frame = frameEach
                else:
                    sub_frame = frameEach[ix:ix + w, iy:iy + h]
                cv2.rectangle(frameEach, (ix, iy), (ix + w, iy + h), (0, 0, 255), 2)
                tracker.init([ix, iy, cx - ix, cy - iy], frameEach)
                Begin_train_flag = False
                onTracking = True
                KCFGroundList.append([ix, iy, cx - ix, cy - iy])
            elif (onTracking):
                framenum += 1
                t0 = time.time()
                boundingbox = tracker.update(frameEach)
                print(boundingbox)
                t1 = time.time()
                boundingbox = list(map(int, boundingbox))
                print('boundingbox_1:', boundingbox)
                if boundingbox[2] != 0 and boundingbox[3] != 0:
                    sub_frame = frameEach[boundingbox[1]: boundingbox[1] + boundingbox[3],
                                boundingbox[0]: boundingbox[0] + boundingbox[2]]
                cv2.rectangle(frameEach, (boundingbox[0], boundingbox[1]),
                              (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 0, 255), 2)

                duration = 0.8 * duration + 0.2 * (t1 - t0)
                # duration = t1 - t0
                cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
            TrackingFrame_Pathout = ("./Outputframes/frame_00000" + str(framenum) + ".jpg")
            FirstFrame_Pathout = ("./Outputframes/sub_frame_00000" + str(framenum) + ".jpg")

            cv2.imwrite(TrackingFrame_Pathout, frameEach)
            cv2.imwrite(FirstFrame_Pathout, sub_frame)

            Img2 = Image.open(TrackingFrame_Pathout)
            Img2 = Img2.resize((600, 400), Image.ANTIALIAS)
            Img2 = ImageTk.PhotoImage(Img2)

            Img3 = Image.open(FirstFrame_Pathout)
            Img3 = Img3.resize((300, 200), Image.ANTIALIAS)
            Img3 = ImageTk.PhotoImage(Img3)

            Label_FirstFrameShow.config(image=Img3)
            Label_FirstFrameShow.image = Img3
            Label_FirstFrameShow.update()
            Label_ObjectTrackingShow.config(image=Img2)
            Label_ObjectTrackingShow.image = Img2
            Label_ObjectTrackingShow.update()

            Text_TrackingLog.delete(1.0, tkinter.END)
            detect_fps = str(int(1 / duration))
            Text_content = '目标信息:\t{0},\t{1}\n目标方位:\t{2},\t{3}\n跟踪帧率:\t{4} \t置信率:\t{5}\n云台工作模式:\t{6}\n'.format(framenum,
                                                                                                               framenum,
                                                                                                               framenum,
                                                                                                               framenum,
                                                                                                               detect_fps,
                                                                                                               framenum,
                                                                                                               framenum)
            Text_TrackingLog.insert(tkinter.END, Text_content)
            Text_TrackingLog.see(tkinter.END)

            c = cv2.waitKey(27) & 0xFF
            if c == 27 or c == ord('q'):
                break

        '''
        将跟踪框位置写入文件
       '''
        file_write_obj = open("./framespositionstore/KCF_withAdaptiveFrames.txt", 'w')
        for i in range(len(KCFGroundList)):
            # print(strllp2[i] + "已经写入")
            file_write_obj.writelines(str(KCFGroundList[i]))
            file_write_obj.write('\n')
        file_write_obj.close()

        Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
        Text_TrackingLog.update()
        Text_TrackingLog.config(state='disabled')
        cap.release()
    elif (Algorithm_name == "KCF with adaptive frames" and Test_RecGroundTruth_Name == ""):
        selectingObject = False
        initTracking = False
        onTracking = False
        ix, iy, cx, cy = -1, -1, -1, -1
        w, h = 0, 0
        duration = 0.01
        Performancelistx = []
        Performancelisty = []
        KCFGroundList = [[0, 0, 0, 0]]

        cap = cv2.VideoCapture(Test_Video_Name)
        # 从first_detect_times开始读取，用yolo来检测
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_detect_times)
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        tracker = kcftracker.KCFTracker(True, True, True)
        ix = Box[0][0]
        iy = Box[0][1]
        cx = Box[1][0]
        cy = Box[1][1]
        Begin_train_flag = True
        framenum = 1
        Text_TrackingLog.insert(tkinter.END, '***********执行跟踪程序*************\n')
        yolocount = 0
        while (cap.isOpened()):
            Text_TrackingLog.config(state='normal')
            Text_TrackingLog.update()
            ret, frameEach = cap.read()
            sub_frame = []
            if not ret:
                break
            if (Begin_train_flag):
                if w == 0 or h == 0:
                    sub_frame = frameEach
                else:
                    sub_frame = frameEach[ix:ix + w, iy:iy + h]
                cv2.rectangle(frameEach, (ix, iy), (ix + w, iy + h), (0, 0, 255), 2)
                tracker.init([ix, iy, cx - ix, cy - iy], frameEach)
                Begin_train_flag = False
                onTracking = True
            elif (onTracking):
                yolocount = yolocount + 1
                t0 = time.time()
                print(yolocount)

                # 每隔100帧，清理一次列表，防止降低运行速度
                if yolocount % 100 == 0:
                    yololist = [yololist[-2], yololist[-1]]
                    KCFGroundList = [KCFGroundList[-2], KCFGroundList[-1]]

                # kcf跟踪模式
                if yolo_flag == False:
                    boundingbox = tracker.update(frameEach)
                    boundingbox = list(map(int,boundingbox))
                    x_kcf = boundingbox[0]
                    y_kcf = boundingbox[1]
                    w_kcf = boundingbox[2]
                    h_kcf = boundingbox[3]
                    if w_kcf < 10:
                        yolo_flag = True
                    # 如果jumptimes > 0，每帧执行一次yolo
                    if jumptimes > 0:
                        im, result = det.detect(frameEach)
                        # print('im.shape = ', im.shape[1], ' ', im.shape[0])
                        if (len(result) > 0):  # 如果yolo能检测出东西
                            x_yolo = result[0][0]
                            y_yolo = result[0][1]
                            w_yolo = result[0][2] - result[0][0]
                            h_yolo = result[0][3] - result[0][1]
                            distance_yolo = 0
                            yololist.append([x_yolo, y_yolo, w_yolo, h_yolo])
                            if len(yololist) > 1:  # 不是第一帧yolo
                                dx_yolo = x_yolo - yololist[-2][0]
                                dy_yolo = y_yolo - yololist[-2][1]
                                distance_yolo = dx_yolo * dx_yolo + dy_yolo * dy_yolo
                                # 选出一个距离上帧yolo框最近的yolo框
                                for i in range(1, len(result)):
                                    dx_temp = result[i][0] - yololist[-2][0]
                                    dy_temp = result[i][1] - yololist[-2][1]
                                    distance_temp = dx_temp * dx_temp + dy_temp * dy_temp
                                    if distance_temp < distance_yolo:
                                        x_yolo = result[i][0]
                                        y_yolo = result[i][1]
                                        w_yolo = result[i][2] - result[i][0]
                                        h_yolo = result[i][3] - result[i][1]
                                        distance_yolo = distance_temp
                                        yololist[-1] = [x_yolo, y_yolo, w_yolo, h_yolo]

                            # 如果yolo跳变，认为yolo暂时不可靠，和kcf比较，不考虑miss，miss后只会激活yolo跟踪模式
                            # 通过jumptimes进一步判断yolo是否可靠
                            if (math.fabs(x_yolo - x_kcf) > 200) or (math.fabs(y_yolo - y_kcf) > 200):
                                del yololist[-1]
                                # yolo不可靠也可以理解为yolo没有检测出目标，因此misscount+1
                                misscount += 1
                                if misscount >= 3:
                                    miss_flag = True
                                    yolo_flag = True
                                # 更新KCFGroundList
                                KCFGroundList.append([x_kcf, y_kcf, w_kcf, h_kcf])

                                jumptimes += 1
                                if jumptimes >= 4:  # 看连续3帧是否接近，若接近，认为是目标
                                    jump_same_flag = True
                                    for i in jump_delta:
                                        if math.fabs(i) > 50:
                                            jump_same_flag = False
                                    if jump_same_flag:  # yolo结果可靠
                                        yololist.append([x_yolo, y_yolo, w_yolo, h_yolo])
                                        misscount = 0
                                        jump_same_flag = False
                                        jumptimes = 0
                                        jump_delta = []
                                        jumplist = []
                                        KCFGroundList.append(yololist[-1])
                                        # yolo画框
                                        cv2.rectangle(frameEach, (x_yolo, y_yolo), (x_yolo + w_yolo,
                                                                                    y_yolo + h_yolo), (0, 255, 0), 2)
                                        t1 = time.time()
                                        duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                                        sub_frame = frameEach[y_kcf: y_kcf + h_kcf, x_kcf: x_kcf + w_kcf]
                                        cv2.rectangle(frameEach, (x_kcf, y_kcf), (x_kcf + w_kcf, y_kcf + h_kcf),
                                                      (0, 0, 255), 2)
                                        cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        cv2.putText(frameEach, 'kcf model: yolo jump, success', (8, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        cv2.putText(frameEach, 'edgemiss = ' +
                                                    str(edgemiss_flag), (8, 80),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        continue  # 因为flag值改变，不执行后续语句
                                    # 每隔3帧清空一次
                                    jumptimes = 0
                                    jump_delta = []
                                    jumplist = []
                                elif (jumptimes == 1):
                                    jumplist.append(x_yolo)
                                    jumplist.append(y_yolo)
                                elif (jumptimes > 1):
                                    jump_delta.append(jumplist[-2] - x_yolo)
                                    jump_delta.append(jumplist[-1] - y_yolo)
                                    jumplist.append(x_yolo)
                                    jumplist.append(y_yolo)
                                print('kcf model: yolo fail, jump')
                                cv2.putText(frameEach, 'kcf model: yolo jump, fail', (8, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'edgemiss = ' +
                                            str(edgemiss_flag), (8, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:  # yolo不跳变，可靠
                                jumptimes = 0
                                jump_delta = []
                                jumplist = []
                                misscount = 0
                                miss_flag = True

                                # # 判断kcf是否偏离
                                # if (math.fabs(x_kcf - x_yolo) > 100) or (math.fabs(y_kcf - y_yolo)) or \
                                #     (float(w_kcf) / float(w_yolo) < 0.7) or (float(w_yolo) / float(w_kcf) < 0.7):
                                #     tracker.init([x_yolo, y_yolo, w_yolo, h_yolo], im)

                                # 不用判断kcf是否偏离，只要yolo可靠，就用yolo更新
                                tracker.init([x_yolo, y_yolo, w_yolo, h_yolo], im)
                                # yolo画绿框
                                cv2.rectangle(frameEach, (x_yolo, y_yolo), (x_yolo + w_yolo, y_yolo + h_yolo),
                                              (0, 255, 0), 2)
                                # 更新KCFGroundList
                                KCFGroundList.append(yololist[-1])
                                cv2.putText(frameEach, 'kcf model: yolo, success', (8, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'edgemiss = ' +
                                            str(edgemiss_flag), (8, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print('kcf model: yolo success')
                        else:  # yolo检测不到东西
                            jumptimes = 0
                            jump_delta = []
                            jumplist = []
                            misscount += 1
                            if misscount >= 3:
                                miss_flag = True
                                yolo_flag = True  # miss后进入yolo模式
                            print('kcf model: yolo fail, nothing')
                            cv2.putText(frameEach, 'kcf model: yolo nothing, fail', (8, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(frameEach, 'edgemiss = ' +
                                        str(edgemiss_flag), (8, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # 如果jumptimes=0，每10帧执行一次yolo
                    elif yolocount % 10 == 0:
                        im, result = det.detect(frameEach)
                        # print('im.shape = ', im.shape[1], ' ', im.shape[0])
                        if (len(result) > 0):  # 如果yolo能检测出东西
                            x_yolo = result[0][0]
                            y_yolo = result[0][1]
                            w_yolo = result[0][2] - result[0][0]
                            h_yolo = result[0][3] - result[0][1]
                            distance_yolo = 0
                            yololist.append([x_yolo, y_yolo, w_yolo, h_yolo])
                            if len(yololist) > 1:  # 不是第一帧yolo
                                dx_yolo = x_yolo - yololist[-2][0]
                                dy_yolo = y_yolo - yololist[-2][1]
                                distance_yolo = dx_yolo * dx_yolo + dy_yolo * dy_yolo
                                # 选出一个距离上帧yolo框最近的yolo框
                                for i in range(1, len(result)):
                                    dx_temp = result[i][0] - yololist[-2][0]
                                    dy_temp = result[i][1] - yololist[-2][1]
                                    distance_temp = dx_temp * dx_temp + dy_temp * dy_temp
                                    if distance_temp < distance_yolo:
                                        x_yolo = result[i][0]
                                        y_yolo = result[i][1]
                                        w_yolo = result[i][2] - result[i][0]
                                        h_yolo = result[i][3] - result[i][1]
                                        distance_yolo = distance_temp
                                        yololist[-1] = [x_yolo, y_yolo, w_yolo, h_yolo]

                            # 如果yolo跳变，认为yolo暂时不可靠，和kcf比较，不考虑miss，
                            # miss后只会激活yolo跟踪模式，通过jumptimes进一步判断yolo是否可靠
                            if (math.fabs(x_yolo - x_kcf) > 200) or (math.fabs(y_yolo - y_kcf) > 200):
                                del yololist[-1]
                                # yolo不可靠也可以理解为yolo没有检测出目标，因此misscount+1
                                misscount += 1
                                if misscount >= 3:
                                    miss_flag = True
                                    yolo_flag = True
                                # 更新KCFGroundList
                                KCFGroundList.append([x_kcf, y_kcf, w_kcf, h_kcf])

                                jumptimes += 1
                                if jumptimes >= 4:  # 看连续3帧是否接近，若接近，认为是目标
                                    jump_same_flag = True
                                    for i in jump_delta:
                                        if math.fabs(i) > 50:
                                            jump_same_flag = False
                                    if jump_same_flag:  # yolo结果可靠
                                        yololist.append([x_yolo, y_yolo, w_yolo, h_yolo])
                                        misscount = 0
                                        jump_same_flag = False
                                        jumptimes = 0
                                        jump_delta = []
                                        jumplist = []
                                        KCFGroundList.append(yololist[-1])
                                        # yolo画框
                                        cv2.rectangle(frameEach, (x_yolo, y_yolo), (x_yolo + w_yolo,
                                                                                    y_yolo + h_yolo), (0, 255, 0), 2)
                                        t1 = time.time()
                                        duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                                        sub_frame = frameEach[y_kcf: y_kcf + h_kcf, x_kcf: x_kcf + w_kcf]
                                        cv2.rectangle(frameEach, (x_kcf, y_kcf), (x_kcf + w_kcf, y_kcf + h_kcf),
                                                      (0, 0, 255), 2)
                                        cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        cv2.putText(frameEach, 'kcf model: yolo jump, success', (8, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        cv2.putText(frameEach, 'edgemiss = ' +
                                                    str(edgemiss_flag), (8, 80),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        continue  # 因为flag值改变，不执行后续语句
                                    # 每隔3帧清空一次
                                    jumptimes = 0
                                    jump_delta = []
                                    jumplist = []
                                elif (jumptimes == 1):
                                    jumplist.append(x_yolo)
                                    jumplist.append(y_yolo)
                                elif (jumptimes > 1):
                                    jump_delta.append(jumplist[-2] - x_yolo)
                                    jump_delta.append(jumplist[-1] - y_yolo)
                                    jumplist.append(x_yolo)
                                    jumplist.append(y_yolo)
                                print('kcf model: yolo fail, jump')
                                cv2.putText(frameEach, 'kcf model: yolo jump, fail', (8, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'edgemiss = ' +
                                            str(edgemiss_flag), (8, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:  # yolo不跳变，可靠
                                jumptimes = 0
                                jump_delta = []
                                jumplist = []
                                misscount = 0
                                miss_flag = True

                                # # 判断kcf是否偏离
                                # if (math.fabs(x_kcf - x_yolo) > 100) or (math.fabs(y_kcf - y_yolo)) or \
                                #     (float(w_kcf) / float(w_yolo) < 0.7) or (float(w_yolo) / float(w_kcf) < 0.7):
                                #     tracker.init([x_yolo, y_yolo, w_yolo, h_yolo], im)

                                # 不用判断kcf是否偏离，只要yolo可靠，就用yolo更新
                                tracker.init([x_yolo, y_yolo, w_yolo, h_yolo], im)
                                # yolo画绿框
                                cv2.rectangle(frameEach, (x_yolo, y_yolo), (x_yolo + w_yolo, y_yolo + h_yolo),
                                              (0, 255, 0), 2)
                                # 更新KCFGroundList
                                KCFGroundList.append(yololist[-1])
                                cv2.putText(frameEach, 'kcf model: yolo, success', (8, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'edgemiss = ' +
                                            str(edgemiss_flag), (8, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                # cv2.putText(frameEach, str(jumptimes), (8, 110),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print('kcf model: yolo success')
                        else:  # yolo检测不到东西
                            jumptimes = 0
                            jump_delta = []
                            jumplist = []
                            misscount += 1
                            if misscount >= 3:
                                miss_flag = True
                                yolo_flag = True  # miss后进入yolo模式
                            print('kcf model: yolo fail, nothing')
                            cv2.putText(frameEach, 'kcf model: yolo nothing, fail', (8, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(frameEach, 'edgemiss = ' +
                                        str(edgemiss_flag), (8, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else: # 只进行kcf检测
                        # kcf也不允许跳变
                        if (math.fabs(x_kcf - KCFGroundList[-1][0]) < 100) or (math.fabs(y_kcf -
                                    KCFGroundList[-1][1]) < 100):
                            KCFGroundList.append([x_kcf, y_kcf, w_kcf, h_kcf])
                            print('kcf model: only kcf, success')
                            cv2.putText(frameEach, 'kcf model: only kcf, success', (8, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(frameEach, 'edgemiss = ' +
                                        str(edgemiss_flag), (8, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            # cv2.putText(frameEach, str(jumptimes), (8, 110),
                            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else: # kcf跳变，概率很低但是存在，不画框
                            print('kcf model: only kcf, fail')
                            sub_frame = frameEach[500: 700, 500: 700]  # 随便给个数，防止程序报错
                            t1 = time.time()
                            duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                            cv2.putText(frameEach, 'kcf model: only kcf, jump, fail', (8, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(frameEach, 'edgemiss = ' +
                                        str(edgemiss_flag), (8, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.')
                                        , (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            print('kcf model: only kcf, jump, fail')
                            continue
                    # 无论yolo检测结果如何，都用kcf画框
                    t1 = time.time()
                    duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                    sub_frame = frameEach[y_kcf: y_kcf + h_kcf, x_kcf: x_kcf + w_kcf]
                    cv2.rectangle(frameEach, (x_kcf, y_kcf), (x_kcf + w_kcf, y_kcf + h_kcf),
                                  (0, 0, 255), 2)
                    cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


                # yolo跟踪模式
                else:
                    # 如果边缘miss且没出现jumptimes，每5帧执行一次yolo检测，提高fps
                    if (edgemiss_flag == True) and (yolocount % 5 != 0) and (jumptimes == 0):
                        sub_frame = frameEach[500: 700, 500: 700]
                        t1 = time.time()
                        duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                        cv2.putText(frameEach, 'yolo model: edgemiss, dont detect', (8, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frameEach, 'edgemiss = ' +
                                    str(edgemiss_flag), (8, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'),
                                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print('yolo model: edgemiss, dont detect')
                    # 如果miss但不是边缘miss，且没出现jumptimes，则每2帧检测一次
                    elif (edgemiss_flag == False) and (miss_flag == True) \
                            and (yolocount % 2 != 0) and (jumptimes == 0):
                        sub_frame = frameEach[500: 700, 500: 700]
                        t1 = time.time()
                        duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                        cv2.putText(frameEach, 'yolo model: miss, dont detect', (8, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frameEach, 'edgemiss = ' +
                                    str(edgemiss_flag), (8, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'),
                                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        print('yolo model: miss, dont detect')
                    # 如果没miss，每帧都执行yolo检测
                    else:
                        im, result = det.detect(frameEach)
                        if (len(result) > 0):  # 如果yolo能检测出东西
                            x = result[0][0]
                            y = result[0][1]
                            w = result[0][2] - result[0][0]
                            h = result[0][3] - result[0][1]
                            distance = 0
                            yololist.append([x, y, w, h])
                            dx = x - yololist[-2][0]
                            dy = y - yololist[-2][1]
                            distance = dx * dx + dy * dy
                            # # 选出一个距离上帧yolo框最近的yolo框
                            # for i in range(1, len(result)):
                            #     dx_temp = result[i][0] - yololist[-2][0]
                            #     dy_temp = result[i][1] - yololist[-2][1]
                            #     distance_temp = dx_temp * dx_temp + dy_temp * dy_temp
                            #     if distance_temp < distance:
                            #         x = result[i][0]
                            #         y = result[i][1]
                            #         w = result[i][2] - result[0][0]
                            #         h = result[i][3] - result[0][1]
                            #         distance = distance_temp
                            #         yololist[-1] = [x, y, w, h]

                            # 若edgemiss_flag=0，且yolo跳变，则yolo暂时不可靠，具体还需jump_same_flag进一步判断
                            print('dx = ', dx, ' dy = ', dy)
                            if (edgemiss_flag == False) and ((math.fabs(dx) > 50) or (math.fabs(dy) > 50)):
                                del yololist[-1]
                                misscount += 1
                                if misscount >= 3:
                                    miss_flag = True
                                sub_frame = frameEach[500: 700, 500: 700] # 随便给个数，防止程序报错
                                t1 = time.time()
                                duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                                cv2.putText(frameEach, 'yolo model: jump, fail', (8, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'edgemiss = ' +
                                            str(edgemiss_flag), (8, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'),
                                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print('yolo model: jump, fail')
                                jumptimes += 1
                                print('jumptimes = ', jumptimes)
                                if jumptimes >= 3: # 看连续2帧是否接近，若接近，认为是目标
                                    jump_same_flag = True
                                    for i in jump_delta:
                                        print('i in jump_delta = ', i)
                                        if math.fabs(i) > 50:
                                            jump_same_flag = False
                                    if jump_same_flag: # yolo结果可靠
                                        yololist.append([x, y, w, h])
                                        edgemiss_flag = False
                                        miss_flag = False
                                        misscount = 0
                                        jump_same_flag = False
                                        jumptimes = 0
                                        jump_delta = []
                                        jumplist = []
                                        KCFGroundList.append(yololist[-1])
                                        # yolo画框
                                        cv2.rectangle(frameEach, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        t1 = time.time()
                                        duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                                        # sub_frame = frameEach[y: y + h, x: x + w]
                                        sub_frame = frameEach[500: 700, 500: 700]
                                        cv2.putText(frameEach, 'yolo model: jump, success', (8, 50),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        cv2.putText(frameEach, 'edgemiss = ' +
                                                    str(edgemiss_flag), (8, 80),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        cv2.putText(frameEach,
                                                    'FPS: ' + str(1 / duration)[:4].strip('.'),
                                                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                        print('yolo model: jump, success')
                                        continue # 因为flag值改变，不执行后续语句
                                    # 每隔3帧清空一次
                                    jumptimes = 0
                                    jump_delta = []
                                    jumplist = []
                                elif (jumptimes == 1):
                                    jumplist.append(x)
                                    jumplist.append(y)
                                elif (jumptimes > 1): # 前面=3时jumptimes置0，防止再次进入
                                    jump_delta.append(jumplist[-2] - x)
                                    jump_delta.append(jumplist[-1] - y)
                                    jumplist.append(x)
                                    jumplist.append(y)
                            # edgemiss_flag=0，yolo不跳变，yolo可靠
                            elif (edgemiss_flag == False) and ((math.fabs(dx) <= 50) and (math.fabs(dy) <= 50)):
                                miss_flag = False
                                misscount = 0
                                # w>20，下帧kcf跟踪
                                if w >= 20:
                                    yolo_flag = False
                                    tracker.init(yololist[-1], im)
                                KCFGroundList.append(yololist[-1])
                                # yolo画框
                                cv2.rectangle(frameEach, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                t1 = time.time()
                                duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                                # sub_frame = frameEach[y: y + h, x: x + w]
                                sub_frame = frameEach[500: 700, 500: 700]
                                cv2.putText(frameEach, 'yolo model: success', (8, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'edgemiss = ' +
                                            str(edgemiss_flag), (8, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'),
                                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print('yolo model: success')
                            # 若edgemiss_flag=1，但yolo检测结果不在边界，则yolo不可靠
                            elif (edgemiss_flag == True) and ((x > 50) and (x + w < im.shape[1] - 50) and (y > 50)
                                                              and (y + h < im.shape[0] - 50)):
                                del yololist[-1]
                                misscount += 1
                                sub_frame = frameEach[500: 700, 500: 700]  # 随便给个数，防止程序报错
                                t1 = time.time()
                                duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                                cv2.putText(frameEach, 'yolo model: edgemiss, jump, fail', (8, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'edgemiss = ' +
                                            str(edgemiss_flag), (8, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'),
                                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print('yolo model: edgemiss, jump, fail')
                            # 若edgemiss_flag=1，且yolo检测结果在边界，则yolo可靠，该判别条件必须在带有edgemiss_flag==0
                            # 判别条件之后
                            else:
                                KCFGroundList.append(yololist[-1])
                                # w>20，下帧kcf跟踪
                                if w >= 20:
                                    yolo_flag = False
                                    tracker.init(yololist[-1], im)
                                # yolo画框
                                cv2.rectangle(frameEach, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                t1 = time.time()
                                duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                                # sub_frame = frameEach[y: y + h, x: x + w]
                                sub_frame = frameEach[500: 700, 500: 700]
                                cv2.putText(frameEach, 'yolo model: edgemiss, success', (8, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'edgemiss = ' +
                                            str(edgemiss_flag), (8, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'),
                                            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                print('yolo model: edgemiss, success')
                                miss_flag = False
                                edgemiss_flag = False
                                misscount = 0
                        else: # yolo检测不到东西
                            # 判定是否边缘miss，必须先是miss，才能再是边缘miss
                            if (miss_flag == True) and ((yololist[-1][0] < 30) or (yololist[-1][0] + yololist[-1][2] >
                                    im.shape[1] - 30) or (yololist[-1][1] < 30) or (yololist[-1][1] + yololist[-1][3] >
                                                                                      im.shape[0] - 30)):
                                edgemiss_flag = True
                            misscount += 1
                            if misscount >= 3:
                                miss_flag = True
                            sub_frame = frameEach[500: 700, 500: 700]
                            t1 = time.time()
                            duration = 0.0001 * duration + 0.9999 * (t1 - t0)
                            cv2.putText(frameEach, 'yolo model: nothing, fail', (8, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(frameEach, 'edgemiss = ' +
                                        str(edgemiss_flag), (8, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'),
                                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            print('yolo model: nothing, fail')



            TrackingFrame_Pathout = ("C:/dataset/KCF_Test_results/Outputframes_temp/frame_00000" + str(framenum) + ".jpg")
            FirstFrame_Pathout = ("C:/dataset/KCF_Test_results/Outputframes_temp/sub_frame_00000" + str(framenum) + ".jpg")

            framenum += 1
            cv2.imwrite(TrackingFrame_Pathout, frameEach)
            cv2.imwrite(FirstFrame_Pathout, sub_frame)

            Img2 = Image.open(TrackingFrame_Pathout)
            Img2 = Img2.resize((600, 400), Image.ANTIALIAS)
            Img2 = ImageTk.PhotoImage(Img2)
            Img3 = Image.open(FirstFrame_Pathout)
            Img3 = Img3.resize((300, 200), Image.ANTIALIAS)
            Img3 = ImageTk.PhotoImage(Img3)

            Label_FirstFrameShow.config(image=Img3)
            Label_FirstFrameShow.image = Img3
            Label_FirstFrameShow.update()
            Label_ObjectTrackingShow.config(image=Img2)
            Label_ObjectTrackingShow.image = Img2
            Label_ObjectTrackingShow.update()

            Text_TrackingLog.see(tkinter.END)
            c = cv2.waitKey(27) & 0xFF
            if c == 27 or c == ord('q'):
                break

        file_write_obj = open("./framespositionstore/KCF_withAdaptiveFrames.txt", 'w')
        for i in range(len(KCFGroundList)):
            file_write_obj.writelines(str(KCFGroundList[i]))
            file_write_obj.write('\n')
        file_write_obj.close()

        Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
        Text_TrackingLog.update()
        Text_TrackingLog.config(state='disabled')
        cap.release()
    elif (Algorithm_name == "SiamRPN" and Test_RecGroundTruth_Name != ""):
        TrackingBoundingBoxPath = "./framespositionstore/KCF_withAdaptiveFrames.txt"
        selectingObject = False
        initTracking = False
        onTracking = False
        ix, iy, cx, cy = -1, -1, -1, -1
        w, h = 0, 0
        duration = 0.01
        Performancelistx = []
        Performancelisty = []
        KCFGroundList = []
        RecGroundList = []
        IOUlistx = []
        IOUlisty = []
        dogmp4IOUfilename1 = Test_RecGroundTruth_Name
        file = open(dogmp4IOUfilename1, "r")
        line = file.readline()
        while (line):
            # print(line)
            line = file.readline()
            line = line.rstrip("\n")
            RecGroundList.append(line)
        file.close()

        def draw_boundingbox(event, x, y, flags, param):
            global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
            if event == cv2.EVENT_LBUTTONDOWN:
                selectingObject = True
                onTracking = False
                ix, iy = x, y
                cx, cy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                cx, cy = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                selectingObject = False
                if (abs(x - ix) > 10 and abs(y - iy) > 10):
                    w, h = abs(x - ix), abs(y - iy)
                    ix, iy = min(x, ix), min(y, iy)
                    initTracking = True
                else:
                    onTracking = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                onTracking = False
                if (w > 0):
                    ix, iy = x - w / 2, y - h / 2
                    initTracking = True

        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        the_first_frame_flag, the_first_frame = cap.read()
        cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
        pathIn = 'get_the_first_frame.jpg'
        pathOut = './Outputframes/frame_000001.jpg'
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        tracker = kcftracker.KCFTracker(True, True, True)
        ix = Box[0][0]
        iy = Box[0][1]
        cx = Box[1][0]
        cy = Box[1][1]
        Begin_train_flag = True
        framenum = 0
        print(Box)
        print(framenum)
        Text_TrackingLog.insert(tkinter.END, '***********执行跟踪程序*************\n')
        Missflag = False
        count = 120
        while (cap.isOpened()):
            sub_frame = []
            Text_TrackingLog.config(state='normal')
            if (count == 0):
                TrackingStr2 = ("**************自动重检**************\n")
                test = code.yolo_detect(pathIn, pathOut)
                Box = [(test[2], test[3])]
                Box.append(((test[2] + test[4]), test[3] + test[5]))
                tracker = kcftracker.KCFTracker(True, True, True)
                ix = Box[0][0]
                iy = Box[0][1]
                cx = Box[1][0]
                cy = Box[1][1]
                Begin_train_flag = True
                Text_TrackingLog.insert(tkinter.END, TrackingStr2)
                Text_TrackingLog.update()
                count = 120
            Text_TrackingLog.update()
            ret, frameEach = cap.read()
            if not ret:
                break
            if (Begin_train_flag):
                framenum += 1
                if w == 0 or h == 0:
                    sub_frame = frameEach
                else:
                    sub_frame = frameEach[ix:ix + w, iy:iy + h]
                cv2.rectangle(frameEach, (ix, iy), (ix + w, iy + h), (0, 0, 255), 2)
                tracker.init([ix, iy, cx - ix, cy - iy], frameEach)
                Begin_train_flag = False
                onTracking = True
                KCFGroundList.append([ix, iy, cx - ix, cy - iy])
            elif (onTracking):
                framenum += 1
                t0 = time.time()
                boundingbox = tracker.update(frameEach)
                t1 = time.time()
                boundingbox = list(map(int, boundingbox))
                print(boundingbox)
                if boundingbox[2] != 0 and boundingbox[3] != 0:
                    sub_frame = frameEach[boundingbox[1]: boundingbox[1] + boundingbox[3],
                                boundingbox[0]: boundingbox[0] + boundingbox[2]]
                cv2.rectangle(frameEach, (boundingbox[0], boundingbox[1]),
                              (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 0, 255), 2)
                duration = 0.8 * duration + 0.2 * (t1 - t0)
                duration = t1 - t0

                cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255), 2)
            TrackingFrame_Pathout = ("./Outputframes/frame_00000" + str(framenum) + ".jpg")
            FirstFrame_Pathout = ("./Outputframes/sub_frame_00000" + str(framenum) + ".jpg")

            cv2.imwrite(TrackingFrame_Pathout, frameEach)
            cv2.imwrite(FirstFrame_Pathout, sub_frame)

            Img2 = Image.open(TrackingFrame_Pathout)
            Img2 = Img2.resize((600, 400), Image.ANTIALIAS)
            Img2 = ImageTk.PhotoImage(Img2)
            Img3 = Image.open(FirstFrame_Pathout)
            Img3 = Img3.resize((300, 200), Image.ANTIALIAS)
            Img3 = ImageTk.PhotoImage(Img3)

            Label_FirstFrameShow.config(image=Img3)
            Label_FirstFrameShow.image = Img3
            Label_FirstFrameShow.update()
            Label_ObjectTrackingShow.config(image=Img2)
            Label_ObjectTrackingShow.image = Img2
            Label_ObjectTrackingShow.update()

            Text_TrackingLog.see(tkinter.END)
            c = cv2.waitKey(27) & 0xFF
            if c == 27 or c == ord('q'):
                break

        '''
        将跟踪框位置写入文件
       '''
        file_write_obj = open("./framespositionstore/KCF_withAdaptiveFrames.txt", 'w')
        for i in range(len(KCFGroundList)):
            file_write_obj.writelines(str(KCFGroundList[i]))
            file_write_obj.write('\n')
        file_write_obj.close()

        Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
        Text_TrackingLog.update()
        Text_TrackingLog.config(state='disabled')
        cap.release()
    elif (Algorithm_name == "SiamRPN" and Test_RecGroundTruth_Name == ""):
        TrackingBoundingBoxPath = "./framespositionstore/KCF_withAdaptiveFrames.txt"
        selectingObject = False
        initTracking = False
        onTracking = False
        ix, iy, cx, cy = -1, -1, -1, -1
        w, h = 0, 0
        duration = 0.01
        Performancelistx = []
        Performancelisty = []
        KCFGroundList = []

        def draw_boundingbox(event, x, y, flags, param):
            global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
            if event == cv2.EVENT_LBUTTONDOWN:
                selectingObject = True
                onTracking = False
                ix, iy = x, y
                cx, cy = x, y
            elif event == cv2.EVENT_MOUSEMOVE:
                cx, cy = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                selectingObject = False
                if (abs(x - ix) > 10 and abs(y - iy) > 10):
                    w, h = abs(x - ix), abs(y - iy)
                    ix, iy = min(x, ix), min(y, iy)
                    initTracking = True
                else:
                    onTracking = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                onTracking = False
                if (w > 0):
                    ix, iy = x - w / 2, y - h / 2
                    initTracking = True

        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        tracker = kcftracker.KCFTracker(True, True, True)
        ix = Box[0][0]
        iy = Box[0][1]
        cx = Box[1][0]
        cy = Box[1][1]
        Begin_train_flag = True
        framenum = 1
        Text_TrackingLog.insert(tkinter.END, '***********执行跟踪程序*************\n')
        Missflag = False
        count = 120
        while (cap.isOpened()):
            Text_TrackingLog.config(state='normal')
            Text_TrackingLog.update()
            ret, frameEach = cap.read()
            sub_frame = []
            if not ret:
                break
            if (Begin_train_flag):
                if w == 0 or h == 0:
                    sub_frame = frameEach
                else:
                    sub_frame = frameEach[ix:ix + w, iy:iy + h]
                cv2.rectangle(frameEach, (ix, iy), (ix + w, iy + h), (0, 0, 255), 2)
                tracker.init([ix, iy, cx - ix, cy - iy], frameEach)
                Begin_train_flag = False
                onTracking = True
            elif (onTracking):
                t0 = time.time()
                boundingbox = tracker.update(frameEach)
                t1 = time.time()
                boundingbox = list(map(int, boundingbox))
                print(boundingbox)
                if boundingbox[2] != 0 and boundingbox[3] != 0:
                    sub_frame = frameEach[boundingbox[1]: boundingbox[1] + boundingbox[3],
                                boundingbox[0]: boundingbox[0] + boundingbox[2]]
                cv2.rectangle(frameEach, (boundingbox[0], boundingbox[1]),
                              (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 0, 255), 2)
                duration = 0.8 * duration + 0.2 * (t1 - t0)
                # duration = t1 - t0
                KCFGroundList.append(boundingbox)

                cv2.putText(frameEach, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255), 2)
            TrackingFrame_Pathout = ("./Outputframes/frame_00000" + str(framenum) + ".jpg")
            FirstFrame_Pathout = ("./Outputframes/sub_frame_00000" + str(framenum) + ".jpg")

            framenum += 1
            cv2.imwrite(TrackingFrame_Pathout, frameEach)
            cv2.imwrite(FirstFrame_Pathout, sub_frame)
            Img2 = Image.open(TrackingFrame_Pathout)
            Img2 = Img2.resize((600, 400), Image.ANTIALIAS)
            Img2 = ImageTk.PhotoImage(Img2)
            Img3 = Image.open(FirstFrame_Pathout)
            Img3 = Img3.resize((300, 200), Image.ANTIALIAS)
            Img3 = ImageTk.PhotoImage(Img3)

            Label_FirstFrameShow.config(image=Img3)
            Label_FirstFrameShow.image = Img3
            Label_FirstFrameShow.update()
            Label_ObjectTrackingShow.config(image=Img2)
            Label_ObjectTrackingShow.image = Img2
            Label_ObjectTrackingShow.update()

            Text_TrackingLog.see(tkinter.END)
            c = cv2.waitKey(27) & 0xFF
            if c == 27 or c == ord('q'):
                break


        file_write_obj = open("./framespositionstore/KCF_withAdaptiveFrames.txt", 'w')
        for i in range(len(KCFGroundList)):
            file_write_obj.writelines(str(KCFGroundList[i]))
            file_write_obj.write('\n')
        file_write_obj.close()

        Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
        Text_TrackingLog.update()
        Text_TrackingLog.config(state='disabled')
        cap.release()

    elif (Algorithm_name == "KCF" and Test_RecGroundTruth_Name == ""):
        Performancelistx = []
        Performancelisty = []
        RPNGroundlist = []
        RecGroundList = []
        IOUlistx = []
        IOUlisty = []
        KCFGroundList = []
        TrackingBoundingBoxPath = "./framespositionstore/KCF_Frames.txt"
        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        the_first_frame_flag, the_first_frame = cap.read()
        cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
        pathIn = 'get_the_first_frame.jpg'
        pathOut = './Outputframes/frame_000001.jpg'
        test = code.yolo_detect(pathIn, pathOut)
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        frame_count = 1
        frame_flag = True
        train_flag = True
        detect_flag = True
        tracker = KCFTracker(feature_type='raw', sub_feature_type='gray')
        count = 20
        framenum = 1
        while (frame_flag):
            framenum += 1
            frame_flag, frameEach = cap.read()
            sub_frame = []
            if (frame_flag):
                frame_clone = frameEach.copy()
                if train_flag:
                    start_time = time.time()
                    TrackerBox = [Box[0][0], Box[0][1], Box[1][0] - Box[0][0], Box[1][1] - Box[0][1]]
                    if TrackerBox[2] == 0 and TrackerBox[3] == 0:
                        sub_frame = frameEach
                    else:
                        sub_frame = frameEach[Box[0][0]:Box[1][0], Box[0][1]:Box[1][1]]
                    start_time = time.time()
                    tracker.train(frameEach, TrackerBox)
                    total_time = time.time() - start_time
                    train_flag = False
                else:
                    start_time = time.time()
                    res = tracker.detect(frame_clone)
                    total_time = time.time() - start_time


                    if res[2] != 0 and res[3] != 0:
                        sub_frame = frameEach[int(res[1]):int(res[1] + res[3]), int(res[0]):int(res[0] + res[2])]
                    cv2.rectangle(frameEach, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])),
                                  (0, 0, 205), 2)
                    KCFGroundList.append(res)
                    # cv2.imshow("frame", frameEach)
                    count = count - 1
                    TrackingFrame_Pathout = ("./Outputframes/frame_00000" + str(framenum) + ".jpg")
                    FirstFrame_Pathout = ("./Outputframes/sub_frame_00000" + str(framenum) + ".jpg")

                    framenum += 1
                    cv2.imwrite(TrackingFrame_Pathout, frameEach)
                    cv2.imwrite(FirstFrame_Pathout, sub_frame)

                    Img2 = Image.open(TrackingFrame_Pathout)
                    Img2 = Img2.resize((600, 400), Image.ANTIALIAS)
                    Img2 = ImageTk.PhotoImage(Img2)
                    Img3 = Image.open(FirstFrame_Pathout)
                    Img3 = Img3.resize((300, 200), Image.ANTIALIAS)
                    Img3 = ImageTk.PhotoImage(Img3)

                    Label_FirstFrameShow.config(image=Img3)
                    Label_FirstFrameShow.image = Img3
                    Label_FirstFrameShow.update()
                    Label_ObjectTrackingShow.config(image=Img2)
                    Label_ObjectTrackingShow.image = Img2
                    Label_ObjectTrackingShow.update()

            else:
                break

        file_write_obj = open("./framespositionstore/KCF_Frames.txt", 'w')
        for i in range(len(KCFGroundList)):
            file_write_obj.writelines(str(KCFGroundList[i]))
            file_write_obj.write('\n')
        file_write_obj.close()
        Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
        Text_TrackingLog.update()
        Text_TrackingLog.config(state='disabled')
        cap.release()
    elif (Algorithm_name == "KCF" and Test_RecGroundTruth_Name != ""):
        Performancelistx = []
        Performancelisty = []
        RecGroundList = []
        IOUlistx = []
        IOUlisty = []
        KCFGroundList = []
        dogmp4IOUfilename1 = Test_RecGroundTruth_Name
        file = open(dogmp4IOUfilename1, "r")
        line = file.readline()
        while (line):
            # print(line)
            line = file.readline()
            line = line.rstrip("\n")
            RecGroundList.append(line)
        file.close()
        TrackingBoundingBoxPath = "./framespositionstore/KCF_Frames.txt"
        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        the_first_frame_flag, the_first_frame = cap.read()
        cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
        pathIn = 'get_the_first_frame.jpg'
        pathOut = './Outputframes/frame_000001.jpg'
        test = code.yolo_detect(pathIn, pathOut)
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        frame_count = 1
        frame_flag = True
        train_flag = True
        detect_flag = True
        tracker = KCFTracker(feature_type='raw', sub_feature_type='gray')
        count = 20
        framenum = 1
        while (frame_flag):
            frame_flag, frameEach = cap.read()
            sub_frame = []
            if (frame_flag):
                frame_clone = frameEach.copy()
                if train_flag:
                    start_time = time.time()
                    TrackerBox = [Box[0][0], Box[0][1], Box[1][0] - Box[0][0], Box[1][1] - Box[0][1]]
                    if TrackerBox[2] == 0 and TrackerBox[3] == 0:
                        sub_frame = frameEach
                    else:
                        sub_frame = frameEach[Box[0][0]:Box[1][0], Box[0][1]:Box[1][1]]
                    start_time = time.time()
                    tracker.train(frameEach, TrackerBox)
                    total_time = time.time() - start_time
                    train_flag = False
                else:
                    start_time = time.time()
                    res = tracker.detect(frame_clone)

                    total_time = time.time() - start_time
                    if res[2] != 0 and res[3] != 0:
                        sub_frame = frameEach[int(res[1]):int(res[1] + res[3]), int(res[0]):int(res[0] + res[2])]
                    cv2.rectangle(frameEach, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])),
                                  (0, 0, 205), 2)
                    KCFGroundList.append(res)
                    count = count - 1
                    TrackingFrame_Pathout = ("./Outputframes/frame_00000" + str(framenum) + ".jpg")
                    FirstFrame_Pathout = ("./Outputframes/sub_frame_00000" + str(framenum) + ".jpg")

                    framenum += 1
                    cv2.imwrite(TrackingFrame_Pathout, frameEach)
                    cv2.imwrite(FirstFrame_Pathout, sub_frame)

                    Img2 = Image.open(TrackingFrame_Pathout)
                    Img2 = Img2.resize((600, 400), Image.ANTIALIAS)
                    Img2 = ImageTk.PhotoImage(Img2)
                    Img3 = Image.open(FirstFrame_Pathout)
                    Img3 = Img3.resize((300, 200), Image.ANTIALIAS)
                    Img3 = ImageTk.PhotoImage(Img3)

                    Label_FirstFrameShow.config(image=Img3)
                    Label_FirstFrameShow.image = Img3
                    Label_FirstFrameShow.update()
                    Label_ObjectTrackingShow.config(image=Img2)
                    Label_ObjectTrackingShow.image = Img2
                    Label_ObjectTrackingShow.update()

            else:
                break

        file_write_obj = open("./framespositionstore/KCF_Frames.txt", 'w')
        for i in range(len(KCFGroundList)):
            file_write_obj.writelines(str(KCFGroundList[i]))
            file_write_obj.write('\n')
        file_write_obj.close()
        Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
        Text_TrackingLog.update()
        Text_TrackingLog.config(state='disabled')
        cap.release()


def TestAlgorithmChoose():
    global Algorithm_name
    Windows_Reply = tkinter.Tk()
    Windows_Reply.title("选择测试算法")
    Windows_Reply.geometry("500x300")
    '''
    响应界面按钮布局设计
    '''

    # 选择视频按钮，次级响应，弹出确定提示，销毁无用窗口
    def ButtonAlgorithmReplyThree():
        flag = tkinter.messagebox.askokcancel("Attention", "确定选择此算法？")
        print(flag)
        if flag:
            global Algorithm_name
            Algorithm_name = "KCF"
            Text_TrackingLog.config(state='normal')
            Text_TrackingLog.delete('1.0', 'end')
            Text_TrackingLog.insert(tkinter.END, "算法" + Algorithm_name + "已经载入\n")
            Text_TrackingLog.config(state='disabled')
            Windows_Reply.destroy()

    def ButtonAlgorithmRePlyTwo():
        flag = tkinter.messagebox.askokcancel("Attention", "确定选择此算法？")
        print(flag)
        if flag:
            global Algorithm_name
            Algorithm_name = "SiamRPN"
            Text_TrackingLog.config(state='normal')
            Text_TrackingLog.delete('1.0', 'end')
            Text_TrackingLog.insert(tkinter.END, "算法" + Algorithm_name + "已经载入\n")
            Text_TrackingLog.config(state='disabled')
            Windows_Reply.destroy()

    def ButtonAlgorithmRePlyOne():
        flag = tkinter.messagebox.askokcancel("Attention", "确定选择此算法？")
        print(flag)
        if flag:
            global Algorithm_name
            Algorithm_name = "KCF with adaptive frames"
            Text_TrackingLog.config(state='normal')
            Text_TrackingLog.delete('1.0', 'end')
            Text_TrackingLog.insert(tkinter.END, "算法" + Algorithm_name + "已经载入\n")
            Text_TrackingLog.config(state='disabled')
            Windows_Reply.destroy()

    LabelAlgorithmReply1 = tkinter.Label(Windows_Reply, text="KCF with adaptive frames", width=42, heigh=2,
                                         background="pink")
    LabelAlgorithmReply1.place(x=50, y=10)
    LabelAlgorithmReply2 = tkinter.Label(Windows_Reply, text="SiamRPN", width=42, heigh=2, background=color_back)
    LabelAlgorithmReply2.place(x=50, y=70)
    LabelAlgorithmReply3 = tkinter.Label(Windows_Reply, text="KCF", width=42, heigh=2,
                                         background="green")
    LabelAlgorithmReply3.place(x=50, y=130)

    ButtonAlgorithmChooseOne = tkinter.Button(Windows_Reply, text="选择该算法", width=10, height=2,
                                              command=ButtonAlgorithmRePlyOne)
    ButtonAlgorithmChooseOne.place(x=350, y=10)
    ButtonAlgorithmChooseTwo = tkinter.Button(Windows_Reply, text="选择该算法", width=10, height=2,
                                              command=ButtonAlgorithmRePlyTwo)
    ButtonAlgorithmChooseTwo.place(x=350, y=70)
    ButtonAlgorithmChooseThree = tkinter.Button(Windows_Reply, text="选择该算法", width=10, height=2,
                                                command=ButtonAlgorithmReplyThree)
    ButtonAlgorithmChooseThree.place(x=350, y=130)


def TestVideoChoose():
    Windows_Reply = tkinter.Tk()
    Windows_Reply.title("选择测试数据")
    Windows_Reply.geometry("500x300")
    '''
    响应界面按钮布局设计
    '''

    # 选择视频按钮，次级响应，弹出确定提示，销毁无用窗口
    def ButtonVideoRePlyThree():
        flag = tkinter.messagebox.askokcancel("Attention", "确定选择此视频？")
        print(flag)
        if flag:
            global Test_Video_Name
            Test_Video_Name = "./testVideo/1.avi"
            Text_TrackingLog.config(state='normal')
            Text_TrackingLog.insert(tkinter.END, "视频" + Test_Video_Name + "已经载入\n")
            Text_TrackingLog.config(state='disabled')
            Windows_Reply.destroy()

    def ButtonVideoRePlyTwo():
        flag = tkinter.messagebox.askokcancel("Attention", "确定选择此视频？")
        print(flag)
        if flag:
            global Test_Video_Name
            Test_Video_Name = r"C:\dataset\HIT_code_video\video_figure\video\scene11.mp4"
            Text_TrackingLog.config(state='normal')
            Text_TrackingLog.insert(tkinter.END, "视频" + Test_Video_Name + "已经载入\n")
            Text_TrackingLog.config(state='disabled')
            Windows_Reply.destroy()

    def ButtonVideoRePlyOne():
        flag = tkinter.messagebox.askokcancel("Attention", "确定选择此视频？")
        print(flag)
        if flag:
            global Test_RecGroundTruth_Name
            global Test_Video_Name
            Test_RecGroundTruth_Name = "./sourcece/OTBdogmp4_groundtruth_Rectdest.txt"
            Test_Video_Name = "./testVideo/dog.avi"
            Text_TrackingLog.config(state='normal')
            Text_TrackingLog.insert(tkinter.END, "视频" + Test_Video_Name + "已经载入\n")
            Text_TrackingLog.config(state='disabled')
            Windows_Reply.destroy()

    LabelVideoReply1 = tkinter.Label(Windows_Reply, text="Video1:OTB100-DOG(带有OTBIOU标准测评)", width=42, heigh=2,
                                     background="blue")
    LabelVideoReply1.place(x=50, y=10)
    LabelVideoReply2 = tkinter.Label(Windows_Reply, text="Video2:Aeroplane_of_desert(无IOU测评标准)", width=42, heigh=2,
                                     background="gray")
    LabelVideoReply2.place(x=50, y=70)
    LabelVideoReply3 = tkinter.Label(Windows_Reply, text="Video3:Aeroplane_of_sky(无IOU测评标准)", width=42, heigh=2,
                                     background="red")
    LabelVideoReply3.place(x=50, y=130)

    '''
    新的按钮功能：
    选择测试数据
    '''
    ButtonVideoChooseOne = tkinter.Button(Windows_Reply, text="选择该视频", width=10, height=2, command=ButtonVideoRePlyOne)
    ButtonVideoChooseOne.place(x=350, y=10)
    ButtonVideoChooseTwo = tkinter.Button(Windows_Reply, text="选择该视频", width=10, height=2, command=ButtonVideoRePlyTwo)
    ButtonVideoChooseTwo.place(x=350, y=70)
    ButtonVideoChooseThree = tkinter.Button(Windows_Reply, text="选择该视频", width=10, height=2,
                                            command=ButtonVideoRePlyThree)
    ButtonVideoChooseThree.place(x=350, y=130)


'''
保存跟踪视频结果函数
'''


def ExeTrackingVideoSave():
    fourcc = cv2.VideoWriter_fourcc("I", "4", "2", "0")
    video_writer = cv2.VideoWriter('./result.avi', fourcc, fps, (1920, 1080))
    Dir_length = (
        len([name for name in os.listdir("./Outputframes") if os.path.isfile(os.path.join("./Outputframes", name))]))
    print(Dir_length)
    for i in range(Dir_length - 1):
        i = i + 1
        Pathshit = ("Outputframes/frame_00000" + str(i) + ".jpg")
        img = cv2.imread("Outputframes/frame_00000" + str(i) + ".jpg")
        try:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            video_writer.write(img)
        except:
            print("nope")
            continue
    video_writer.release()
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.insert(tkinter.END, "**********视频生成完毕*********\n")
    Text_TrackingLog.update()

    root = tkinter.Tk()
    root.withdraw()
    file_path = tkinter.filedialog.asksaveasfilename(filetypes=[("AVI", ".avi")])
    print(file_path)
    shutil.copyfile("./result.avi", file_path + ".avi")
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.insert(tkinter.END, "******视频成功保存至：" + file_path + "******\n")
    Text_TrackingLog.update()
    Text_TrackingLog.config(state='disabled')


'''
保存跟踪框记录文件
'''


def ExeFramesSave():
    global TrackingBoundingBoxPath
    root = tkinter.Tk()
    root.withdraw()
    file_path = tkinter.filedialog.asksaveasfilename(filetypes=[("TXT", ".txt")])
    print(file_path)
    shutil.copyfile(TrackingBoundingBoxPath, file_path + ".txt")
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.insert(tkinter.END, "******跟踪框记录文件成功保存至：" + file_path + "******\n")
    Text_TrackingLog.update()
    Text_TrackingLog.config(state='disabled')


'''
保存性能图文件
'''


def ExeFPSPerformanceSave():
    root = tkinter.Tk()
    root.withdraw()
    file_path = tkinter.filedialog.asksaveasfilename(filetypes=[("JPG", "jpg")])
    print(file_path)
    shutil.copyfile("./FinalPerformance/fpsfinalperformance.jpg", file_path + "ffpf.jpg")
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.insert(tkinter.END, "******FPS性能图文件成功保存至：" + file_path + "******\n")
    Text_TrackingLog.update()
    Text_TrackingLog.config(state='disabled')


def ExeIOUPerformanceSave():
    root = tkinter.Tk()
    root.withdraw()
    file_path = tkinter.filedialog.asksaveasfilename(filetypes=[("JPG", "jpg")])
    print(file_path)
    shutil.copyfile("./FinalPerformance/IOUfinalperformance.jpg", file_path + "ifpf.jpg")
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.insert(tkinter.END, "******IOU性能图文件成功保存至：" + file_path + "******\n")
    Text_TrackingLog.update()
    Text_TrackingLog.config(state='disabled')


'''
系统复位功能
'''


def ExeReset():
    global Algorithm_name
    global Test_RecGroundTruth_Name
    global Test_Video_Name
    Algorithm_name = ""
    Test_Video_Name = ""
    Test_RecGroundTruth_Name = ""
    TrackingBoundingBoxPath = ""
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.delete('1.0', 'end')
    Text_TrackingLog.insert(1.0, "无信号输入\n")
    Text_TrackingLog.config(state='disabled')

    Text_DetectLog.config(state='normal')
    Text_DetectLog.delete('1.0', 'end')
    Text_DetectLog.insert(1.0, "无信号输入\n")
    Text_DetectLog.config(state='disabled')

    Img_Background = Image.open("./sourcece/black.jpg")
    Img_Background = Img_Background.resize((480, 320), Image.ANTIALIAS)
    Img = ImageTk.PhotoImage(Img_Background)
    Label_FirstFrameShow.config(image=Img)
    Label_FirstFrameShow.image = Img
    Label_FirstFrameShow.update()

    Img_Background2 = Image.open("./sourcece/black.jpg")
    Img_Background2 = Img_Background2.resize((600, 400), Image.ANTIALIAS)
    Img2 = ImageTk.PhotoImage(Img_Background2)
    Label_ObjectTrackingShow.config(image=Img2)
    Label_ObjectTrackingShow.image = Img2
    Label_ObjectTrackingShow.update()

    Img_Background3 = Image.open("./sourcece/black.jpg")
    Img_Background3 = Img_Background3.resize((390, 230), Image.ANTIALIAS)
    Img3 = ImageTk.PhotoImage(Img_Background3)

    Img_Background4 = Image.open("./sourcece/black.jpg")
    Img_Background4 = Img_Background4.resize((390, 230), Image.ANTIALIAS)
    Img4 = ImageTk.PhotoImage(Img_Background4)

    Outputframe_Path = "./Outputframes"
    FinalPerformance_Path = "./FinalPerformance"
    fpsPerformanceframes_Path = "./fpsPerformanceframes"
    IOUPerformanceframes_Path = "./IOUPerformanceframes"
    framespositionstore_Path = "./framespositionstore"

    def del_file(path):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):
                del_file(c_path)
            else:
                os.remove(c_path)

    del_file(Outputframe_Path)
    del_file(FinalPerformance_Path)
    del_file(fpsPerformanceframes_Path)
    del_file(IOUPerformanceframes_Path)
    del_file(framespositionstore_Path)


'''
全局变量
'''
color_label = "DarkGrey"
color_back = "LightGrey"
video_back = "black"

'''
最右侧按钮群布局
'''
tkfontset1 = tkfont.Font(size=14, weight=tkfont.BOLD)

Label_ButtonGroupTitle = tkinter.Label(Windows, text="检测跟踪", width=16, height=2, background=color_label,
                                       font=tkfontset1)
Label_ButtonGroupTitle.place(x=1118, y=5)
Label_ButtonGroupTitle = tkinter.Label(Windows, text="云台控制", width=16, height=2, background=color_label,
                                       font=tkfontset1)
Label_ButtonGroupTitle.place(x=1118, y=261)
Label_ButtonGroupTitle = tkinter.Label(Windows, text="系统控制", width=16, height=2, background=color_label,
                                       font=tkfontset1)
Label_ButtonGroupTitle.place(x=1118, y=517)

Label_ButtonGroup = tkinter.Label(Windows, width=25, height=12, background=color_back)
Label_ButtonGroup.place(x=1118, y=50)
Label_ButtonGroup2 = tkinter.Label(Windows, width=25, height=12, background=color_back)
Label_ButtonGroup2.place(x=1118, y=306)
Label_ButtonGroup2 = tkinter.Label(Windows, width=25, height=5, background=color_back)
Label_ButtonGroup2.place(x=1118, y=562)

# 测试视频选择按钮
Button_VedSelect = tkinter.Button(Windows, text='选择测试视频', width=20, height=2, command=TestVideoChoose)  # 实例化一个Button
Button_VedSelect.place(x=1133, y=57)
# 跟踪算法选择按钮
Button_AlgSelect = tkinter.Button(Windows, text='选择跟踪算法', width=20, height=2,
                                  command=TestAlgorithmChoose)  # 实例化一个Button
Button_AlgSelect.place(x=1133, y=107)
# 执行检测按钮
Button_ExeDetect = tkinter.Button(Windows, text='执行初帧检测', width=20, height=2, command=ExeDetection)  # 实例化一个Button
Button_ExeDetect.place(x=1133, y=157)
# 执行跟踪按钮
Button_ExeTracking = tkinter.Button(Windows, text='执行跟踪', width=20, height=2, command=ExeTracking)  # 实例化一个Button
Button_ExeTracking.place(x=1133, y=207)

# 工作模式按钮
Button_VedSelect = tkinter.Button(Windows, text='工作模式', width=20, height=2, command=TestVideoChoose)  # 实例化一个Button
Button_VedSelect.place(x=1133, y=313)
# 急停按钮
Button_AlgSelect = tkinter.Button(Windows, text='急停', width=20, height=2, command=TestAlgorithmChoose)  # 实例化一个Button
Button_AlgSelect.place(x=1133, y=363)
# 复位按钮
Button_ExeDetect = tkinter.Button(Windows, text='复位', width=20, height=2, command=ExeDetection)  # 实例化一个Button
Button_ExeDetect.place(x=1133, y=413)
# 参数调整按钮
Button_ExeTracking = tkinter.Button(Windows, text='参数调整', width=20, height=2, command=ExeTracking)  # 实例化一个Button
Button_ExeTracking.place(x=1133, y=463)

# 结果查看按钮
Button_VedSelect = tkinter.Button(Windows, text='结果查看', width=10, height=4, command=TestVideoChoose)  # 实例化一个Button
Button_VedSelect.place(x=1128, y=564)
# 复位按钮
Button_ExeReset = tkinter.Button(Windows, text='系统复位', width=10, height=4, command=ExeReset)  # 实例化一个Button
Button_ExeReset.place(x=1208, y=564)

'''
输出检测结果功能块布局
'''
Label_DetectTitle = tkinter.Label(Windows, text="检测结果", width=40, height=2, background=color_label, font=tkfontset1)
Label_DetectTitle.place(x=670, y=5)
# 设置初始背景图，保障Label合理
Img_Background = Image.open("./sourcece/black.jpg")
Img_Background = Img_Background.resize((442, 350), Image.ANTIALIAS)
Img = ImageTk.PhotoImage(Img_Background)
Label_FirstFrameShow = tkinter.Label(Windows, width=442, height=350, background=video_back, image=Img)
Label_FirstFrameShow.place(x=670, y=50)

Text_DetectLog = tkinter.Text(Windows, font=("Arial", 12), width=37, height=1, background="white")

'''
输出跟踪结果功能块布局
'''
Img_Background2 = Image.open("./sourcece/black.jpg")
Img_Background2 = Img_Background2.resize((662, 593), Image.ANTIALIAS)
Img2 = ImageTk.PhotoImage(Img_Background2)
Label_TrackingTitle = tkinter.Label(Windows, text="跟踪过程", width=60, height=2, background=color_label, font=tkfontset1)
Label_TrackingTitle.place(x=2, y=5)
Label_ObjectTrackingShow = tkinter.Label(Windows, width=662, height=593, background=video_back, image=Img2)
Label_ObjectTrackingShow.place(x=2, y=50)
'''
输出跟踪性能图功能块布局
'''

'''
系统日志输出布局
'''
Label_Logtitle = tkinter.Label(Windows, text="系统日志", width=40, height=1, background=color_label, font=tkfontset1)
Label_Logtitle.place(x=670, y=405)
Text_TrackingLog = tkinter.scrolledtext.ScrolledText(Windows, width=61, height=16, background="white")
Text_TrackingLog.insert(1.0, "无信号输入")
Text_TrackingLog.config(state='disabled')
Text_TrackingLog.place(x=670, y=432)
'''
使用提示Label
'''
Label_UsingTips = tkinter.Label(Windows, text="测试下一个视频前请进行系统复位，清除缓存，同时注意历史保存，必须选定算法和测试视频！", width=77, height=2,
                                background="red")
Label_UsingTips.place(x=830, y=700)
Windows.mainloop()
