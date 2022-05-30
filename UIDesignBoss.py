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

det = Detector()
test = []
TrackingBoundingBoxPath = ""
fps = 20

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


# InputVideoPath=r"B:\Document\Code\586\KCFTest\testVideo\areaplane_of_sky.mp4"
# InputVideoPath=r"B:\Document\Code\586\KCFTest\testVideo\aeroplane_of_desert.mp4"

def ExeDetection():
    global test
    # det = Detector()
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.insert(tkinter.END, "*********开始执行检测识别**********\n")
    Text_TrackingLog.update()
    global Img
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(Test_Video_Name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    the_first_frame_flag, the_first_frame = cap.read()
    # cv2.imshow('b',the_first_frame)
    # cv2.waitKey(1000)
    cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
    pathIn = './get_the_first_frame.jpg'
    # pathOut = './Outputframes/frame_000001.jpg'
    # ret, frameEach = cap.read()
    # gROI = cv2.selectROI("ROI frame", the_first_frame, False)
    # if (not gROI):
    #     print("空框选，退出")
    #     quit()
    #
    # [ix, iy, w, h] = gROI  # (430, 196, 53, 38) 4个参数是矩形左上角的坐标和矩形的宽高
    # test = [0, 1, ix, iy, w, h]
    # cx = ix + w
    # cy = iy + h
    # test = code.yolo_detect(pathIn, pathOut)
    im, result = det.detect(the_first_frame)
    x1=result[0][0]
    y1=result[0][1]
    x2=result[0][2]
    y2=result[0][3]
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('./Outputframes/frame_000001.jpg', im)
    pathOut = './Outputframes/frame_000001.jpg'


    w = result[0][2] - result[0][0]
    h = result[0][3] - result[0][1]
    test = [0, 1, result[0][0], result[0][1], w, h]

    Text_DetectLog.config(state="normal")
    Text_DetectLog.delete('1.0', 'end')
    # Text_DetectLog.insert("insert", test[6] + "--------" + test[7])
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

        #
        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        the_first_frame_flag, the_first_frame = cap.read()
        cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
        pathIn = 'get_the_first_frame.jpg'
        pathOut = './Outputframes/frame_000001.jpg'
        # frameZoom = cv2.imread(pathIn)
        # test = code.yolo_detect(pathIn, pathOut)
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        # 即Box=[(x1,y1),(x2,y2)]
        # cv2.rectangle(frameZoom, Box[0], Box[1], (0, 0, 205), 2)
        tracker = kcftracker.KCFTracker(True, True, True)
        ix = Box[0][0]
        iy = Box[0][1]
        cx = Box[1][0]
        cy = Box[1][1]
        Begin_train_flag = True
        framenum = 0
        # print(Box)
        print(framenum)
        Text_TrackingLog.insert(tkinter.END, '***********执行跟踪程序*************\n')
        Missflag = False
        count = 120
        while (cap.isOpened()):
            Text_TrackingLog.config(state='normal')
            # Text_TrackingLog2.delete('1.0', 'end')
            # count-=1
            sub_frame = []
            if (count == 0):
                TrackingStr2 = ("**************自动重检**************\n")
                test = code.yolo_detect(pathIn, pathOut)
                print("test:", test)
                Box = [(test[2], test[3])]
                Box.append((test[2] + test[4], test[3] + test[5]))
                # cv2.rectangle(frameZoom, Box[0], Box[1], (0, 0, 205), 2)
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

                # print([ix, iy, w, h])
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
                duration = t1 - t0
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

        # finalfpsperformance=0
        # for i in range(len(Performancelistx)):
        #     finalfpsperformance+=float(Performancelisty[i])
        # finalfpsperformance=finalfpsperformance/(len(Performancelistx))
        # plt.figure(figsize=(16, 4))
        # plt.plot(Performancelistx, Performancelisty)
        # plt.title("fps_final_performance:with an average of"+str(round(finalfpsperformance,2))+"fps")
        # plt.xlabel('frames')
        # plt.ylabel('fps')
        # plt.legend()
        # plt.savefig("./FinalPerformance/fpsfinalperformance.jpg")

        # finaliouperformance = 0
        # for i in range(len(IOUlisty)):
        #     finaliouperformance += float(IOUlisty[i])
        # finaliouperformance = finaliouperformance / len(IOUlisty)
        # plt.figure(figsize=(16, 4))
        # plt.plot(IOUlistx, IOUlisty)
        # plt.title("iou_final_performance:with an average of"+str(round(finaliouperformance,2)))
        # plt.xlabel('frames')
        # plt.ylabel('iou')
        # plt.legend()
        # plt.savefig("./FinalPerformance/ioufinalperformance.jpg")

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
        # TrackingBoundingBoxPath="./framespositionstore/KCF_withAdaptiveFrames.txt"
        # 假如yolo没有检测到目标，就会直接报错
        # 假如yolo每一帧都检测，速度会减慢100倍
        selectingObject = False
        initTracking = False
        onTracking = False
        ix, iy, cx, cy = -1, -1, -1, -1
        w, h = 0, 0
        duration = 0.01
        Performancelistx = []
        Performancelisty = []
        KCFGroundList = []

        # def draw_boundingbox(event, x, y, flags, param):
        #     global selectingObject, initTracking, onTracking, ix, iy, cx, cy, w, h
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         selectingObject = True
        #         onTracking = False
        #         ix, iy = x, y
        #         cx, cy = x, y
        #     elif event == cv2.EVENT_MOUSEMOVE:
        #         cx, cy = x, y
        #     elif event == cv2.EVENT_LBUTTONUP:
        #         selectingObject = False
        #         if (abs(x - ix) > 10 and abs(y - iy) > 10):
        #             w, h = abs(x - ix), abs(y - iy)
        #             ix, iy = min(x, ix), min(y, iy)
        #             initTracking = True
        #         else:
        #             onTracking = False
        #     elif event == cv2.EVENT_RBUTTONDOWN:
        #         onTracking = False
        #         if (w > 0):
        #             ix, iy = x - w / 2, y - h / 2
        #             initTracking = True

        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        # cv2.rectangle(frameZoom, Box[0], Box[1], (0, 0, 205), 2)
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
        yolocount = 0
        while (cap.isOpened()):
            Text_TrackingLog.config(state='normal')
            # Text_TrackingLog2.delete('1.0', 'end')
            # count-=1
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
                # print([ix, iy, w, h])
                tracker.init([ix, iy, cx - ix, cy - iy], frameEach)
                Begin_train_flag = False
                onTracking = True
            elif (onTracking):
                print(yolocount)
                yolocount = yolocount + 1
                t0 = time.time()
                if (yolocount % 10 == 0):
                    cv2.imwrite("get_the_first_frame.jpg", frameEach)
                    pathIn = './get_the_first_frame.jpg'
                    pathOut = './yoloframebox/frame_000001.jpg'
                    # test = code.yolo_detect(pathIn, pathOut)
                    im, result = det.detect(frameEach)
                    if len(result) > 0:
                        w = result[0][2] - result[0][0]
                        h = result[0][3] - result[0][1]
                        test = [0, 1, result[0][0], result[0][1], w, h]
                        cv2.rectangle(frameEach, (test[2], test[3]),
                                      (test[2] + test[4], test[3] + test[5]), (0, 255, 0), 10)
                        boundingbox = tracker.update(frameEach)
                        boundingbox = list(map(int, boundingbox))
                        dx = boundingbox[0] - test[2]
                        dy = boundingbox[1] - test[3]
                        dw = boundingbox[2] - test[4]
                        dh = boundingbox[3] - test[5]
                        print("boundingbox-yolo:", [dx, dy, dw, dh])
                        if (dx * dx + dy * dy) > 200:
                            # frameEach = im
                            tracker.init([test[2], test[3], test[4], test[5]], im)
                boundingbox = tracker.update(frameEach)
                t1 = time.time()
                boundingbox = list(map(int, boundingbox))
                print("boundingbox_2:", boundingbox)
                if boundingbox[2] != 0 and boundingbox[3] != 0:
                    sub_frame = frameEach[boundingbox[1]: boundingbox[1] + boundingbox[3],
                                boundingbox[0]: boundingbox[0] + boundingbox[2]]
                cv2.rectangle(frameEach, (boundingbox[0], boundingbox[1]),
                              (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]), (0, 0, 255), 2)
                # sub_frame = frameEach[boundingbox[0] : boundingbox[0] + boundingbox[2] , boundingbox[1] : boundingbox[1] + boundingbox[3]]
                duration = 0.99 * duration + 0.01 * (t1 - t0)
                # duration = t1 - t0
                KCFGroundList.append(boundingbox)

                # TrackingStr2 = ("Frames-per-second:" + str(1 / duration)[:4].strip('.') + "\n")
                # Performancelistx.append(framenum)
                # Performancelisty.append(str(1 / duration)[:4].strip('.'))
                # if (framenum % 20 == 0):
                #     plt.figure(figsize=(16, 4))
                #     fpsPerformancePathname = ("./fpsPerformanceframes/frame_00000" + str(framenum) + ".jpg")
                #     plt.title("fps_performance")
                #     plt.plot(Performancelistx, Performancelisty)
                #     plt.xlabel('frames')
                #     plt.ylabel('fps')
                #     plt.legend()
                #     plt.savefig(fpsPerformancePathname)
                #     Imgfps = Image.open(fpsPerformancePathname)
                #     Imgfps = Imgfps.resize((390, 230), Image.ANTIALIAS)
                #     Imgfps = ImageTk.PhotoImage(Imgfps)
                #     Label_Fpsperformanceshows.config(image=Imgfps)
                #     Label_Fpsperformanceshows.image = Imgfps
                #     Label_Fpsperformanceshows.update()
                # Text_TrackingLog.insert(tkinter.END, TrackingStr2)
                # Text_TrackingLog.update()
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

        # finalfpsperformance = 0
        # for i in range(len(Performancelistx)):
        #     finalfpsperformance += float(Performancelisty[i])
        # finalfpsperformance = finalfpsperformance / 1
        # plt.figure(figsize=(16, 4))
        # plt.plot(Performancelistx, Performancelisty)
        # plt.title("fps_final_performance:with an average of" + str(round(finalfpsperformance, 2)) + "fps")
        # plt.xlabel('frames')
        # plt.ylabel('fps')
        # plt.legend()
        # plt.savefig("./FinalPerformance/fpsfinalperformance.jpg")

        file_write_obj = open("./framespositionstore/KCF_withAdaptiveFrames.txt", 'w')
        for i in range(len(KCFGroundList)):
            file_write_obj.writelines(str(KCFGroundList[i]))
            file_write_obj.write('\n')
        file_write_obj.close()

        Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
        Text_TrackingLog.update()
        Text_TrackingLog.config(state='disabled')
        cap.release()
    # elif(Algorithm_name=="SiamRPN" and Test_RecGroundTruth_Name==""):
    #     TrackingBoundingBoxPath = "B:\Document\Code\\586\KCFTest\\framespositionstore\RPN_Frames.txt"
    #     cap = cv2.VideoCapture("B:\Document\Code\\586\KCFTest\\testVideo\\aeroplane_of_desert.avi")
    #     net_path = 'pretrained/siamrpn/model.pth'
    #     tracker = TrackerSiamRPN(net_path=net_path)
    #     train_flag = True
    #     Performancelistx = []
    #     Performancelisty = []
    #     RPNGroundlist=[]
    #     # cv2.namedWindow('frame')
    #     the_first_frame_flag, the_first_frame = cap.read()
    #     cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
    #     pathIn = 'get_the_first_frame.jpg'
    #     pathOut = './Outputframes/frame_000001.jpg'
    #     test = code.yolo_detect(pathIn, pathOut)
    #     Box = [(test[2], test[3])]
    #     Box.append(((test[2] + test[4]), test[3] + test[5]))
    #     print(Box)
    #     rect_flag = False
    #     framenum = 1
    #     frameflag=False
    #     while (cap.isOpened()):
    #         ret, frame= cap.read()
    #         # cv2.imshow('frame', frame)
    #         if not ret:
    #             break
    #         if train_flag:
    #             print('start training the tracker...')
    #             start_time = time.time()
    #             gtRect = [Box[0][0], Box[0][1], Box[1][0] - Box[0][0], Box[1][1] - Box[0][1]]
    #             # gtRect = [refPt[0][0], refPt[0][1], refPt[1][0] - refPt[0][0], refPt[1][1] - refPt[0][1]]
    #             tracker.init(frame, gtRect)
    #             total_time = time.time() - start_time
    #             print("Training used time:", total_time)
    #             train_flag = False
    #         else:
    #             start_time = time.time()
    #             res = tracker.update(frame)
    #             total_time = time.time() - start_time
    #             # print("Frames-per-second:", 1. / total_time)
    #             SiamRPNTrackingStr = ("Frames-per-second:" + str(round((1 / total_time), 4)) + "\n")
    #             Performancelistx.append(framenum)
    #             Performancelisty.append(str(1 / total_time)[:4].strip('.'))
    #             if (framenum % 20 == 0):
    #                 plt.figure(figsize=(16, 4))
    #                 fpsPerformancePathname = ("./fpsPerformanceframes/frame_00000" + str(framenum) + ".jpg")
    #                 plt.title("fps_performance")
    #                 plt.plot(Performancelistx, Performancelisty)
    #                 plt.xlabel('frames')
    #                 plt.ylabel('fps')
    #                 plt.legend()
    #                 plt.savefig(fpsPerformancePathname)
    #                 Imgfps = Image.open(fpsPerformancePathname)
    #                 Imgfps = Imgfps.resize((390, 230), Image.ANTIALIAS)
    #                 Imgfps = ImageTk.PhotoImage(Imgfps)
    #                 Label_Fpsperformanceshows.config(image=Imgfps)
    #                 Label_Fpsperformanceshows.image = Imgfps
    #                 Label_Fpsperformanceshows.update()
    #             Text_TrackingLog.insert(tkinter.END, SiamRPNTrackingStr)
    #             Text_TrackingLog.update()
    #             Text_TrackingLog.see(tkinter.END)
    #             cv2.rectangle(frame, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])),
    #                           (0, 255, 0), 2)
    #             RPNGroundlist.append([int(res[0]),int(res[1]),int(res[2]),int(res[3])])
    #
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #         TrackingFrame_Pathout = ("./Outputframes/frame_00000" + str(framenum) + ".jpg")
    #         framenum += 1
    #         cv2.imwrite(TrackingFrame_Pathout, frame)
    #         c = cv2.waitKey(27) & 0xFF
    #         if c == 27 or c == ord('q'):
    #             break
    #         Img2 = Image.open(TrackingFrame_Pathout)
    #         Img2 = Img2.resize((600, 400), Image.ANTIALIAS)
    #         Img2 = ImageTk.PhotoImage(Img2)
    #         Label_ObjectTrackingShow.config(image=Img2)
    #         Label_ObjectTrackingShow.image = Img2
    #         Label_ObjectTrackingShow.update()
    #     file_write_obj = open("./framespositionstore/RPN_Frames.txt", 'w')
    #     finalfpsperformance = 0
    #     for i in range(len(Performancelistx)):
    #         finalfpsperformance += float(Performancelisty[i])
    #     finalfpsperformance = finalfpsperformance / (len(Performancelistx))
    #     plt.figure(figsize=(16, 4))
    #     plt.plot(Performancelistx, Performancelisty)
    #     plt.title("fps_final_performance:with an average of" + str(round(finalfpsperformance, 2)) + "fps")
    #     plt.xlabel('frames')
    #     plt.ylabel('fps')
    #     plt.legend()
    #     plt.savefig("B:\Document\Code\\586\KCFTest\FinalPerformance\\fpsfinalperformance.jpg")
    #     for i in range(len(RPNGroundlist)):
    #         file_write_obj.writelines(str(RPNGroundlist[i]))
    #         file_write_obj.write('\n')
    #     file_write_obj.close()
    #     Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
    #     Text_TrackingLog.update()
    #     Text_TrackingLog.config(state='disabled')
    #     cap.release()
    #     # cv2.destroyAllWindows()
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

        #
        cap = cv2.VideoCapture(Test_Video_Name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        the_first_frame_flag, the_first_frame = cap.read()
        cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
        pathIn = 'get_the_first_frame.jpg'
        pathOut = './Outputframes/frame_000001.jpg'
        # frameZoom = cv2.imread(pathIn)
        # test = code.yolo_detect(pathIn, pathOut)
        Box = [(test[2], test[3])]
        Box.append(((test[2] + test[4]), test[3] + test[5]))
        # cv2.rectangle(frameZoom, Box[0], Box[1], (0, 0, 205), 2)
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
            # Text_TrackingLog2.delete('1.0', 'end')
            # count-=1
            if (count == 0):
                TrackingStr2 = ("**************自动重检**************\n")
                test = code.yolo_detect(pathIn, pathOut)
                Box = [(test[2], test[3])]
                Box.append(((test[2] + test[4]), test[3] + test[5]))
                # cv2.rectangle(frameZoom, Box[0], Box[1], (0, 0, 205), 2)
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
                # print([ix, iy, w, h])
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

                # TrackingStr2 = ("Frames-per-second:" + str(1 / duration)[:4].strip('.') + "\n")
                # Performancelistx.append(framenum)
                # Performancelisty.append(str(1 / duration)[:4].strip('.'))
                # KCFGroundList.append(boundingbox)
                # # print("\\\\\\\\\\\\\\\\\\\\")
                # # print([ix,iy,cx,cy])
                # print(framenum)
                # Kx1 = int(KCFGroundList[framenum-2][0])
                # Ky1 = int(KCFGroundList[framenum -2][1])
                # Kx2 = int(KCFGroundList[framenum -2][2]) + Kx1
                # Ky2 = int(KCFGroundList[framenum -2][3]) + Ky1
                # Rx1 = int(RecGroundList[framenum -2].split(",")[0])
                # Ry1 = int(RecGroundList[framenum -2].split(",")[1])
                # Rx2 = int(RecGroundList[framenum -2].split(",")[2]) + Rx1
                # Ry2 = int(RecGroundList[framenum -2].split(",")[1]) + Ry1
                # IOUx1 = 0
                # IOUx2 = 0
                # IOUy1 = 0
                # IOUy2 = 0
                # if Rx1 <= Kx1:
                #     IOUx1 = Kx1
                # else:
                #     IOUx1 = Rx1
                # if Ry1 <= Ky1:
                #     IOUy1 = Ky1
                # else:
                #     IOUy1 = Ry1
                # if Rx2 > Kx2:
                #     IOUx2 = Kx2
                # else:
                #     IOUx2 = Rx2
                # if Ry2 > Ky2:
                #     IOUy2 = Ky2
                # else:
                #     IOUy2 = Ry2
                # SKCF = (Kx2 - Kx1) * (Ky2 - Ky1)
                # # print(SKCF)
                # SRec = (Rx2 - Rx1) * (Ry2 - Ry1)
                # # print(SRec)
                # SBingo = (IOUx2 - IOUx1) * (IOUy2 - IOUy1)
                # # print(SBingo)
                # IOU = ((SBingo) / (SKCF + SRec - SBingo))
                # IOUlistx.append(framenum)
                # IOUlisty.append(abs(IOU))
                # if(framenum%20==0):
                #     plt.figure(figsize=(16, 4))
                #     IOUPerformancePathname = ("./IOUPerformanceframes/frame_00000" + str(framenum) + ".jpg")
                #     plt.title("iou_performance")
                #     plt.plot(IOUlistx, IOUlisty)
                #     plt.xlabel('frames')
                #     plt.ylabel('iou')
                #     plt.legend()
                #     plt.savefig(IOUPerformancePathname)
                #     Imgiou = Image.open(IOUPerformancePathname)
                #     Imgiou = Imgiou.resize((390, 230), Image.ANTIALIAS)
                #     Imgiou = ImageTk.PhotoImage(Imgiou)
                #     Label_IOUperfpormanceshows.config(image=Imgiou)
                #     Label_IOUperfpormanceshows.image = Imgiou
                #     Label_IOUperfpormanceshows.update()

                #     plt.figure(figsize=(16, 4))
                #     fpsPerformancePathname = ("./fpsPerformanceframes/frame_00000" + str(framenum) + ".jpg")
                #     plt.title("fps_performance")
                #     plt.plot(Performancelistx, Performancelisty)
                #     plt.xlabel('frames')
                #     plt.ylabel('fps')
                #     plt.legend()
                #     plt.savefig(fpsPerformancePathname)
                #     Imgfps = Image.open(fpsPerformancePathname)
                #     Imgfps = Imgfps.resize((390, 230), Image.ANTIALIAS)
                #     Imgfps = ImageTk.PhotoImage(Imgfps)
                #     Label_Fpsperformanceshows.config(image=Imgfps)
                #     Label_Fpsperformanceshows.image = Imgfps
                #     Label_Fpsperformanceshows.update()
                # Text_TrackingLog.insert(tkinter.END, TrackingStr2)
                # Text_TrackingLog.update()
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

        # finalfpsperformance=0
        # for i in range(len(Performancelistx)):
        #     finalfpsperformance+=float(Performancelisty[i])
        # finalfpsperformance=finalfpsperformance/(len(Performancelistx))
        # plt.figure(figsize=(16, 4))
        # plt.plot(Performancelistx, Performancelisty)
        # plt.title("fps_final_performance:with an average of"+str(round(finalfpsperformance,2))+"fps")
        # plt.xlabel('frames')
        # plt.ylabel('fps')
        # plt.legend()
        # plt.savefig("./FinalPerformance/fpsfinalperformance.jpg")

        # finaliouperformance = 0
        # for i in range(len(IOUlisty)):
        #     finaliouperformance += float(IOUlisty[i])
        # finaliouperformance = finaliouperformance / len(IOUlisty)
        # plt.figure(figsize=(16, 4))
        # plt.plot(IOUlistx, IOUlisty)
        # plt.title("iou_final_performance:with an average of"+str(round(finaliouperformance,2)))
        # plt.xlabel('frames')
        # plt.ylabel('iou')
        # plt.legend()
        # plt.savefig("./FinalPerformance/ioufinalperformance.jpg")

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
        # cv2.rectangle(frameZoom, Box[0], Box[1], (0, 0, 205), 2)
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
            # Text_TrackingLog2.delete('1.0', 'end')
            # count-=1
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
                # print([ix, iy, w, h])
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
                duration = t1 - t0
                KCFGroundList.append(boundingbox)

                # TrackingStr2 = ("Frames-per-second:" + str(1 / duration)[:4].strip('.') + "\n")
                # Performancelistx.append(framenum)
                # Performancelisty.append(str(1 / duration)[:4].strip('.'))
                # if (framenum % 20 == 0):
                #     plt.figure(figsize=(16, 4))
                #     fpsPerformancePathname = ("./fpsPerformanceframes/frame_00000" + str(framenum) + ".jpg")
                #     plt.title("fps_performance")
                #     plt.plot(Performancelistx, Performancelisty)
                #     plt.xlabel('frames')
                #     plt.ylabel('fps')
                #     plt.legend()
                #     plt.savefig(fpsPerformancePathname)
                #     Imgfps = Image.open(fpsPerformancePathname)
                #     Imgfps = Imgfps.resize((390, 230), Image.ANTIALIAS)
                #     Imgfps = ImageTk.PhotoImage(Imgfps)
                #     Label_Fpsperformanceshows.config(image=Imgfps)
                #     Label_Fpsperformanceshows.image = Imgfps
                #     Label_Fpsperformanceshows.update()
                # Text_TrackingLog.insert(tkinter.END, TrackingStr2)
                # Text_TrackingLog.update()
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

        # finalfpsperformance = 0
        # for i in range(len(Performancelistx)):
        #     finalfpsperformance += float(Performancelisty[i])
        # finalfpsperformance = finalfpsperformance / (len(Performancelistx))
        # plt.figure(figsize=(16, 4))
        # plt.plot(Performancelistx, Performancelisty)
        # plt.title("fps_final_performance:with an average of" + str(round(finalfpsperformance, 2)) + "fps")
        # plt.xlabel('frames')
        # plt.ylabel('fps')
        # plt.legend()
        # plt.savefig("./FinalPerformance/fpsfinalperformance.jpg")

        file_write_obj = open("./framespositionstore/KCF_withAdaptiveFrames.txt", 'w')
        for i in range(len(KCFGroundList)):
            file_write_obj.writelines(str(KCFGroundList[i]))
            file_write_obj.write('\n')
        file_write_obj.close()

        Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
        Text_TrackingLog.update()
        Text_TrackingLog.config(state='disabled')
        cap.release()

    # elif(Algorithm_name=="SiamRPN" and Test_RecGroundTruth_Name!=""):
    #     TrackingBoundingBoxPath = "B:\Document\Code\\586\KCFTest\\framespositionstore\RPN_Frames.txt"
    #     cap = cv2.VideoCapture(Test_Video_Name)
    #     net_path = 'pretrained/siamrpn/model.pth'
    #     tracker = TrackerSiamRPN(net_path=net_path)
    #     train_flag = True
    #     Performancelistx = []
    #     Performancelisty = []
    #     RPNGroundlist = []
    #     RecGroundList = []
    #     IOUlistx = []
    #     IOUlisty = []
    #     # cv2.namedWindow('frame')
    #     dogmp4IOUfilename1 = Test_RecGroundTruth_Name
    #     file = open(dogmp4IOUfilename1, "r")
    #     line = file.readline()
    #     while (line):
    #         # print(line)
    #         line = file.readline()
    #         line = line.rstrip("\n")
    #         RecGroundList.append(line)
    #     file.close()
    #     the_first_frame_flag, the_first_frame = cap.read()
    #     cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
    #     pathIn = 'get_the_first_frame.jpg'
    #     pathOut = './Outputframes/frame_000001.jpg'
    #     test = code.yolo_detect(pathIn, pathOut)
    #     Box = [(test[2], test[3])]
    #     Box.append(((test[2] + test[4]), test[3] + test[5]))
    #     print(Box)
    #     rect_flag = False
    #     framenum = 1
    #     frameflag = False
    #     while (cap.isOpened()):
    #         ret, frame = cap.read()
    #         # cv2.imshow('frame', frame)
    #         if not ret:
    #             break
    #         if train_flag:
    #             print('start training the tracker...')
    #             start_time = time.time()
    #             gtRect = [Box[0][0], Box[0][1], Box[1][0] - Box[0][0], Box[1][1] - Box[0][1]]
    #             # gtRect = [refPt[0][0], refPt[0][1], refPt[1][0] - refPt[0][0], refPt[1][1] - refPt[0][1]]
    #             tracker.init(frame, gtRect)
    #             total_time = time.time() - start_time
    #             print("Training used time:", total_time)
    #             train_flag = False
    #         else:
    #             start_time = time.time()
    #             res = tracker.update(frame)
    #             total_time = time.time() - start_time
    #             # print("Frames-per-second:", 1. / total_time)
    #             SiamRPNTrackingStr = ("Frames-per-second:" + str(round((1 / total_time), 4)) + "\n")
    #             Performancelistx.append(framenum)
    #             Performancelisty.append(str(1 / total_time)[:4].strip('.'))
    #             RPNGroundlist.append(res)
    #             Kx1 = int(RPNGroundlist[framenum - 2][0])
    #             Ky1 = int(RPNGroundlist[framenum - 2][1])
    #             Kx2 = int(RPNGroundlist[framenum - 2][2]) + Kx1
    #             Ky2 = int(RPNGroundlist[framenum - 2][3]) + Ky1
    #             Rx1 = int(RecGroundList[framenum - 2].split(",")[0])
    #             Ry1 = int(RecGroundList[framenum - 2].split(",")[1])
    #             Rx2 = int(RecGroundList[framenum - 2].split(",")[2]) + Rx1
    #             Ry2 = int(RecGroundList[framenum - 2].split(",")[1]) + Ry1
    #             IOUx1 = 0
    #             IOUx2 = 0
    #             IOUy1 = 0
    #             IOUy2 = 0
    #             if Rx1 <= Kx1:
    #                 IOUx1 = Kx1
    #             else:
    #                 IOUx1 = Rx1
    #             if Ry1 <= Ky1:
    #                 IOUy1 = Ky1
    #             else:
    #                 IOUy1 = Ry1
    #             if Rx2 > Kx2:
    #                 IOUx2 = Kx2
    #             else:
    #                 IOUx2 = Rx2
    #             if Ry2 > Ky2:
    #                 IOUy2 = Ky2
    #             else:
    #                 IOUy2 = Ry2
    #             SKCF = (Kx2 - Kx1) * (Ky2 - Ky1)
    #             # print(SKCF)
    #             SRec = (Rx2 - Rx1) * (Ry2 - Ry1)
    #             # print(SRec)
    #             SBingo = (IOUx2 - IOUx1) * (IOUy2 - IOUy1)
    #             # print(SBingo)
    #             IOU = ((SBingo) / (SKCF + SRec - SBingo))
    #             IOUlistx.append(framenum)
    #             IOUlisty.append(abs(IOU))
    #             if (framenum % 20 == 0):
    #                 plt.figure(figsize=(16, 4))
    #                 IOUPerformancePathname = ("./IOUPerformanceframes/frame_00000" + str(framenum) + ".jpg")
    #                 plt.title("iou_performance")
    #                 plt.plot(IOUlistx, IOUlisty)
    #                 plt.xlabel('frames')
    #                 plt.ylabel('iou')
    #                 plt.legend()
    #                 plt.savefig(IOUPerformancePathname)
    #                 Imgiou = Image.open(IOUPerformancePathname)
    #                 Imgiou = Imgiou.resize((390, 230), Image.ANTIALIAS)
    #                 Imgiou = ImageTk.PhotoImage(Imgiou)
    #                 Label_IOUperfpormanceshows.config(image=Imgiou)
    #                 Label_IOUperfpormanceshows.image = Imgiou
    #                 Label_IOUperfpormanceshows.update()
    #
    #                 plt.figure(figsize=(16, 4))
    #                 fpsPerformancePathname = ("./fpsPerformanceframes/frame_00000" + str(framenum) + ".jpg")
    #                 plt.title("fps_performance")
    #                 plt.plot(Performancelistx, Performancelisty)
    #                 plt.xlabel('frames')
    #                 plt.ylabel('fps')
    #                 plt.legend()
    #                 plt.savefig(fpsPerformancePathname)
    #                 Imgfps = Image.open(fpsPerformancePathname)
    #                 Imgfps = Imgfps.resize((390, 230), Image.ANTIALIAS)
    #                 Imgfps = ImageTk.PhotoImage(Imgfps)
    #                 Label_Fpsperformanceshows.config(image=Imgfps)
    #                 Label_Fpsperformanceshows.image = Imgfps
    #                 Label_Fpsperformanceshows.update()
    #
    #             Text_TrackingLog.insert(tkinter.END, SiamRPNTrackingStr)
    #             Text_TrackingLog.update()
    #             Text_TrackingLog.see(tkinter.END)
    #             cv2.rectangle(frame, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])),
    #                           (0, 255, 0), 2)
    #             RPNGroundlist.append([int(res[0]), int(res[1]), int(res[2]), int(res[3])])
    #
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #         TrackingFrame_Pathout = ("./Outputframes/frame_00000" + str(framenum) + ".jpg")
    #         framenum += 1
    #         cv2.imwrite(TrackingFrame_Pathout, frame)
    #         c = cv2.waitKey(27) & 0xFF
    #         if c == 27 or c == ord('q'):
    #             break
    #         Img2 = Image.open(TrackingFrame_Pathout)
    #         Img2 = Img2.resize((600, 400), Image.ANTIALIAS)
    #         Img2 = ImageTk.PhotoImage(Img2)
    #         Label_ObjectTrackingShow.config(image=Img2)
    #         Label_ObjectTrackingShow.image = Img2
    #         Label_ObjectTrackingShow.update()
    #     file_write_obj = open("./framespositionstore/RPN_Frames.txt", 'w')
    #     finalfpsperformance = 0
    #     for i in range(len(Performancelistx)):
    #         finalfpsperformance += float(Performancelisty[i])
    #     finalfpsperformance = finalfpsperformance / (len(Performancelistx))
    #     plt.figure(figsize=(16, 4))
    #     plt.plot(Performancelistx, Performancelisty)
    #     plt.title("fps_final_performance:with an average of" + str(round(finalfpsperformance, 2)) + "fps")
    #     plt.xlabel('frames')
    #     plt.ylabel('fps')
    #     plt.legend()
    #     plt.savefig("B:\Document\Code\\586\KCFTest\FinalPerformance\\fpsfinalperformance.jpg")
    #     finaliouperformance = 0
    #     for i in range(len(IOUlisty)):
    #         finaliouperformance += float(IOUlisty[i])
    #     finaliouperformance = finaliouperformance / len(IOUlisty)
    #     plt.figure(figsize=(16, 4))
    #     plt.plot(IOUlistx, IOUlisty)
    #     plt.title("iou_final_performance:with an average of" + str(round(finaliouperformance, 2)))
    #     plt.xlabel('frames')
    #     plt.ylabel('iou')
    #     plt.legend()
    #     plt.savefig("B:\Document\Code\\586\KCFTest\FinalPerformance\\ioufinalperformance.jpg")
    #     for i in range(len(RPNGroundlist)):
    #         file_write_obj.writelines(str(RPNGroundlist[i]))
    #         file_write_obj.write('\n')
    #     file_write_obj.close()
    #     Text_TrackingLog.insert(tkinter.END, "*************跟踪结束**************\n")
    #     Text_TrackingLog.update()
    #     Text_TrackingLog.config(state='disabled')
    #     cap.release()
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

                    # TrackingStr2 = ("Frames-per-second:" + str(round((1 / total_time), 4))  + "\n")
                    # Text_TrackingLog.insert(tkinter.END,TrackingStr2)
                    # Text_TrackingLog.update()
                    # Text_TrackingLog.see(tkinter.END)
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
                    # Performancelistx.append(framenum)
                    # Performancelisty.append(str(1 / total_time)[:4].strip('.'))
                    # if(framenum%20==0):
                    #     plt.figure(figsize=(16, 4))
                    #     fpsPerformancePathname = ("./fpsPerformanceframes/frame_00000" + str(framenum) + ".jpg")
                    #     plt.title("fps_performance")
                    #     plt.plot(Performancelistx, Performancelisty)
                    #     plt.xlabel('frames')
                    #     plt.ylabel('fps')
                    #     plt.legend()
                    #     plt.savefig(fpsPerformancePathname)
                    #     Imgfps = Image.open(fpsPerformancePathname)
                    #     Imgfps = Imgfps.resize((390, 230), Image.ANTIALIAS)
                    #     Imgfps = ImageTk.PhotoImage(Imgfps)
                    #     Label_Fpsperformanceshows.config(image=Imgfps)
                    #     Label_Fpsperformanceshows.image = Imgfps
                    #     Label_Fpsperformanceshows.update()
            else:
                break
        # finalfpsperformance = 0
        # for i in range(len(Performancelistx)):
        #     finalfpsperformance += float(Performancelisty[i])
        # finalfpsperformance = finalfpsperformance / (len(Performancelistx))
        # plt.figure(figsize=(16, 4))
        # plt.plot(Performancelistx, Performancelisty)
        # plt.title("fps_final_performance:with an average of" + str(round(finalfpsperformance, 2)) + "fps")
        # plt.xlabel('frames')
        # plt.ylabel('fps')
        # plt.legend()
        # plt.savefig("./FinalPerformance/fpsfinalperformance.jpg")
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
                    # TrackingStr2 = ("Frames-per-second:" + str(round((1 / total_time), 4)) + "\n")
                    # Text_TrackingLog.insert(tkinter.END, TrackingStr2)
                    # Text_TrackingLog.update()
                    # Text_TrackingLog.see(tkinter.END)
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
                    # Performancelistx.append(framenum)
                    # Performancelisty.append(str(1 / total_time)[:4].strip('.'))
                    # print(framenum)
                    # Kx1 = int(KCFGroundList[framenum - 2][0])
                    # Ky1 = int(KCFGroundList[framenum - 2][1])
                    # Kx2 = int(KCFGroundList[framenum - 2][2]) + Kx1
                    # Ky2 = int(KCFGroundList[framenum - 2][3]) + Ky1
                    # Rx1 = int(RecGroundList[framenum - 2].split(",")[0])
                    # Ry1 = int(RecGroundList[framenum - 2].split(",")[1])
                    # Rx2 = int(RecGroundList[framenum - 2].split(",")[2]) + Rx1
                    # Ry2 = int(RecGroundList[framenum - 2].split(",")[1]) + Ry1
                    # IOUx1 = 0
                    # IOUx2 = 0
                    # IOUy1 = 0
                    # IOUy2 = 0
                    # if Rx1 <= Kx1:
                    #     IOUx1 = Kx1
                    # else:
                    #     IOUx1 = Rx1
                    # if Ry1 <= Ky1:
                    #     IOUy1 = Ky1
                    # else:
                    #     IOUy1 = Ry1
                    # if Rx2 > Kx2:
                    #     IOUx2 = Kx2
                    # else:
                    #     IOUx2 = Rx2
                    # if Ry2 > Ky2:
                    #     IOUy2 = Ky2
                    # else:
                    #     IOUy2 = Ry2
                    # SKCF = (Kx2 - Kx1) * (Ky2 - Ky1)
                    # # print(SKCF)
                    # SRec = (Rx2 - Rx1) * (Ry2 - Ry1)
                    # # print(SRec)
                    # SBingo = (IOUx2 - IOUx1) * (IOUy2 - IOUy1)
                    # # print(SBingo)
                    # IOU = ((SBingo) / (SKCF + SRec - SBingo))
                    # IOUlistx.append(framenum)
                    # IOUlisty.append(abs(IOU))
                    # if (framenum % 20 == 0):
                    #     plt.figure(figsize=(16, 4))
                    #     fpsPerformancePathname = ("./fpsPerformanceframes/frame_00000" + str(framenum) + ".jpg")
                    #     plt.title("fps_performance")
                    #     plt.plot(Performancelistx, Performancelisty)
                    #     plt.xlabel('frames')
                    #     plt.ylabel('fps')
                    #     plt.legend()
                    #     plt.savefig(fpsPerformancePathname)
                    #     Imgfps = Image.open(fpsPerformancePathname)
                    #     Imgfps = Imgfps.resize((390, 230), Image.ANTIALIAS)
                    #     Imgfps = ImageTk.PhotoImage(Imgfps)
                    #     Label_Fpsperformanceshows.config(image=Imgfps)
                    #     Label_Fpsperformanceshows.image = Imgfps
                    #     Label_Fpsperformanceshows.update()

                    #     plt.figure(figsize=(16, 4))
                    #     IOUPerformancePathname = ("./IOUPerformanceframes/frame_00000" + str(framenum) + ".jpg")
                    #     plt.title("iou_performance")
                    #     plt.plot(IOUlistx, IOUlisty)
                    #     plt.xlabel('frames')
                    #     plt.ylabel('iou')
                    #     plt.legend()
                    #     plt.savefig(IOUPerformancePathname)
                    #     Imgiou = Image.open(IOUPerformancePathname)
                    #     Imgiou = Imgiou.resize((390, 230), Image.ANTIALIAS)
                    #     Imgiou = ImageTk.PhotoImage(Imgiou)
                    #     Label_IOUperfpormanceshows.config(image=Imgiou)
                    #     Label_IOUperfpormanceshows.image = Imgiou
                    #     Label_IOUperfpormanceshows.update()
            else:
                break
        # finalfpsperformance = 0
        # for i in range(len(Performancelistx)):
        #     finalfpsperformance += float(Performancelisty[i])
        # finalfpsperformance = finalfpsperformance / (len(Performancelistx))
        # plt.figure(figsize=(16, 4))
        # plt.plot(Performancelistx, Performancelisty)
        # plt.title("fps_final_performance:with an average of" + str(round(finalfpsperformance, 2)) + "fps")
        # plt.xlabel('frames')
        # plt.ylabel('fps')
        # plt.legend()
        # plt.savefig("./FinalPerformance/fpsfinalperformance.jpg")
        # finaliouperformance = 0
        # for i in range(len(IOUlisty)):
        #     finaliouperformance += float(IOUlisty[i])
        # finaliouperformance = finaliouperformance / len(IOUlisty)
        # plt.figure(figsize=(16, 4))
        # plt.plot(IOUlistx, IOUlisty)
        # plt.title("iou_final_performance:with an average of" + str(round(finaliouperformance, 2)))
        # plt.xlabel('frames')
        # plt.ylabel('iou')
        # plt.legend()
        # plt.savefig("./sourcece/ioufinalperformance.jpg")
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
    # Label_ObjectTracking = tkinter.Label(Windows, text="实时跟踪情况", width=85, height=2, background="pink")
    # Label_ObjectTracking.place(x=520, y=10)
    # ButtonVideoReply1 = tkinter.Button(Windows_Reply, text="选择该算法", width=10, height=2, command=ButtonVideoRePlyOne)
    # ButtonVideoReply1.place(x=350, y=10)
    # ButtonVideoReply2 = tkinter.Button(Windows_Reply, text="选择该算法", width=10, height=2, command=ButtonVideoRePlyTwo)
    # ButtonVideoReply2.place(x=350, y=70)
    #
    # '''
    # 新的按钮功能：
    # 选择测试数据
    # '''
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
            Test_Video_Name = r"C:\dataset\HIT_code_video\video_figure\video\3.avi"
            # E:\dplearning\KCFTest\testVideo
            # Test_Video_Name = "./testVideo/aeroplane_of_desert.mp4"
            # E:\wenmiao\hagongda\video\采集视频\5.avi
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
    # Label_ObjectTracking = tkinter.Label(Windows, text="实时跟踪情况", width=85, height=2, background="pink")
    # Label_ObjectTracking.place(x=520, y=10)
    # ButtonVideoReply1 = tkinter.Button(Windows_Reply,text="选择该视频",width=10,height=2,command=ButtonVideoRePlyOne)
    # ButtonVideoReply1.place(x=350, y=10)
    # ButtonVideoReply2 = tkinter.Button(Windows_Reply, text="选择该视频", width=10, height=2,command=ButtonVideoRePlyTwo)
    # ButtonVideoReply2.place(x=350, y=70)
    # ButtonVideoReply3 = tkinter.Button(Windows_Reply, text="选择该视频", width=10, height=2,command=ButtonVideoRePlyThree)
    # ButtonVideoReply3.place(x=350, y=130)
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
        # PicChange = Image.open('./Outputframes/frame_00000' +str(i)+".jpg" )
        # PicChange = PicChange.resize((720, 576), Image.ANTIALIAS)
        # PicChange
        # B:\Document\Code\586\KCFTest\Outputframes\frame_000001.jpg
        Pathshit = ("Outputframes/frame_00000" + str(i) + ".jpg")
        img = cv2.imread("Outputframes/frame_00000" + str(i) + ".jpg")
        try:
            img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            # print(img)
            video_writer.write(img)
        except:
            print("nope")
            continue
    video_writer.release()
    Text_TrackingLog.config(state='normal')
    Text_TrackingLog.insert(tkinter.END, "**********视频生成完毕*********\n")
    Text_TrackingLog.update()
    # print("视频生成完毕")
    # path_ = askdirectory()
    # path.set(path_)

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
    # Label_Fpsperformanceshows.config(image=Img3)
    # Label_Fpsperformanceshows.image = Img3
    # Label_Fpsperformanceshows.update()

    Img_Background4 = Image.open("./sourcece/black.jpg")
    Img_Background4 = Img_Background4.resize((390, 230), Image.ANTIALIAS)
    Img4 = ImageTk.PhotoImage(Img_Background4)
    # Label_IOUperfpormanceshows.config(image=Img4)
    # Label_IOUperfpormanceshows.image = Img4
    # Label_IOUperfpormanceshows.update()

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

# Label_DetectLog=tkinter.Label(Windows,text="识别目标种类/置信率：",width=47,height=1,background=color_label)
# Label_DetectLog.place(x=670,y=410)
Text_DetectLog = tkinter.Text(Windows, font=("Arial", 12), width=37, height=1, background="white")
# Text_DetectLog.insert(1.0,"无信号输入")
# Text_DetectLog.place(x=670,y=440)
# Text_DetectLog.config(state='disabled')
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
# Label_TrackingPerformanceTitle=tkinter.Label(Windows,text="跟踪性能(left-right:fps-IOU)",width=73,height=1,background=color_label,font=tkfontset1)
# Label_TrackingPerformanceTitle.place(x=10,y=480)
# Img_Background3 = Img_Background.resize((390,230),Image.ANTIALIAS)
# Img3=ImageTk.PhotoImage(Img_Background3)
# Label_Fpsperformanceshows=tkinter.Label(Windows,width=390,height=230,background=color_back,image=Img3)
# Label_Fpsperformanceshows.place(x=10,y=505)
# Label_IOUperfpormanceshows=tkinter.Label(Windows,width=390,height=230,background=color_back,image=Img3)
# Label_IOUperfpormanceshows.place(x=420,y=505)
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

# os.system("pause")
