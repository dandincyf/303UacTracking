import cv2
from sourcece.siamrpn import TrackerSiamRPN
import time

refPt = []
cropping = False
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    # ＃抓取对全局变量的引用
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    # 如果单击鼠标左键，则记录开始
    # （x，y）坐标并指示正在进行裁剪
    # 执行
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    # check to see if the left mouse button was released
    # 检查是否释放了鼠标左键
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        # ＃记录结束（x，y）坐标并指出
        # 裁剪操作完成
        refPt.append((x, y))
        cropping = False

        # draw a rectangle around the region of interest
        # 在感兴趣的区域周围绘制一个矩形
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("frame", frame)

def main():

    cap = cv2.VideoCapture("B:\Document\Code\\586\KCFTest\\testVideo\\1.avi")
    net_path = 'pretrained/siamrpn/model.pth'
    tracker = TrackerSiamRPN(net_path=net_path)
    train_flag = True
    cv2.namedWindow('frame')
    the_first_frame_flag, the_first_frame = cap.read()
    # cv2.imwrite("get_the_first_frame.jpg", the_first_frame)
    # pathIn = 'get_the_first_frame.jpg'
    # pathOut = './Outputframes/frame_000001.jpg'
    # frameZoom = cv2.imread(pathIn)
    # test = code.yolo_detect(pathIn, pathOut)
    # test = [0, 0, 58, 100, 28, 23]
    # Box = [(test[2], test[3])]
    # Box.append(((test[2] + test[4]), test[3] + test[5]))
    # print(Box)
    rect_flag = False
    while True:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        if not ret:
            break
        while not rect_flag:
            clone = frame.copy()
            key = cv2.waitKey(1) & 0xFF
            # # if the 'r' key is pressed, reset the cropping region
            # # 如果r键被按下，重置捕获区域
            if key == ord("r"):
                frame = clone.copy()
            # # if the 'c' key is pressed, break from the loop
            # # 如果c键被按下，跳出循环
            elif key == ord("c"):
                break

            # if there are two reference points, then crop the region of interest
            #     如果有两个参考点，则裁剪感兴趣的区域
            # from the image and display it
            #     从图像中显示出来

            if len(refPt) == 2:
                rect_flag = True
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                cv2.imshow("ROI", roi)
        if train_flag:
            print('start training the tracker...')
            start_time = time.time()
            # Box = [(test[2], test[3])]
            # Box.append(((test[2] + test[4]), test[3] + test[5]))
            # ix = Box[0][0]
            # iy = Box[0][1]
            # cx = Box[1][0]
            # cy = Box[1][1]
            # gtRect=[Box[0][0],Box[0][1],Box[1][0]-Box[0][0],Box[1][1]-Box[0][1]]
            gtRect = [refPt[0][0], refPt[0][1], refPt[1][0] - refPt[0][0], refPt[1][1] - refPt[0][1]]
            tracker.init(frame,gtRect)
            total_time = time.time() - start_time
            print("Training used time:", total_time)
            train_flag = False
        else:
            start_time = time.time()
            res = tracker.update(frame)
            total_time = time.time() - start_time
            print("Frames-per-second:", 1. / total_time)
            cv2.rectangle(frame, (int(res[0]), int(res[1])), (int(res[0] + res[2]), int(res[1] + res[3])), (0, 255, 0),
                          2)
            cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()