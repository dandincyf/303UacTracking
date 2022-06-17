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

Test_Video_Name = r"C:\dataset\HIT_code_video\video_figure\video\scene11.1.avi"
cap = cv2.VideoCapture(Test_Video_Name)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

while (cap.isOpened()):
    ret, frameEach = cap.read()
    if not ret:
        break
    # cv2.imwrite("get_the_first_frame.jpg", frameEach)
    cv2.imshow('1', frameEach)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break