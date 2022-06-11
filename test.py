import cv2

Test_Video_Name = r"C:\dataset\HIT_code_video\video_figure\video\scene7.mp4"
cap = cv2.VideoCapture(Test_Video_Name)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
ret, frameEach = cap.read()
# print(frameEach)
sub_frame = frameEach[-1:0, -1:0]
print(sub_frame)

