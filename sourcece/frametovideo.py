import cv2
import os
fps=10
fourcc = cv2.VideoWriter_fourcc("I", "4", "2", "0")
video_writer = cv2.VideoWriter('./panda.avi', fourcc, fps, (312, 233))
Dir_length=(len([name for name in os.listdir("B:\Document\Code\\586\KCFTest\\testVideo\Panda\img") if os.path.isfile(os.path.join("B:\Document\Code\\586\KCFTest\\testVideo\Panda\img", name))]))
print(Dir_length)
for i in range(Dir_length-1):
    i=i+1
    # PicChange = Image.open('./Outputframes/frame_00000' +str(i)+".jpg" )
    # PicChange = PicChange.resize((720, 576), Image.ANTIALIAS)
    # PicChange
    # B:\Document\Code\586\KCFTest\Outputframes\frame_000001.jpg
    Pathshit=("B:\Document\Code\\586\KCFTest\\testVideo\Panda\img\\0" +str(i)+".jpg")
    print(Pathshit)
    img = cv2.imread("B:\Document\Code\\586\KCFTest\\testVideo\Panda\img\\0" +str(i)+".jpg" )
    try:
        img = cv2.resize(img, (312, 233), interpolation=cv2.INTER_CUBIC)
    # print(img)
        video_writer.write(img)
    except:
        print("nope")
        continue
video_writer.release()
