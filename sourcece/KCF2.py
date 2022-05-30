import numpy as np
import cv2
import time
import sys
fps=10
import threading
PY3 = sys.version_info >= (3,)

if PY3:
    xrange = range

#前面是一些数学工具，用来计算复数和FFT等，看函数名
# ffttools
def fftd(img, backwards=False):
    # shape of img can be (m,n), (m,n,1) or (m,n,2)
    # in my test, fft provided by numpy and scipy are slower than cv2.dft
    return cv2.dft(np.float32(img), flags=(
        (cv2.DFT_INVERSE | cv2.DFT_SCALE) if backwards else cv2.DFT_COMPLEX_OUTPUT))  # 'flags =' is necessary!

def real(img):
    return img[:, :, 0]


def imag(img):
    return img[:, :, 1]


def complexMultiplication(a, b):
    res = np.zeros(a.shape, a.dtype)

    res[:, :, 0] = a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]
    res[:, :, 1] = a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]
    return res


def complexDivision(a, b):
    res = np.zeros(a.shape, a.dtype)
    divisor = 1. / (b[:, :, 0] ** 2 + b[:, :, 1] ** 2)

    res[:, :, 0] = (a[:, :, 0] * b[:, :, 0] + a[:, :, 1] * b[:, :, 1]) * divisor
    res[:, :, 1] = (a[:, :, 1] * b[:, :, 0] + a[:, :, 0] * b[:, :, 1]) * divisor
    return res


def rearrange(img):
    # return np.fft.fftshift(img, axes=(0,1))

    assert (img.ndim == 2)
    img_ = np.zeros(img.shape, img.dtype)
    xh, yh = img.shape[1] // 2, img.shape[0] // 2
    img_[0:yh, 0:xh], img_[yh:img.shape[0], xh:img.shape[1]] = img[yh:img.shape[0], xh:img.shape[1]], img[0:yh, 0:xh]
    img_[0:yh, xh:img.shape[1]], img_[yh:img.shape[0], 0:xh] = img[yh:img.shape[0], 0:xh], img[0:yh, xh:img.shape[1]]
    return img_

# recttools
def x2(rect):
    return rect[0] + rect[2]


def y2(rect):
    return rect[1] + rect[3]


def limit(rect, limit):
    if (rect[0] + rect[2] > limit[0] + limit[2]):
        rect[2] = limit[0] + limit[2] - rect[0]
    if (rect[1] + rect[3] > limit[1] + limit[3]):
        rect[3] = limit[1] + limit[3] - rect[1]
    if (rect[0] < limit[0]):
        rect[2] -= (limit[0] - rect[0])
        rect[0] = limit[0]
    if (rect[1] < limit[1]):
        rect[3] -= (limit[1] - rect[1])
        rect[1] = limit[1]
    if (rect[2] < 0):
        rect[2] = 0
    if (rect[3] < 0):
        rect[3] = 0
    return rect


def getBorder(original, limited):
    res = [0, 0, 0, 0]
    res[0] = limited[0] - original[0]
    res[1] = limited[1] - original[1]
    res[2] = x2(original) - x2(limited)
    res[3] = y2(original) - y2(limited)
    assert (np.all(np.array(res) >= 0))
    return res

def subwindow(img, window, borderType=cv2.BORDER_CONSTANT):
    cutWindow = [x for x in window]
    limit(cutWindow, [0, 0, img.shape[1], img.shape[0]])  # modify cutWindow
    assert (cutWindow[2] > 0 and cutWindow[3] > 0)
    border = getBorder(window, cutWindow)
    res = img[cutWindow[1]:cutWindow[1] + cutWindow[3], cutWindow[0]:cutWindow[0] + cutWindow[2]]

    if (border != [0, 0, 0, 0]):
        res = cv2.copyMakeBorder(res, border[1], border[3], border[0], border[2], borderType)
    return res


# KCF tracker
class KCFTracker:
    def __init__(self, hog=False, fixed_window=False, multiscale=True):
        self.lambdar = 0.0001  # regularization
        self.padding = 2.5  # extra area surrounding the target
        self.output_sigma_factor = 0.125  # bandwidth of gaussian target

        if (hog):  # HOG feature
            # VOT
            self.interp_factor = 0.075  # linear interpolation factor for adaptation
            self.sigma = 0.2  # gaussian kernel bandwidth
            # TPAMI   #interp_factor = 0.02   #sigma = 0.5
            self.cell_size = 4  # HOG cell size
            self._hogfeatures = True
        else:  # raw gray-scale image # aka CSK tracker
            self.interp_factor = 0.075
            self.sigma = 0.2
            self.cell_size = 1
            self._hogfeatures = False

        if (multiscale):
            self.template_size = 96  # template size
            self.scale_step = 1.1  # scale step for multi-scale estimation
            self.scale_weight = 0.96  # to downweight detection scores of other scales for added stability
            if(not fixed_window):
                fixed_window = True
        elif (fixed_window):
            self.template_size = 96
            self.scale_step = 1
        else:
            self.template_size = 1
            self.scale_step = 1

        self._tmpl_sz = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0., 0., 0., 0.]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self.size_patch = [0, 0, 0]  # [int,int,int]
        self._scale = 1.  # float
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._tmpl = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self.hann = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])

    #使用幅值做差来定位峰值的位置，返回的是需要改变的偏移量大小
    def subPixelPeak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)

    #初始化hanning窗，只执行一次，使用opencv函数做的
    def createHanningMats(self):
        hann2t, hann1t = np.ogrid[0:self.size_patch[0], 0:self.size_patch[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self.size_patch[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self.size_patch[0] - 1)))
        hann2d = hann2t * hann1t

        if (self._hogfeatures):
            hann1d = hann2d.reshape(self.size_patch[0] * self.size_patch[1])
            self.hann = np.zeros((self.size_patch[2], 1), np.float32) + hann1d      #相当于把1D汉宁窗复制成多个通道
        else:
            self.hann = hann2d
        self.hann = self.hann.astype(np.float32)

    #创建高斯峰函数，函数只在第一帧的时候执行
    def createGaussianPeak(self, sizey, sizex):                                      #构建响应图，只在初始化用
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return fftd(res)

    #使用带宽SIGMA计算高斯卷积核以用于所有图像x，y之间的相对位移
    def gaussianCorrelation(self, x1, x2):
        #HOG features
        if (self._hogfeatures):
            c = np.zeros((self.size_patch[0], self.size_patch[1]), np.float32)
            for i in xrange(self.size_patch[2]):
                x1aux = x1[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                x2aux = x2[i, :].reshape((self.size_patch[0], self.size_patch[1]))
                caux = cv2.mulSpectrums(fftd(x1aux), fftd(x2aux), 0, conjB=True)
                caux = real(fftd(caux, True))
                # caux = rearrange(caux)
                c += caux
            c = rearrange(c)
        #Gray features
        else:
            c = cv2.mulSpectrums(fftd(x1), fftd(x2), 0, conjB=True)  # 'conjB=' is necessary!在做乘法之前取第二个输入数组的共轭.
            c = fftd(c, True)
            c = real(c)
            c = rearrange(c)

        if (x1.ndim == 3 and x2.ndim == 3):
            d = (np.sum(x1[:, :, 0] * x1[:, :, 0]) + np.sum(x2[:, :, 0] * x2[:, :, 0]) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])
        elif (x1.ndim == 2 and x2.ndim == 2):
            d = (np.sum(x1 * x1) + np.sum(x2 * x2) - 2.0 * c) / (
                        self.size_patch[0] * self.size_patch[1] * self.size_patch[2])

        d = d * (d >= 0)
        d = np.exp(-d / (self.sigma * self.sigma))

        return d

    #从图像得到子窗口，通过赋值填充并检测特征
    def getFeatures(self, image, inithann, scale_adjust=1.0):
        extracted_roi = [0, 0, 0, 0]  # [int,int,int,int]
        cx = self._roi[0] + self._roi[2] / 2  # float         self._roi [229.28751704333027, 193.2002430105049, 167.0, 249.0]
        cy = self._roi[1] + self._roi[3] / 2  # float

        #初始化hanning窗，其实只执行一次，只在第一帧的时候inithann=1
        if (inithann):
            padded_w = self._roi[2] * self.padding
            padded_h = self._roi[3] * self.padding

            #按照长宽比例修改长宽大小，保证比较大的边为template_size大小
            if (self.template_size > 1):
                if (padded_w >= padded_h):
                    self._scale = padded_w / float(self.template_size)    #把最大的边缩小到96，_scale是缩小比例
                else:
                    self._scale = padded_h / float(self.template_size)    # _tmpl_sz是滤波模板的大小也是裁剪下的PATCH大小
                self._tmpl_sz[0] = int(padded_w / self._scale)
                self._tmpl_sz[1] = int(padded_h / self._scale)
            else:
                self._tmpl_sz[0] = int(padded_w)
                self._tmpl_sz[1] = int(padded_h)
                self._scale = 1.

            #设置_tmpl_sz的长宽：向上取原来长宽的最小2*cell_size倍，其中，较大边长为104
            if self._hogfeatures:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // (
                            2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // (
                            2 * self.cell_size) * 2 * self.cell_size + 2 * self.cell_size
            else:
                self._tmpl_sz[0] = int(self._tmpl_sz[0]) // 2 * 2
                self._tmpl_sz[1] = int(self._tmpl_sz[1]) // 2 * 2

        #检测区域大小
        extracted_roi[2] = int(scale_adjust * self._scale * self._tmpl_sz[0])        #选取从原图中扣下的图片位置大小
        extracted_roi[3] = int(scale_adjust * self._scale * self._tmpl_sz[1])
        #检测区域左上角坐标
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)

        #提取目标区域像素，超边界则做填充
        z = subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)                       #z是当前帧被裁剪下的搜索区域
        #按照比例缩小边界大小
        if (z.shape[1] != self._tmpl_sz[0] or z.shape[0] != self._tmpl_sz[1]):          #缩小到96
            z = cv2.resize(z, tuple(self._tmpl_sz))

        #提取HOG特征点
        if (self._hogfeatures):
            '''
            mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
            mapp = fhog.getFeatureMaps(z, self.cell_size, mapp)
            mapp = fhog.normalizeAndTruncate(mapp, 0.2)
            mapp = fhog.PCAFeatureMaps(mapp)
            self.size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))   #size_patch为列表，保存裁剪下来的特征图的【长，宽，通道】
            FeaturesMap = mapp['map'].reshape((self.size_patch[0] * self.size_patch[1],
                                               self.size_patch[2])).T  # (size_patch[2], size_patch[0]*size_patch[1])
            '''
            FeaturesMap = z
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5  # 从此FeatureMap从-0.5到0.5
            self.size_patch = [z.shape[0], z.shape[1], z.shape[2]]  # size_patch为列表，保存裁剪下来的特征图的【长，宽，1】
            FeaturesMap=FeaturesMap.reshape((self.size_patch[0] * self.size_patch[1],
                                               self.size_patch[2])).T
        else:                                                     #将RGB图变为单通道灰度图
            if (z.ndim == 3 and z.shape[2] == 3):
                FeaturesMap = cv2.cvtColor(z,cv2.COLOR_BGR2GRAY)  # z:(size_patch[0], size_patch[1], 3)  FeaturesMap:(size_patch[0], size_patch[1])   #np.int8  #0~255
            elif (z.ndim == 2):
                FeaturesMap = z  # (size_patch[0], size_patch[1]) #np.int8  #0~255
            FeaturesMap = FeaturesMap.astype(np.float32) / 255.0 - 0.5          #从此FeatureMap从-0.5到0.5
            self.size_patch = [z.shape[0], z.shape[1], 1]                       #size_patch为列表，保存裁剪下来的特征图的【长，宽，1】

        if (inithann):
            self.createHanningMats()  # createHanningMats need size_patch

        FeaturesMap = self.hann * FeaturesMap                                   #加汉宁（余弦）窗减少频谱泄露
        return FeaturesMap

    #z为前一帧样本，x为当前帧图像
    #p为中心的位移，pv为输出的峰值
    def detect(self, z, x):                                             #z是_tmpl即特征的平均，x是当前帧的特征
        #做变换得到计算结果res
        k = self.gaussianCorrelation(x, z)
        res = real(fftd(complexMultiplication(self._alphaf, fftd(k)), True))            #得到响应图

        #使用opencv的minMaxLoc来定位峰值坐标位置
        _, pv, _, pi = cv2.minMaxLoc(res)  # pv:float  pi:tuple of int                  #pv:响应最大值 pi:相应最大点的索引数组
        #子像素峰值检测，坐标是非整形的
        p = [float(pi[0]), float(pi[1])]  # cv::Point2f, [x,y]  #[float,float]          #得到响应最大的点索引的float表示

        if (pi[0] > 0 and pi[0] < res.shape[1] - 1):
            p[0] += self.subPixelPeak(res[pi[1], pi[0] - 1], pv, res[pi[1], pi[0] + 1])  #使用幅值做差来定位峰值的位置
        if (pi[1] > 0 and pi[1] < res.shape[0] - 1):
            p[1] += self.subPixelPeak(res[pi[1] - 1, pi[0]], pv, res[pi[1] + 1, pi[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.
                                                                                        #得出偏离采样中心的位移
        return p, pv                                                                    #返回偏离采样中心的位移和峰值

    def train(self, x, train_interp_factor):
        k = self.gaussianCorrelation(x, x)
        alphaf = complexDivision(self._prob, fftd(k) + self.lambdar)                    #alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
                                                                                        #_prob是初始化时的高斯响应图，相当于y
        self._tmpl = (1 - train_interp_factor) * self._tmpl + train_interp_factor * x           #_tmpl是截取的特征的加权平均
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf     #_alphaf是频域中相关滤波模板的加权平均

    #使用第一帧和它的跟踪框，初始化KCF跟踪器
    def init(self, roi, image):
        self._roi = list(map(float,roi))
        assert (roi[2] > 0 and roi[3] > 0)
        self._tmpl = self.getFeatures(image, 1)                                                 #_tmpl是截取的特征的加权平均
        self._prob = self.createGaussianPeak(self.size_patch[0], self.size_patch[1])            #_prob是初始化时的高斯响应图
        self._alphaf = np.zeros((self.size_patch[0], self.size_patch[1], 2), np.float32)        #_alphaf是频域中的相关滤波模板，有两个通道分别实部虚部
        self.train(self._tmpl, 1.0)
    #基于当前帧更新目标位置
    def update(self, image):
        # self._roi是跟踪框参数，x,y，长，宽,shape[0]是图片宽度，shape[1]是图片高度
        #修正边界
        # 如果跟踪框右边界小于零说明从左侧漂移出帧边界，向右调整，左侧边界调整为-self._roi[2] + 1
        # 如果跟踪框上边界小于零说明从下侧漂移出帧便捷，向上调整，下边界调整为-self._roi[3] + 1
        # 如果跟踪框大小已经贴近图片大小，则减小跟踪框大小。
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1                #修正边界
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 1
        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 2
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 2

        #跟踪框中心调整
        # x中心坐标修改为当前跟踪框中心坐标。
        cx = (self._roi[0] + self._roi[2] / 2)                                                 #尺度框中心
        cy = (self._roi[1] + self._roi[3] / 2)

        #尺度不变时检测峰值结果
        loc, peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0))

        #略大尺度和略小尺度进行检测
        # 多尺度估计时的尺度步长，说明应当调整框的大小。
        if (self.scale_step != 1):
            # Test at a smaller _scale
            # 重新检测峰值结果
            new_loc, new_peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, 1.0 / self.scale_step))
            # 使用1.06对当前峰值做增益，如果仍然大于之前的峰值，应该使用小号模板，调整框的大小。同时步长减小。
            if (1.06* new_peak_value > peak_value):
                #print("small")
                loc = new_loc
                peak_value = new_peak_value
                self._scale /= self.scale_step
                self._roi[2] /= self.scale_step
                self._roi[3] /= self.scale_step

            # Test at a bigger _scale
            new_loc, new_peak_value = self.detect(self._tmpl, self.getFeatures(image, 0, self.scale_step))
            # 使用0.96做减益,仍然大于之前的峰值,应该使用大号模板,调整框的大小，同时步长增加。
            if (self.scale_weight * new_peak_value > peak_value):
                #print("bigger")
                loc = new_loc
                peak_value = new_peak_value
                self._scale *= self.scale_step
                self._roi[2] *= self.scale_step
                self._roi[3] *= self.scale_step

        #因为返回的只有中心坐标，使用尺度和中心坐标调整坐标框，HOG元胞数组尺寸
        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self.cell_size * self._scale         #loc是中心相对移动量
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self.cell_size * self._scale

        # 再一次调整边界。
        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 1
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 1
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 2
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        #使用当前的检测框来训练样本参数
        x = self.getFeatures(image, 0, 1.0)
        self.train(x, self.interp_factor)
        #if peak_value>1.1 or peak_value < 0.75:
            #print(peak_value)
        #返回检测框
        return peak_value, self._roi

if __name__ == '__main__' :
    '''
    pathIn='./Inputframes/frame_000001.jpg'
pathOut='./Outputframes/frame_000001.jpg'
test=code.yolo_detect(pathIn,pathOut)
# print(str(test[0])+"/"+str(test[1])+"/"+str(test[2])+"/"+str(test[3]))
Box=[(test[0],test[1])]
Box.append(((test[0]+test[2]),test[1]+test[3]))
frameZoom=cv2.imread("./Inputframes/frame_000001.jpg")

cv2.rectangle(frameZoom, Box[0],Box[1], (0,0,205), 2)
tracker = KCF2.KCFTracker()
video = cv2.VideoCapture('test2.mp4')
ok, frame = video.read()

# bbox = cv2.selectROI(frame, False)
bbox=(test[0],test[1],test[2],test[3])
frames = 0
print(bbox)
#print(frame)
start = time.time()
tracker.init(bbox,frame)
'''



    # fps=20
    tracker = KCFTracker()
    video = cv2.VideoCapture('B:\Document\Code\\586\KCFTest\\testVideo\\areoplane_of_tinyandShelter.mp4')
    ok, frame = video.read()
    bbox = cv2.selectROI(frame, False)
    print(bbox)
    frames = 0
    #print(bbox)
    #print(frame)
    start = time.time()
    tracker.init(bbox,frame)
    while True:
        _,frame = video.read()
        if(_):
            # t = threading.Thread(target=loop())
            # t.start()
            # t.join()
            peak_value, item = tracker.update(frame)
            frames += 1
            time.sleep(0.01)
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))                    
            #print(item)
            x = int(item[0])
            y = int(item[1])
            w = int(item[2])
            h = int(item[3])

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.imshow("track",frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break