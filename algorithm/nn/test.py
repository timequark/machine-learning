import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.utils.np_utils import to_categorical
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
import cv2  
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

'''
【直方图】
直方图是对图像的中的像素点的值进行统计，一般情况下直方图都是灰度图像，
直方图x轴是灰度值（一般0~255），y轴就是图像中每一个灰度级对应的像素点的个数，
即横坐标表示图像中各个像素点的灰度级，纵坐标表示具有该灰度级的像素个数。

OpenCV实现傅里叶变换与傅里叶逆变换
- 高通滤波器将提取图像的边缘轮廓
- 低通滤波器是指通过低频的滤波器，衰减高频而通过低频，常用于模糊图像。
    低通滤波器与高通滤波器相反，当一个像素与周围像素的插值小于一个特定值时，平滑该像素的亮度，常用于去燥和模糊化处理。
    如PS软件中的高斯模糊，就是常见的模糊滤波器之一，属于削弱高频信号的低通滤波器。低通滤波器中心区域为白色255，其他区域为黑色0。

图片直方图
https://blog.csdn.net/asialee_bird/article/details/109526930

OpenCV里的imshow()和Matplotlib.pyplot的imshow()的区别
https://www.jb51.net/article/175041.htm

Opencv中的cv2.calcHist()函数的作用及返回值
https://blog.csdn.net/star_sky_sc/article/details/122371392
'''

def test_hist(name, img_path):
    '''
    图片直方图
    '''
    logging.info('cv2.IMREAD_COLOR = {}'.format(cv2.IMREAD_COLOR))
    logging.info('cv2.IMREAD_GRAYSCALE = {}'.format(cv2.IMREAD_GRAYSCALE))
    logging.info('cv2.IMREAD_UNCHANGED = {}'.format(cv2.IMREAD_UNCHANGED))
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR) # OpenCV里彩色图片加载时是按照BGR的顺序
    cv2.imshow(name,img_bgr)
    b,g,r=cv2.split(img_bgr)   # 通道的拆分
    img_rgb=cv2.merge((r,g,b)) # 通道的融合
    plt.subplot(211), plt.imshow(img_rgb)
    plt.subplot(212), plt.hist(img_bgr.ravel(), 256)
    
    # cv2.imshow(name,img) 
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    
    plt.show()
    
def test_mask(img_path):
    '''
    图片带掩膜遮罩
    '''
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #0表示灰度图
    
    # 创建掩膜mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[50:200, 50:150] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask) # 利用掩膜（mask）进行“与”操作
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full,'red'), plt.plot(hist_mask,'black')
    plt.xlim([0, 256])
    plt.show()

def test_image_equalize(name, img_path):
    '''
    图片直方图均衡化
    '''
    img_bgr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)   # 0表示灰度图
    equ = cv2.equalizeHist(img_bgr)     # 将要均衡化的原图像【要求是灰度图像】作为参数传入，则返回值即为均衡化后的图像。
    plt.hist(equ.ravel(),256)
    plt.show()
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #自适应直方图均衡化
    res_clahe = clahe.apply(img_bgr)
    res = np.hstack((img_bgr, equ, res_clahe))
    cv2.imshow('',res)
    
    plt.show()

def test_tensor(name, img_path):
    X = []
    y_label = []
    
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR) # OpenCV里彩色图片加载时是按照BGR的顺序
    X.append(np.array(img_bgr))
    y_label.append('girl')
    y_label.append('boy')
    y_label.append('cat')
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_label) # 标签编码
    y = to_categorical(y) # 将标签转换为One-hot编码
    X = np.array(X) # 将X从列表转换为张量数组
    X = X/255 # 将X张量归一化
    
    logging.info('')
    
if __name__ == '__main__':
    imgfile = 'C:/Users/Administrator/Pictures/00.jpg'
    # test_hist('', imgfile)
    # test_mask(imgfile)
    # test_image_equalize('', imgfile)
    test_tensor('', imgfile)
    logging.info('over')
