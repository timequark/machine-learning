import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
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
图片直方图
https://blog.csdn.net/asialee_bird/article/details/109526930

OpenCV里的imshow()和Matplotlib.pyplot的imshow()的区别
https://www.jb51.net/article/175041.htm
'''

def test_hist(name, img_path):
    img_bgr = cv2.imread(img_path, 1) # OpenCV里彩色图片加载时是按照BGR的顺序
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
    img = cv2.imread(img_path, 0) #0表示灰度图
    
    # 创建掩膜mask
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[50:200, 50:150] = 255
    masked_img = cv2.bitwise_and(img, img, mask=mask)#利用掩膜（mask）进行“与”操作
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full,'red'), plt.plot(hist_mask,'black')
    plt.xlim([0, 256])
    plt.show()

if __name__ == '__main__':
    imgfile = 'C:/Users/Administrator/Pictures/00.jpg'
    test_hist('', imgfile)
    # test_mask(imgfile)
    logging.info('over')
