import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# from sklearnex import patch_sklearn, config_context, get_config
# patch_sklearn()

from sklearn import datasets, svm
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import cv2


'''
CIFAR-10
https://blog.csdn.net/qq_52309640/article/details/120941410

numpy.reshape
https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy-reshape
https://blog.csdn.net/ch1209498273/article/details/78966192

KNeighborsClassifier
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.predict

'''

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

logging.info(matplotlib.get_backend())

def test(cifar10Dir):
    # STEP 1: 切分训练集、测试集
    imgPathsTrain = []
    y_train = []
    imgPathsTest = []
    y_test = []
    clsfiDirs = list(map(lambda x: os.path.join(cifar10Dir, x), os.listdir(cifar10Dir)))
    clsfiDirs.sort(key=lambda x:x, reverse=False)
    for clsfi in clsfiDirs:
        imgfiles = list(map(lambda x: os.path.join(clsfi, x), os.listdir(clsfi)))
        imgfiles.sort(key=lambda x:x, reverse=False)
        testidx = 0
        for imgfile in imgfiles:
            # 每个分类前200张做为测试集
            if testidx < 100:
                imgPathsTest.append(imgfile)
                y_test.append(int(os.path.basename(os.path.dirname(imgfile)).split('-')[0]))
            elif testidx < 1000:
                imgPathsTrain.append(imgfile)
                y_train.append(int(os.path.basename(os.path.dirname(imgfile)).split('-')[0]))
            testidx += 1
    
    # STEP 2: 图片加载、直方图处理
    X_train = []
    X_test  = []
    
    # 训练集
    for i in imgPathsTrain:
        logging.info('load {} ...'.format(i))
        # image = cv2.imread(i)
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        
        # 图像像素大小一致
        # img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
 
        # 计算图像直方图并存储至X数组
        # channels : [0, 1, 2], 表示 G/B/R 3通道
        # hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0.0, 255.0, 0.0, 255.0, 0.0, 255.0])
        # hist = hist.astype(np.float16)
        hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
        
        X_train.append(((hist/255).flatten()))
    
    # 测试集
    for i in imgPathsTest:
        logging.info('load {} ...'.format(i))
        # image = cv2.imread(i)
        image = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        
        # 图像像素大小一致
        # img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
 
        # 计算图像直方图并存储至X数组
        # channels : [0, 1, 2], 表示 G/B/R 3通道
        # hist = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0.0, 255.0, 0.0, 255.0, 0.0, 255.0])
        # hist = hist.astype(np.float16)
        hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
        
        X_test.append(((hist/255).flatten()))
    
    # STEP 3: kNN
    # k 值选择
    validation_accuracies = []
    for k in [1, 3, 5, 10, 20, 50]:
        
        print('-------------')
        print('k: {}'.format(k))
        print('-------------')
        clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        predictions_labels = clf.predict(X_test)
        
        print('预测结果:')
        print(predictions_labels)
        
        print('算法评价:')
        print((classification_report(y_test, predictions_labels)))
        
        # keep track of what works on the validation set
        validation_accuracies.append((k, clf.score(X_test, y_test)))
    
    print(validation_accuracies)
        
        
    
    logging.info('')

if __name__ == '__main__':
    # import dpctl
    # conf = get_config()
    # with config_context(target_offload="gpu:0"):
    test('/data-warehouse/machine-learning-dataset/cifar-10/train')
    logging.info('over')
