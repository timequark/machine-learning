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
import numpy as np
import cv2

'''
CIFAR-10 数据集介绍
https://blog.csdn.net/weixin_45084253/article/details/123597716
http://www.cs.utoronto.ca/~kriz/cifar.html

制作数据集到文件jpg
https://blog.csdn.net/qq_40755283/article/details/125209463
'''

def unpickle(file):#打开cifar-10文件的其中一个batch（一共5个batch）
    import pickle
    with open(os.path.join('F:/data-warehouse/machine-learning-dataset/cifar-10-python/cifar-10-batches-py', file), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_batch=unpickle("data_batch_1")#打开cifar-10文件的data_batch_1
cifar_data=data_batch[b'data']#这里每个字典键的前面都要加上b
cifar_label=data_batch[b'labels']
cifar_data=np.array(cifar_data)#把字典的值转成array格式，方便操作
print(cifar_data.shape)#(10000,3072), 3072 = 32 * 32 * 3
cifar_label=np.array(cifar_label)
print(cifar_label.shape)#(10000,)

label_name=['airplane','automobile','brid','cat','deer','dog','frog','horse','ship','truck']

def imwrite_images(k):#k的值可以选择1-10000范围内的值
    lableImgCount = {}
    for lb in label_name:
        lableImgCount[lb] = 0
    for i in range(k):
        image=cifar_data[i]
        image=image.reshape(-1,1024)
        r=image[0,:].reshape(32,32)#红色分量
        g=image[1,:].reshape(32,32)#绿色分量
        b=image[2,:].reshape(32,32)#蓝色分量
        img=np.zeros((32,32,3))
        #RGB还原成彩色图像
        img[:,:,0]=r
        img[:,:,1]=g
        img[:,:,2]=b
        lbIdx = cifar_label[i]
        lb = label_name[lbIdx]
        lableImgCount[lb] += 1
        fpath = 'F:/data-warehouse/machine-learning-dataset/cifar-10/train/{}-{}/{:0>6}.jpg'.format(lbIdx, lb, lableImgCount[lb])
        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        cv2.imwrite(fpath, img)

if __name__ == '__main__':
    imwrite_images(10000)
    logging.info('over')
