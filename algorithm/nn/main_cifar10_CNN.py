import os
import logging
import pickle
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
from sklearn.neural_network import MLPClassifier
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models
from keras import backend as K
from matplotlib.colors import ListedColormap
import cv2


'''
卷积神经网络
https://www.kancloud.cn/apachecn/hands-on-ml-zh/1948557

Keras
https://keras.io/
https://keras.io/zh/why-use-keras/
https://keras.io/zh/backend/
https://keras.io/api/layers/convolution_layers/convolution2d/
https://keras.io/getting_started/intro_to_keras_for_engineers/

CIFAR-10 CNN-Capsule
https://keras.io/zh/examples/cifar10_cnn_capsule/


[1] https://blog.csdn.net/weixin_44026026/article/details/119575988

[2] https://blog.csdn.net/zzZ_CMing/article/details/79691459

'''

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

logging.info(matplotlib.get_backend())

def load_cfar10_batch(cifar10Dir, batchfile, gray:bool = False):
    with open(os.path.join(cifar10Dir, batchfile), 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    features = batch['data'].reshape( (len(batch['data']), 3, 32, 32) ).transpose(0, 2, 3, 1)
    if gray:
        features = features.max(axis=3) # 灰度处理，https://blog.csdn.net/qq_41915623/article/details/124547004
    labels = batch['labels']
    
    return features, labels

def test(cifar10Dir):
    gray = False
    X_train, y_train = load_cfar10_batch(cifar10Dir, 'data_batch_1', gray)
    features, labels = load_cfar10_batch(cifar10Dir, 'data_batch_2', gray)
    X_train, y_train = np.concatenate([X_train, features]), np.concatenate([y_train, labels])
    features, labels = load_cfar10_batch(cifar10Dir, 'data_batch_3', gray)
    X_train, y_train = np.concatenate([X_train, features]), np.concatenate([y_train, labels])
    features, labels = load_cfar10_batch(cifar10Dir, 'data_batch_4', gray)
    X_train, y_train = np.concatenate([X_train, features]), np.concatenate([y_train, labels])
    features, labels = load_cfar10_batch(cifar10Dir, 'data_batch_5', gray)
    X_train, y_train = np.concatenate([X_train, features]), np.concatenate([y_train, labels])
    
    X_test, y_test   = load_cfar10_batch(cifar10Dir, 'test_batch', gray)
    
    # 减负
    # TRAIN_COUNT = 4000
    # TEST_COUNT  = 100
    # X_train = X_train[:TRAIN_COUNT, :, :, :]
    # y_train = y_train[:TRAIN_COUNT]
    # X_test = X_test[:TEST_COUNT, :, :, :]
    # y_test = y_test[:TEST_COUNT]
    
    # 查看图片
    # fig, axes = plt.subplots(nrows=3, ncols=20, sharex=True, sharey=True, figsize=(80, 12))
    # imgs = X_train[:60]
    # for image, row in zip([imgs[:20], imgs[20:40], imgs[40:60]], axes):
    #     for img, ax in zip(image, row):
    #         if gray:
    #             ax.imshow(img, 'gray')
    #         else:
    #             ax.imshow(img)
    #         ax.get_xaxis().set_visible(False)
    #         ax.get_yaxis().set_visible(False)
    # fig.tight_layout(pad=0.1)
    # plt.show()
    
    # 重塑
    if gray:
        X_train_rows = X_train.reshape(X_train.shape[0], 32*32)
        X_test_rows  = X_test.reshape(X_test.shape[0], 32*32)
    else:
        X_train_rows = X_train.reshape(X_train.shape[0], 32*32*3)
        X_test_rows  = X_test.reshape(X_test.shape[0], 32*32*3)
    
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    scal = MinMaxScaler()
    X_train_rows = scal.fit_transform(X_train_rows)
    X_test_rows = scal.fit_transform(X_test_rows)
    X_train_m = X_train_rows.reshape( X_train_rows.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    X_test_m = X_test_rows.reshape( X_test_rows.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    y_train_m = to_categorical(y_train, 10) # 将标签转换为One-hot编码
    y_test_m = to_categorical(y_test, 10) # 将标签转换为One-hot编码
    
    logging.info('data_format -> {}'.format(K.image_data_format()))
    
    cnn = models.Sequential() # 贯序模型
    cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', # 卷积
                            input_shape=X_train_m.shape[1:])) 
    cnn.add(layers.MaxPooling2D((2, 2))) # 最大池化
    cnn.add(layers.Dropout(0.25))
    cnn.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same')) # 卷积
    cnn.add(layers.MaxPooling2D((2, 2))) # 最大池化
    cnn.add(layers.Dropout(0.25))
    # cnn.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same')) # 卷积
    # cnn.add(layers.MaxPooling2D((2, 2))) # 最大池化
    # cnn.add(layers.Dropout(0.25))
    # cnn.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same')) # 卷积
    # cnn.add(layers.MaxPooling2D((2, 2))) # 最大池化
    # cnn.add(layers.Dropout(0.25))
    cnn.add(layers.Flatten()) # 展平
    K.random.seed(0)
    cnn.add(layers.Dropout(0.5)) # Dropout
    cnn.add(layers.Dense(128, activation='relu')) # 全连接
    cnn.add(layers.Dense(10, activation='softmax')) # 分类输出
    cnn.compile(loss='categorical_crossentropy', # 损失函数
                optimizer='RMSprop', # 优化器
                metrics=['acc']) # 评估指标

    history = cnn.fit(X_train_m, y_train_m, # 指定训练集
                    epochs=50,     # 指定轮次
                    batch_size=32, # 指定批量大小
                    validation_data=(X_test_m, y_test_m)) # 指定验证集

    # show_history(history)
    print(history.history['val_acc'])
    logging.info('')

def show_history(history): # 显示训练过程中的学习曲线
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test('/data-warehouse/machine-learning-dataset/cifar-10-python/cifar-10-batches-py')
    logging.info('over')
