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
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier
import cv2


'''
[1] https://zhuanlan.zhihu.com/p/28035475

[2] https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing

[3]关于数据标准化、归一化
[ https://blog.csdn.net/algorithmPro/article/details/103045824 ]
enc = ['win','draw','lose','win']	
dec = ['draw','draw','win']

from sklearn.preprocessing import LabelEncoder	
LE = LabelEncoder()		
print( LE.fit(enc) )	
print( LE.classes_ )	
print( LE.transform(dec) )

输出：
    LabelEncoder()
    ['draw' 'lose' 'win']
    [0 0 2]
上面这种编码的问题是，机器学习算法会认为两个临近的值比两个疏远的值要更相似。显然这样不对 (比如，0 和 1 比 0 和 2 距离更近，难道 draw 和 win 比 draw 和 lose更相似？)
要解决这个问题，一个常见的方法是给每个分类创建一个二元属性，即独热编码 (one-hot encoding)
OneHotEncoder
独热编码其实就是把一个整数用向量的形式表现。

[4] http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130


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
    from sklearn.preprocessing import StandardScaler
    scal = StandardScaler()
    X_train_rows = scal.fit_transform(X_train_rows)
    X_test_rows = scal.fit_transform(X_test_rows)
    
    # k 值选择
    validation_accuracies = []
    
    logging.info('training...')
    clf = MLPClassifier(solver='adam', activation='logistic', learning_rate='adaptive', hidden_layer_sizes=(10, 5), random_state=1).fit(X_train_rows, y_train)
    predictions_labels = clf.predict(X_test_rows)
    
    print('预测结果:')
    print(predictions_labels)
    
    print('算法评价:')
    print((classification_report(y_test, predictions_labels)))
    
    print(validation_accuracies)
    
    logging.info('')

if __name__ == '__main__':
    test('/data-warehouse/machine-learning-dataset/cifar-10-python/cifar-10-batches-py')
    logging.info('over')
