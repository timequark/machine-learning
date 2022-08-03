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

'''
https://zhuanlan.zhihu.com/p/31785188
'''

__all__ = ['Sample']

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

logging.info(matplotlib.get_backend())

class Sample:
    
    def __init__(self):
        self.X = None
        self.y = None
    
    def setup(self, option: dict):
        logging.info('setup ...')
        
        # matplotlib.use('TkAgg')
        
        iris = datasets.load_iris()
        features = iris.data
        target = iris.target
        logging.info('{} {}'.format(features.shape,target.shape))
        logging.info('{}'.format(iris.feature_names))
        
        self.X, self.y = features, target
        # self.X = StandardScaler().fit_transform(self.X)
        
        cm = plt.cm.RdBu
        self.cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
        ax = plt.subplot()

        ax.set_title("Input data")
        # Plot the training points
        self.xserials = self.X[:, 2]
        self.yserials = self.X[:, 3]
        ax.scatter(self.xserials, self.yserials, c=self.y, cmap=self.cm_bright)
        ax.set_xticks(range(int(self.xserials.min() - 1), int(self.xserials.max() + 1)))
        ax.set_yticks(range(int(self.yserials.min() - 1), int(self.yserials.max() + 1)))
        plt.tight_layout()
        plt.show()

    def teardown(self):
        logging.info('teardown ...')
        pass
    
    def lookSVCBest(self):
        grid = GridSearchCV(SVC(), param_grid={"C":[0.1, 1, 10], "gamma": [1, 0.1, 0.01]}, cv=4)
        grid.fit(self.X, self.y)
        logging.info("The best parameters are {} with a score of {:0.2f}".format(grid.best_params_, grid.best_score_))
    
    def lookSVCs(self):
        x_min, x_max = self.xserials.min() - 1, self.xserials.max() + 1
        y_min, y_max = self.yserials.min() - 1, self.yserials.max() + 1
        
        '''
        meshgrid 从坐标向量中返回坐标矩阵 [https://blog.csdn.net/littlehaes/article/details/83543459]
        '''
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        
        for i, C in enumerate((0.1, 1, 10)):
            for j, gamma in enumerate((1, 0.1, 0.01)):
                plt.subplot()
                '''
                kernel
                常见的几中核函数 [https://blog.csdn.net/qq_37007384/article/details/88418256]
                    - rbf [https://blog.csdn.net/qq_36264495/article/details/88316482]
                SVC 参数
                https://blog.csdn.net/qq_37007384/article/details/88410998
                
                高斯核
                有人说，高斯核函数的本质是将每一个样本点映射到一个无穷维的特征空间
                http://t.zoukankan.com/jokingremarks-p-14337248.html
                https://zhuanlan.zhihu.com/p/258047472
                '''
                clf = SVC(C=C, gamma=gamma, kernel='rbf')
                clf.fit(self.X, self.y)
                '''
                ravel 降为一维 [https://blog.csdn.net/hanshuobest/article/details/78882425]
                np.c_ 坐标点的X值序列、Y值序列 转换成 坐标点二维矩阵
                
                关于predict, predict_proba [https://www.pythonheidong.com/blog/article/878018/cb35f513c9158417fa57/]
                '''
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

                '''
                reshape 一维转多维 [https://numpy.org/doc/stable/reference/generated/numpy.reshape.html]
                reshaper后的 Z 为xx.shape维的Z值
                '''
                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                
                '''
                (x, y) ->(z) , z 为预测值
                类似于画出等高线 [https://blog.csdn.net/lens___/article/details/83960810]
                '''
                plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

                '''
                pyplot.scatter
                https://blog.csdn.net/yuan_xiangjun/article/details/119514811
                '''
                # Plot also the training points
                plt.scatter(self.xserials, self.yserials, c=self.y, cmap=self.cm_bright, marker='x')

                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.xticks(())
                plt.yticks(())
                plt.xlabel(" gamma=" + str(gamma) + " C=" + str(C))
                plt.show()
                
                logging.info('')
    
if __name__ == '__main__':
    sample = Sample()
    sample.setup(option={})
    # sample.lookSVCBest()
    sample.lookSVCs()
    
    logging.info('over')
