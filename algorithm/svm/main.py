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
https://www.cnblogs.com/pinard/p/6126077.html
'''

__all__ = ['SVMSample']

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

logging.info(matplotlib.get_backend())

class SVMSample:
    
    def __init__(self):
        self.X = None
        self.y = None
    
    def setup(self, option: dict):
        logging.info('setup ...')
        
        # matplotlib.use('TkAgg')
        
        self.X, self.y = make_circles(n_samples=(100, 50), noise=0.2, factor=0.5, random_state=1)
        self.X = StandardScaler().fit_transform(self.X)
        
        cm = plt.cm.RdBu
        self.cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot()

        ax.set_title("Input data")
        # Plot the training points
        self.xserials = self.X[:, 0]
        self.yserials = self.X[:, 1]
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
                clf = SVC(C=C, gamma=gamma)
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

                # Plot also the training points
                plt.scatter(self.xserials, self.yserials, c=self.y, cmap=self.cm_bright)

                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())
                plt.xticks(())
                plt.yticks(())
                plt.xlabel(" gamma=" + str(gamma) + " C=" + str(C))
                plt.show()
                
                logging.info('')
    
if __name__ == '__main__':
    sample = SVMSample()
    sample.setup(option={})
    # sample.lookSVCBest()
    sample.lookSVCs()
    
    logging.info('over')
