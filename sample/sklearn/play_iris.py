import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap

'''
https://zhuanlan.zhihu.com/p/31785188

sklearn
https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html#sphx-glr-auto-examples-datasets-plot-iris-dataset-py

matplotlib API
https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html
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

    def teardown(self):
        logging.info('teardown ...')
        pass
    
    def show2D_Sepal(self):
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

        ax.set_title("")
        # Plot the training points
        self.xserials = self.X[:, 0]
        self.yserials = self.X[:, 1]
        ax.scatter(self.xserials, self.yserials, c=self.y, cmap=self.cm_bright)
        ax.set_xticks(range(int(self.xserials.min() - 1), int(self.xserials.max() + 1)))
        ax.set_yticks(range(int(self.yserials.min() - 1), int(self.yserials.max() + 1)))
        plt.xlabel("Sepal length")
        plt.ylabel("Sepal width")
        plt.tight_layout()
        plt.show()
    
    def show2D_Petal(self):
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

        ax.set_title("")
        # Plot the training points
        self.xserials = self.X[:, 2]
        self.yserials = self.X[:, 3]
        ax.scatter(self.xserials, self.yserials, c=self.y, cmap=self.cm_bright)
        ax.set_xticks(range(int(self.xserials.min() - 1), int(self.xserials.max() + 1)))
        ax.set_yticks(range(int(self.yserials.min() - 1), int(self.yserials.max() + 1)))
        plt.xlabel("Petal length")
        plt.ylabel("Petal width")
        plt.tight_layout()
        plt.show()
    
    def show3D_(self):
        # import some data to play with
        iris = datasets.load_iris()
        X = iris.data[:, :2]  # we only take the first two features.
        y = iris.target

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

        fig = plt.figure(figsize=(8, 16))
        ax1 = fig.add_subplot(211, xlim=(x_min, x_max), ylim=(y_min, y_max))

        # Plot the training points
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
        ax1.set_xlabel("Sepal length")
        ax1.set_ylabel("Sepal width")

        # ax1.xlim(x_min, x_max)
        # ax1.ylim(y_min, y_max)
        # ax1.xticks(())
        # ax1.yticks(())

        # To getter a better understanding of interaction of the dimensions
        # plot the first three PCA dimensions
        # fig = plt.figure(1, figsize=(8, 6))
        ax2 = fig.add_subplot(212, projection="3d")

        X_reduced = decomposition.PCA(n_components=3).fit_transform(iris.data)
        ax2.scatter(
            X_reduced[:, 0],
            X_reduced[:, 1],
            X_reduced[:, 2],
            c=y,
            cmap=plt.cm.Set1,
            edgecolor="k",
            s=40,
        )
        ax2.set_title("First three PCA directions")
        ax2.set_xlabel("1st eigenvector")
        ax2.w_xaxis.set_ticklabels([])
        ax2.set_ylabel("2nd eigenvector")
        ax2.w_yaxis.set_ticklabels([])
        ax2.set_zlabel("3rd eigenvector")
        ax2.w_zaxis.set_ticklabels([])
        
        ax2.set_xlim(X_reduced[:, 0].min()-0.5, X_reduced[:, 0].max()+0.5)
        ax2.set_ylim(X_reduced[:, 1].min()-0.5, X_reduced[:, 1].max()+0.5)
        ax2.set_zlim(X_reduced[:, 2].min()-0.5, X_reduced[:, 2].max()+0.5)

        plt.show()
    
if __name__ == '__main__':
    sample = Sample()
    sample.setup(option={})
    # sample.show2D_Sepal()
    # sample.show2D_Petal()
    sample.show3D_()
    
    logging.info('over')
