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

'''
https://scikit-learn.org/stable/modules/neural_networks_supervised.html
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
        
        self.X = [[0., 0.], [1., 1.], [0.6, 1], [-0.5, 1.2]]
        self.y = [0, 1, 0, 2]
        '''
        solver: 求最优权重的方法
            - ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
            - ‘sgd’ refers to stochastic gradient descent.
            - ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
        '''
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='tanh',
                            hidden_layer_sizes=(5, 2), random_state=1)

        clf.fit(self.X, self.y)
        '''
        predict_proba 含义
        https://blog.csdn.net/anan15151529/article/details/102632463
        '''
        result_p = clf.predict_proba([[0.5, 1.6], [-1, 0]])
        result = clf.predict([[0.5, 1.6], [-1, 0]])
        
        logging.info('')
    
    def teardown(self):
        logging.info('teardown ...')
        pass

if __name__ == '__main__':
    sample = SVMSample()
    sample.setup(option={})
    
    logging.info('over')
