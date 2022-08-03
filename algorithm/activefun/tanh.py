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
由于其图形在定义域0附近近似线性，并且在整个定义域有可导性，该函数广泛用于深度学习领域的神经网络中作为神经元的激活函数使用

https://baike.baidu.com/item/%E5%8F%8C%E6%9B%B2%E6%AD%A3%E5%88%87/3194837?fr=aladdin
'''

__all__ = ['Func']

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

class Func:
    
    def __init__(self):
        self.X = None
        self.y = None
    
    def setup(self, option: dict):
        logging.info('setup ...')
        
        # matplotlib.use('TkAgg')

    def teardown(self):
        logging.info('teardown ...')
        pass
    
    def tanh(self, x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
    def look(self):
        sigmoid_inputs = np.arange(-10,10,0.1)
        sigmoid_outputs = self.tanh(sigmoid_inputs)
        print("Tanh Function Input :: {}".format(sigmoid_inputs))
        print("Tanh Function Output :: {}".format(sigmoid_outputs))
        
        plt.plot(sigmoid_inputs,sigmoid_outputs)
        plt.xlabel("Tanh Inputs")
        plt.ylabel("Tanh Outputs")
        plt.show()

if __name__ == '__main__':
    sample = Func()
    sample.setup(option={})
    sample.look()
    sample.teardown()
    
    logging.info('over')
