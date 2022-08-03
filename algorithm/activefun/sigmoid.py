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
Sigmoid函数是一个在生物学中常见的S型函数，也称为S型生长曲线。
在信息科学中，由于其单增以及反函数单增等性质，Sigmoid函数常被用作神经网络的激活函数，将变量映射到0,1之间

https://baike.baidu.com/item/Sigmoid%E5%87%BD%E6%95%B0/7981407?fr=aladdin
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
    
    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))
    
    def look(self):
        sigmoid_inputs = np.arange(-10,10,0.1)
        sigmoid_outputs = self.sigmoid(sigmoid_inputs)
        print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
        print("Sigmoid Function Output :: {}".format(sigmoid_outputs))
        
        plt.plot(sigmoid_inputs,sigmoid_outputs)
        plt.xlabel("Sigmoid Inputs")
        plt.ylabel("Sigmoid Outputs")
        plt.show()

if __name__ == '__main__':
    sample = Func()
    sample.setup(option={})
    sample.look()
    sample.teardown()
    
    logging.info('over')
