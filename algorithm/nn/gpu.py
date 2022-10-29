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
import cv2  
import numpy as np
import matplotlib.pyplot as plt
import psutil
from pyadl import *

devices = ADLManager.getInstance().getDevices()

print('')

