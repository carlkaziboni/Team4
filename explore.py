"""
This is where all the models will be stored. They will run at the beginning of the flask app
"""

import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
import csv
from sklearn.preprocessing import LabelEncoder