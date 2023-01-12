#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 13:58:18 2021

@author: gilbertec.fleurisma
"""

# For more information: https://scikit-learn.org/stable/modules/tree.html
import pandas as pd
import numpy as np

#from numpy.linalg import inv
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import model_selection
#from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Ridge
#from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

# Read a feature space
input_data = pd.read_csv("/Users/gilbertec.fleurisma/Downloads/CSC-410-assignment01/Data/merged012.csv",header=None)

# Label/Response set
y = input_data[64]

# Drop the labels and store the features
input_data.drop(64,axis=1,inplace=True)
X = input_data

# Generate feature matrix using a Numpy array
tmp = np.array(X)
X1 = tmp[:,0:40]


# Generate label matrix using Numpy array
Y1 = np.array(y)

# Machine learning with 80:20

# Split the data into 80:20
row, col = X.shape

TR = round(row*0.8)
TT = row-TR

# Training with 80% data
X1_train = X1[0:TR-1,:]
Y1_train = Y1[0:TR-1]

clf = Lasso(alpha=1.0)
#clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(X1_train, Y1_train)

# Plot the tree
# tree.plot_tree(clf) 



# Testing with 20% data
X1_test = X1[TR:row,:]
y_test = y[TR:row]

yhat_test = clf.predict(X1_test)


# Confusion matrix analytics
CC_test = confusion_matrix(y_test, yhat_test)

TN = CC_test[0,0]
FP = CC_test[0,1]
FN = CC_test[1,0]
TP = CC_test[1,1]

FPFN = FP+FN
TPTN = TP+TN

Accuracy = 1/(1+(FPFN/TPTN))
print("Our_Accuracy_Score:",Accuracy)

Precision = 1/(1+(FP/TP))
print("Our_Precision_Score:",Precision)

Sensitivity = 1/(1+(FN/TP))
print("Our_Sensitivity_Score:",Sensitivity)

Specificity = 1/(1+(FP/TN))
print("Our_Specificity_Score:",Specificity)


# Built-in accuracy measure
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score



print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
print("BuiltIn_Precision:",metrics.precision_score(y_test, yhat_test, average='micro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test,  average='macro'))

#conda install python-graphviz
# Install graphviz at Anaconda promot --> conda install python-graphviz
import graphviz 
via_fruit = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(via_fruit) 
graph.render("C:/Users/s_suthah/Desktop/Output410/fruits5") 



import math

-(7/12)*math.log(7/12, 2) - (5/12)*math.log(5/12, 2)

-(4/10)*math.log(4/10, 2) - (6/10)*math.log(6/10, 2)