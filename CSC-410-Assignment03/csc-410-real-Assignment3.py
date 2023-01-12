# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 19:45:55 2021

@author: Christina
"""

import pandas as pd
import numpy as np

#from numpy.linalg import inv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso

# Read a feature space
input_data = pd.read_csv("/Users/gilbertec.fleurisma/merged012.csv",header=0)

input_data2 = pd.read_csv("/Users/gilbertec.fleurisma/merged01.csv",header=None)

# Label/Response set
y = input_data['64']

# Drop the labels and store the features
input_data.drop('64',axis=1,inplace=True)
X = input_data


##############################################################################
# Task 1 #####################################################################
##############################################################################


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



first = X_train.iloc[:, 15]
firsttst = X_test.iloc[:, 15]

scnd = X_train.iloc[:, 30]
scndtst = X_test.iloc[:, 30]



# Creating histogram

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(first,bins = [0, 50, 100, 150, 200, 250, 300])

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(firsttst,bins = [0, 50, 100, 150, 200, 250, 300])



fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(scnd,bins = [0, 50, 100, 150, 200, 250, 300])

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(scndtst,bins = [0, 50, 100, 150, 200, 250, 300])

plt.scatter(first,scnd,c=['#1f77b1'])
plt.scatter(firsttst,scndtst,c = ['#ff7f0e'])




##############################################################################
########################## Task 2 ############################################
##############################################################################

from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(penalty='l1',solver='liblinear')
clf_LR.fit(X_train,y_train)
y_test_pred = clf_LR.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
CC_test = confusion_matrix(y_test, y_test_pred)
accuracy_score(y_test, y_test_pred)



X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
xx = pd.Series(y_test_pred, name='65')
df = pd.concat([X_test,y_test,xx], axis=1)

CC_test = np.array(pd.crosstab(index=df['64'], columns=df['65']))

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

from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score



print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("BuiltIn_Precision:",metrics.precision_score(y_test, y_test_pred, average='micro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, y_test_pred,  average='macro'))



##############################################################################
# Task 3 #####################################################################
##############################################################################


from sklearn.ensemble import RandomForestClassifier
rF = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
model = rF.fit(X_train,y_train)
y_test_pred = model.predict(X_test)


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


print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("BuiltIn_Precision:",metrics.precision_score(y_test, y_test_pred, average='micro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, y_test_pred,  average='macro'))



##############################################################################
## Task 4 ####################################################################
##############################################################################


from sklearn import metrics

print(metrics.classification_report(y_test, y_test_pred))






##############################################################################
##############################################################################
############## For two class dataset #########################################
##############################################################################
##############################################################################



input_data = pd.read_csv("/Users/gilbertec.fleurisma/merged01.csv")

# Label/Response set
y = input_data['64']

# Drop the labels and store the features
input_data.drop('64',axis=1,inplace=True)
X = input_data


##############################################################################
# Task 1 #####################################################################
##############################################################################


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)





##############################################################################
########################## Task 2 ############################################
##############################################################################


from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(penalty='l1',solver='liblinear')
clf_LR.fit(X_train,y_train)
y_test_pred = clf_LR.predict(X_test)


X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
xx = pd.Series(y_test_pred, name='65')
df = pd.concat([X_test,y_test,xx], axis=1)

CC_test =np.array( pd.crosstab(index=df['64'], columns=df['65']))


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


print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("BuiltIn_Precision:",metrics.precision_score(y_test, y_test_pred, average='micro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, y_test_pred,  average='macro'))



##############################################################################
# Task 3 #####################################################################
##############################################################################


from sklearn.ensemble import RandomForestClassifier
rF = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
model = rF.fit(X_train,y_train)
y_test_pred = model.predict(X_test)


X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
xx = pd.Series(y_test_pred, name='65')
df1 = pd.concat([X_test,y_test,xx], axis=1)

CC_test =np.array( pd.crosstab(index=df1['64'], columns=df1['65']))
CC_test

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


print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("BuiltIn_Precision:",metrics.precision_score(y_test, y_test_pred, average='micro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, y_test_pred,  average='macro'))

##############################################################################
## Task 4 ####################################################################
##############################################################################


from sklearn import metrics

print(metrics.classification_report(y_test, y_test_pred))



##############################################################################
##############################################################################
############## For class dataset #########################################
##############################################################################
##############################################################################



input_data = pd.read_csv("/Users/gilbertec.fleurisma/merged12.csv")

# Label/Response set
y = input_data['64']

# Drop the labels and store the features
input_data.drop('64',axis=1,inplace=True)
X = input_data


##############################################################################
# Task 1 #####################################################################
##############################################################################


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)





##############################################################################
########################## Task 2 ############################################
##############################################################################


from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression(penalty='l1',solver='liblinear')
clf_LR.fit(X_train,y_train)
y_test_pred = clf_LR.predict(X_test)


X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
xx = pd.Series(y_test_pred, name='65')
df = pd.concat([X_test,y_test,xx], axis=1)

CC_test =np.array( pd.crosstab(index=df['64'], columns=df['65']))


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


print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("BuiltIn_Precision:",metrics.precision_score(y_test, y_test_pred, average='micro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, y_test_pred,  average='macro'))



##############################################################################
# Task 3 #####################################################################
##############################################################################


from sklearn.ensemble import RandomForestClassifier
rF = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
model = rF.fit(X_train,y_train)
y_test_pred = model.predict(X_test)


X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
xx = pd.Series(y_test_pred, name='65')
df1 = pd.concat([X_test,y_test,xx], axis=1)

CC_test =np.array( pd.crosstab(index=df1['64'], columns=df1['65']))
CC_test

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


print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, y_test_pred))
print("BuiltIn_Precision:",metrics.precision_score(y_test, y_test_pred, average='micro'))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, y_test_pred,  average='macro'))


##############################################################################
## Task 4 ####################################################################
##############################################################################


from sklearn import metrics

print(metrics.classification_report(y_test, y_test_pred))




