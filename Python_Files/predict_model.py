# Standard imports
import sys
import io
import string
import json
import math

# Utility imports
from collections import Counter
from pprint import pprint
import numpy as np
import scipy.sparse
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

# Metric calculation functions
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_curve
from sklearn.model_selection import cross_val_score
from imblearn.metrics import classification_report_imbalanced

# Classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.ensemble import BalancedBaggingClassifier

windowSize = 7

def main():
    global windowSize
    if len(sys.argv) > 4:
        print("Invalid Entry!")
        print("Expect arguments in format <python sieveson.py JSONFILE function> ")
    else:
        if (sys.argv[2] == 'mtxBuild' and len(sys.argv) == 4):
            windowSize = int(sys.argv[3])
            mtxBuild(sys.argv[1])
        elif (sys.argv[2] == 'build' and len(sys.argv) == 4):
            windowSize = int(sys.argv[3])
            buildModel(sys.argv[1])
        elif(sys.argv[2] == 'bagClassify' and len(sys.argv) == 4):
            windowSize = int(sys.argv[3])
            bagClassify(sys.argv[1])
        elif(sys.argv[2] == 'predict'):
            predictModel(sys.argv[1])
        elif(sys.argv[2] == 'predictMTX'):
            predictMTX(sys.argv[1])
        else:
            print("Invalid Entry!")
            print("Expect arguments in format <python sieveson.py JSONFILE function> ")


def buildModel(path):
    compressed = np.load(path)
    X = compressed['inputs']
    Y = compressed['labels']
    print X.shape, Y.shape
    '''
    Code for use later (can be diagonostic or reverting back to previous models / parameters)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)
        model = LogisticRegression()
        model.fit(X,y)
        scores = cross_val_score(model, X, y, cv=10)
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        joblib.dump(model, 'logistic.pkl')
    '''
    scaler = StandardScaler()
    print(scaler.fit(X))
    scaledTrainX = scaler.transform(X)
    model = MLPClassifier(hidden_layer_sizes=(windowSize*21,2), max_iter=500)
    model.fit(scaledTrainX,y)
    # scores = cross_val_score(model, scaledTrainX,y,cv=10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Fitted Model !\n" + "Now saving model and scaler")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

def mtxBuild(path):
    mtxCompressed = np.load(path)
    X = mtxCompressed['savedX']
    Y = mtxCompressed['savedY']
    print("Started scaling !")
    X = np.reshape(X,(X.shape[0],X.shape[1] * X.shape[2]))
    scaler = StandardScaler()
    print(scaler.fit(X))
    scaledTrainX = scaler.transform(X)
    print("Splitting into Test and Train")
    X_train, X_test, Y_train, Y_test = train_test_split(scaledTrainX, Y, test_size=0.33, random_state=19)
    model = MLPClassifier(hidden_layer_sizes=(windowSize*20,2), max_iter=500)

    model.fit(X_train,Y_train)
    print("Fitted Model !\n" +  "Now saving model and scaler")
    joblib.dump(model, 'nn_pssm.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    predictedY = model.predict(X_test)
    print(confusion_matrix(Y_test,predictedY))
    print(classification_report(Y_test,predictedY))

def bagClassify(path):
    mtxCompressed = np.load(path)
    X = mtxCompressed['savedX']
    Y = mtxCompressed['savedY']
    print("Started scaling!")
    X = np.reshape(X,(X.shape[0],X.shape[1] * X.shape[2]))
    scaler = StandardScaler()
    print(scaler.fit(X))
    scaledTrainX = scaler.transform(X)
    print("Splitting into Test and Train")
    X_train, X_test, Y_train, Y_test = train_test_split(scaledTrainX, Y, test_size=0.33, random_state=19)
    print("Training Data is distributed as follows: " + str(sorted(Counter(Y_train).items())))
    print("Testing Data is distributed as follows: " + str(sorted(Counter(Y_test).items())))
    std = MLPClassifier(hidden_layer_sizes=(windowSize*20,2),max_iter=500, solver='adam',random_state=19)
    adam = MLPClassifier(hidden_layer_sizes=(windowSize*20,2),max_iter=500,random_state=19)
    fgs = MLPClassifier(hidden_layer_sizes=(windowSize*20,2),max_iter=500, solver='lbfgs',random_state=19)
    # bbc = BalancedBaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(windowSize*20,2),max_iter=500),ratio='auto',replacement=False,random_state=19)
    # bbc.fit(X_train,Y_train)
    std.fit(X_train,Y_train)
    adam.fit(X_train,Y_train)
    fgs.fit(X_train,Y_train)
    print("Fitted Model !\n" +  "Now saving model and scaler")
    # joblib.dump(bbc, 'bag_model.pkl')
    joblib.dump(std, 'std.pkl')
    joblib.dump(adam, 'adam.pkl')
    joblib.dump(fgs, 'fgs.pkl')
    joblib.dump(scaler, 'bag_scaler.pkl')
    predY1 = std.predict(X_test)
    predY2 = adam.predict(X_test)
    predY3 = fgs.predict(X_test)
    print(confusion_matrix(Y_test,predY1))
    print(confusion_matrix(Y_test,predY2))
    print(confusion_matrix(Y_test,predY3))
    print(classification_report(Y_test,predY1))
    print(classification_report(Y_test,predY2))
    print(classification_report(Y_test,predY3))

def ROCplot(predicted,ground):
    fpr, tpr, thresholds = roc_curve(ground,predicted,pos_label=1)

def predictModel(path):
    compressed = np.load(path)
    X = compressed['inputs']
    Y = compressed['labels']
    print X.shape, Y.shape
    print("Loading model and scaler")
    model = joblib.load('model.pkl')
    scaler  = joblib.load('scaler.pkl')
    scaledTestX = scaler.transform(X)

    predictedY = model.predict(scaledTestX)
    print(confusion_matrix(Y,predictedY))
    print(classification_report(Y,predictedY))

def predictMTX(path):
    mtxCompressed = np.load(path)
    X = mtxCompressed['savedX']
    Y = mtxCompressed['savedY']
    print X.shape, Y.shape
    mod_x = np.reshape(X,(X.shape[0],X.shape[1] * X.shape[2]))
    print("Loading model")
    model = joblib.load('nn_pssm.pkl')
    predictedY = model.predict(mod_x)
    print(confusion_matrix(Y,predictedY))
    print(classification_report(Y,predictedY))

if __name__ == "__main__":
    main()
