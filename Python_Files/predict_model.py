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
from scipy import interp
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
from itertools import cycle
from organelle import buildPredict

from Bio import SeqIO

# Metric calculation functions
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, roc_curve, auc
from sklearn.model_selection import cross_val_score
from imblearn.metrics import classification_report_imbalanced

# Classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
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
        elif(sys.argv[2] == 'organelle' and len(sys.argv) == 4):
            windowSize = int(sys.argv[3])
            buildOrganelle(sys.argv[1])
        elif(sys.argv[2] == 'predict'):
            predictModel(sys.argv[1])
        elif(sys.argv[2] == 'predictMTX'):
            predictMTX(sys.argv[1])
        elif(sys.argv[2] == 'predictOrganelle' and len(sys.argv) == 4):
            predictOrganelle(sys.argv[1],sys.argv[3])
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


    '''
        The following models were used:
            Multi Layered Perceptron Classifier (Layout -  441,2)
            Balanced Bagging Meta Estimator (200 Weak Estimators)
            Balanced Baggin Meta Estimator (MLP Layout)
            Support Vector Classifier (Balanced class weights, OneVSRest)
    '''
    # std = MLPClassifier(hidden_layer_sizes=(windowSize*21,2),max_iter=500, solver='adam',random_state=19)
    # svc = SVC(class_weight='balanced',random_state=19,decision_function_shape='ovr')
    # bag = BalancedBaggingClassifier(n_estimators=200,random_state=19)
    # bbc = BalancedBaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(windowSize*21,2),max_iter=500),ratio='auto',replacement=False,random_state=19)

    # Fitting All Models
    # bbc.fit(X_train,Y_train)
    # svc.fit(X_train,Y_train)
    # bag.fit(X_train,Y_train)
    # std.fit(X_train,Y_train)

    print("Fitted Model !\n" +  "Now saving model and scaler")

    # Saving fitted Models and scaler
    # joblib.dump(svc, 'svc.pkl')
    # joblib.dump(bbc, 'bag_model.pkl')
    # joblib.dump(bag,'bag.pkl')
    # joblib.dump(std,'std.pkl')
    # joblib.dump(scaler, 'bag_scaler.pkl')

    # Loading saved Models and scaler
    bbc = joblib.load('bag_model.pkl')
    bag = joblib.load('bag.pkl')
    std = joblib.load('std.pkl')
    # svc = joblib.load('svc.pkl')
    # scaler  = joblib.load('bag_scaler.pkl')

    # predY1 = svc.predict(X_test)
    predY1 = bbc.predict(X_test)
    predY2 = bag.predict(X_test)
    predY3 = std.predict(X_test)

    # Classification Metric display
    # print "SVC"
    # print(confusion_matrix(Y_test,predY1))
    # print(classification_report_imbalanced(Y_test,predY1))
    # print(matthews_corrcoef(Y_test, predY1))
    print "Balanced Bagging MLP"
    print(confusion_matrix(Y_test,predY1))
    print(classification_report_imbalanced(Y_test,predY1))
    print(matthews_corrcoef(Y_test, predY1))
    print "Balanced Bagging"
    print(confusion_matrix(Y_test,predY2))
    print(classification_report_imbalanced(Y_test,predY2))
    print(matthews_corrcoef(Y_test, predY2))
    print "Standard MLP"
    print(confusion_matrix(Y_test,predY3))
    print(classification_report_imbalanced(Y_test,predY3))
    print(matthews_corrcoef(Y_test, predY3))

    probs_bbc = bbc.predict_proba(X_test)
    probs_bag = bag.predict_proba(X_test)
    probs_std = std.predict_proba(X_test)
    # probs_svc = svc.predict_proba(X_test)

    ROCplot(probs_bbc,Y_test,"Plots/ROCplotBBC.png")
    ROCplot(probs_bag,Y_test,"Plots/ROCplotBAG.png")
    ROCplot(probs_std,Y_test,"Plots/ROCplotSTD.png")
    # ROCplot(probs_svc,Y_test,"ROCplotSVC.png")

    multiROCplot([probs_bbc,probs_bag,probs_std],Y_test,"Plots/multiROCplot.png",['Bagging MLP','Bagging','MLP'])

    # bbc_probs_train = bbc.predict_proba(X_train)
    # bag_probs_train = bag.predict_proba(X_train)
    # std_probs_train = std.predict_proba(X_train)
    # svc_probs_train = std.predict_proba(X_train)
    # print len(std_probs_train), len(bag_probs_train), len(bbc_probs_train)


def multiROCplot(probs_list, Y_test,save,models):
    Y_test = np.array([[1 if x == 0 else 0,x] for x in Y_test])
    print probs.shape, Y_test.shape
    assert(len(models) == len(probs_list));
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for j in range(0,len(probs_list)):
        for i in range(0,2):
            fpr[(j,i)], tpr[(j,i)], _ = roc_curve(Y_test[:,i],probs_list[j][:,i])
            # fpr[(j,i)], tpr[(j,i)], _ = roc_curve(Y_test,[probs_list[j][k][i] for k in range(len(probs_list[0]))])
            roc_auc[(j,i)] = auc(fpr[(j,i)],tpr[(j,i)])
        fpr[(j,'micro')], tpr[(j,'micro')], _ = roc_curve(Y_test.ravel(), probs_list[j].ravel())
        roc_auc[(j,'micro')] = auc(fpr[(j,'micro')], tpr[(j,'micro')])
        all_fpr = np.unique(np.concatenate([fpr[(j,k)] for k in range(0,2)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(0,2):
            mean_tpr += interp(all_fpr,fpr[(j,i)],tpr[(j,i)])
        mean_tpr /= 2
        fpr[(j,'macro')] = all_fpr
        tpr[(j,'macro')] = mean_tpr
        roc_auc[(j,'macro')] = auc(fpr[(j,'macro')],tpr[(j,'macro')])
    lw = 2
    pyplot.figure()
    color_list = cycle([['olivedrab', 'darkorange'], ['darkorchid','navy'],['black','firebrick'],['gold','slategrey']])
    for j,colors in zip(range(0,len(probs_list)),color_list):
        pyplot.plot(fpr[(j,'micro')], tpr[(j,'micro')],
             label='micro-average ROC curve (area = {0:0.2f}), Model:{1}'.format(roc_auc[(j,'micro')],models[j]),color=colors[0], linestyle=':', linewidth=4)
        pyplot.plot(fpr[(j,'macro')], tpr[(j,'macro')],
                 label='macro-average ROC curve (area = {0:0.2f}), Model:{1}'.format(roc_auc[(j,'macro')],models[j]),color=colors[1], linestyle=':', linewidth=4)
    pyplot.plot([0, 1], [0, 1], 'r--', lw=lw)
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('Multi-Model Receiver Operating Characteristic Plot')
    l = pyplot.legend(loc='upper center', bbox_to_anchor=(0.5,-0.1))
    # pyplot.show()
    pyplot.savefig(save, bbox_extra_artists=(l,), bbox_inches='tight')

def ROCplot(probs,Y_test,save):
    Y_test = np.array([[1 if x == 0 else 0,x] for x in Y_test])
    # Y_test = label_binarize(Y_test,classes=[0,1])
    print probs.shape, Y_test.shape
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(0,2):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:,i],probs[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(Y_test.ravel(), probs.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0,2)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(0,2):
        mean_tpr += interp(all_fpr,fpr[i],tpr[i])
    mean_tpr /= 2
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'],tpr['macro'])

    lw = 2
    pyplot.figure()
    pyplot.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),color='black', linestyle=':', linewidth=4)
    pyplot.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),color='navy', linestyle=':', linewidth=4)

    colors = cycle(['olivedrab', 'darkorange', 'darkorchid'])
    for i, color in zip(range(0,2), colors):
        pyplot.plot(fpr[i], tpr[i], color=color, lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    pyplot.plot([0, 1], [0, 1], 'r--', lw=lw)
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('Receiver Operating Characteristic Plot')
    pyplot.legend(loc="lower right")
    # pyplot.show()
    pyplot.savefig(save)

def buildOrganelle(path):

    seqs = []
    lengths = []
    for seq_record in SeqIO.parse("../Data_Files/train_files/disprot.fa","fasta"):
        seq = str(seq_record.seq)
        seqs += [seq]
        lengths += [len(seq)]
    temp_org = np.array(buildPredict(seqs))
    probs_org = []
    for i in range(0,len(lengths)):
        probs_org += [temp_org[i]] * lengths[i]
    probs_org = np.array(probs_org)
    print "Len Probs Org: {} ".format(len(probs_org))
    print probs_org.shape
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

    # base = BalancedBaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(windowSize*21,2),max_iter=500),ratio='auto',replacement=False,random_state=19)

    # base.fit(X_train,Y_train)
    # Saving fitted Models and scaler
    # joblib.dump(base, 'org_base.pkl')
    # joblib.dump(scaler, 'bag_scaler.pkl')

    # Loading saved Models and scaler
    base = joblib.load('org_base.pkl')
    # scaler  = joblib.load('bag_scaler.pkl')

    base_probs = base.predict_proba(scaledTrainX)
    print base_probs.shape
    org_X = np.hstack((base_probs,probs_org))
    split_index = len(Y_train)

    org_scaler = StandardScaler()
    print(org_scaler.fit(org_X))
    scaled_org_train = org_scaler.transform(org_X[:split_index])
    scaled_org_test = org_scaler.transform(org_X[split_index:])

    # std = MLPClassifier(hidden_layer_sizes=(6,2),max_iter=500, solver='adam',random_state=19)
    svc = SVC(class_weight='balanced',random_state=19,decision_function_shape='ovo')
    # bag = BalancedBaggingClassifier(n_estimators=200,random_state=19)
    # bbc = BalancedBaggingClassifier(base_estimator=MLPClassifier(hidden_layer_sizes=(6,2),max_iter=500),ratio='auto',replacement=False,random_state=19)
    #
    # std.fit(scaled_org_train,Y_train)
    svc.fit(scaled_org_train,Y_train)
    # bag.fit(scaled_org_train,Y_train)
    # bbc.fit(scaled_org_train,Y_train)

    # Saving fitted Models and scaler
    joblib.dump(svc, 'org_svc.pkl')
    # joblib.dump(bbc, 'org_bbc.pkl')
    # joblib.dump(bag,'org_bag.pkl')
    # joblib.dump(std,'org_std.pkl')
    joblib.dump(org_scaler, 'org_scaler.pkl')

    # Loading saved Models and scaler
    bbc = joblib.load('org_bbc.pkl')
    bag = joblib.load('org_bag.pkl')
    std = joblib.load('org_std.pkl')
    # svc = joblib.load('org_svc.pkl')
    # org_scaler  = joblib.load('org_scaler.pkl')

    predY1 = bbc.predict(scaled_org_test)
    predY2 = bag.predict(scaled_org_test)
    predY3 = std.predict(scaled_org_test)
    predY4 = svc.predict(scaled_org_test)

    # Classification Metric display
    print "Balanced Bagging MLP"
    print(confusion_matrix(Y_test,predY1))
    print(classification_report_imbalanced(Y_test,predY1))
    print(matthews_corrcoef(Y_test, predY1))
    print "Balanced Bagging"
    print(confusion_matrix(Y_test,predY2))
    print(classification_report_imbalanced(Y_test,predY2))
    print(matthews_corrcoef(Y_test, predY2))
    print "Standard MLP"
    print(confusion_matrix(Y_test,predY3))
    print(classification_report_imbalanced(Y_test,predY3))
    print(matthews_corrcoef(Y_test, predY3))
    print "SVC"
    print(confusion_matrix(Y_test,predY4))
    print(classification_report_imbalanced(Y_test,predY4))
    print(matthews_corrcoef(Y_test, predY4))

    probs_bbc = bbc.predict_proba(scaled_org_test)
    probs_bag = bag.predict_proba(scaled_org_test)
    probs_std = std.predict_proba(scaled_org_test)
    probs_svc = svc.decision_function(scaled_org_test)

    ROCplot(probs_bbc,Y_test,"Plots/ROCplotBBC-org.png")
    ROCplot(probs_bag,Y_test,"Plots/ROCplotBAG-org.png")
    ROCplot(probs_std,Y_test,"Plots/ROCplotSTD-org.png")
    ROCplot(probs_svc,Y_test,"Plots/ROCplotSVC-org.png")

    multiROCplot([probs_bbc,probs_bag,probs_std,probs_svc],Y_test,"Plots/multiROCplot-org.png",['Bagging MLP','Bagging','MLP','SVC'])

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
    bbc = joblib.load('bag_model.pkl')
    bag = joblib.load('bag.pkl')
    std = joblib.load('std.pkl')
    # svc = joblib.load('svc.pkl')
    scaler  = joblib.load('bag_scaler.pkl')
    # model = joblib.load('nn_pssm.pkl')
    scaledTestX = scaler.transform(mod_x)

    predY1 = bbc.predict(scaledTestX)
    predY2 = bag.predict(scaledTestX)
    predY3 = std.predict(scaledTestX)

    # Classification Metric display
    print "Balanced Bagging MLP"
    print(confusion_matrix(Y,predY1))
    print(classification_report_imbalanced(Y,predY1))
    print(matthews_corrcoef(Y, predY1))
    print "Balanced Bagging"
    print(confusion_matrix(Y,predY2))
    print(classification_report_imbalanced(Y,predY2))
    print(matthews_corrcoef(Y, predY2))
    print "Standard MLP"
    print(confusion_matrix(Y,predY3))
    print(classification_report_imbalanced(Y,predY3))
    print(matthews_corrcoef(Y, predY3))
    # print "SVC"
    # print(confusion_matrix(Y_test,predY4))
    # print(classification_report_imbalanced(Y_test,predY4))
    # print(matthews_corrcoef(Y_test, predY4))

def predictOrganelle(path,seq_path):
    
    seqs = []
    lengths = []
    for seq_record in SeqIO.parse(seq_path,"fasta"):
        seq = str(seq_record.seq)
        seqs += [seq]
        lengths += [len(seq)]
    temp_org = np.array(buildPredict(seqs))
    probs_org = []
    for i in range(0,len(lengths)):
        probs_org += [temp_org[i]] * lengths[i]
    probs_org = np.array(probs_org)
    print "Len Probs Org: {} ".format(len(probs_org))
    # print probs_org.shape

    mtxCompressed = np.load(path)
    X = mtxCompressed['savedX']
    Y = mtxCompressed['savedY']
    # print X.shape, Y.shape
    mod_x = np.reshape(X,(X.shape[0],X.shape[1] * X.shape[2]))
    
    base = joblib.load('org_base.pkl')
    scaler  = joblib.load('bag_scaler.pkl')
    scaledTestX = scaler.transform(mod_x)

    base_probs = base.predict_proba(scaledTestX)
    # print base_probs.shape
    org_X = np.hstack((base_probs,probs_org))
    print org_X.shape
    bbc = joblib.load('org_bbc.pkl')
    bag = joblib.load('org_bag.pkl')
    std = joblib.load('org_std.pkl')
    svc = joblib.load('org_svc.pkl')
    org_scaler = joblib.load('org_scaler.pkl')

    scaled_org_test = org_scaler.transform(org_X)

    predY1 = bbc.predict(scaled_org_test)
    predY2 = bag.predict(scaled_org_test)
    predY3 = std.predict(scaled_org_test)
    predY4 = svc.predict(scaled_org_test)

    # Classification Metric display
    print "Balanced Bagging MLP"
    print(confusion_matrix(Y,predY1))
    print(classification_report_imbalanced(Y,predY1))
    print(matthews_corrcoef(Y, predY1))
    print "Balanced Bagging"
    print(confusion_matrix(Y,predY2))
    print(classification_report_imbalanced(Y,predY2))
    print(matthews_corrcoef(Y, predY2))
    print "Standard MLP"
    print(confusion_matrix(Y,predY3))
    print(classification_report_imbalanced(Y,predY3))
    print(matthews_corrcoef(Y, predY3))
    print "SVC"
    print(confusion_matrix(Y,predY4))
    print(classification_report_imbalanced(Y,predY4))
    print(matthews_corrcoef(Y, predY4))

if __name__ == "__main__":
    main()
