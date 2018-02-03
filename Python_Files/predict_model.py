import sys
import io
import string
import json
from pprint import pprint
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

aminoAcids = 'ACDEGHILKMNFPQRSTVWY'
windowSize = 21

def main():
    print (sys.argv)
    if len(sys.argv) > 3:
        print("Invalid Entry!")
        print("Expect arguments in format <python sieveson.py JSONFILE function> ")
    else:
        if (sys.argv[2] == 'build'):
            encodeJSON(sys.argv[1],buildModel)
        elif(sys.argv[2] == 'predict'):
            encodeJSON(sys.argv[1],predictModel)
        else:
            print("Invalid Entry!")
            print("Expect arguments in format <python sieveson.py JSONFILE function> ")

def encodeJSON(path,foo):
    seqs = []
    anns = []
    numStructured = 0
    numDisordered = 0
    numContext = 0
    with open(path, 'r') as r:
        original = json.load(r)
    print(len(original))
    for i in range(0, len(original)):
        struct, disord, cont, currentSeq, currentAnn = seqUnravel(original[i]['sequence'],original[i]['consensus'])
        seqs += currentSeq
        anns += currentAnn
        numStructured += struct
        numDisordered += disord
        numContext += cont
    result = [seqs, anns]
    seqs1 = np.asarray(seqs)
    anns1 = np.asarray(anns)
    print(seqs1.shape)
    print(numDisordered, numStructured, numContext)
    foo(seqs1,anns)

def seqUnravel(sequence, consensus):
    result = []
    for i in range(0,len(sequence)):
        oneHot = [0] * 21
        for j in range(0,len(aminoAcids)):
            if aminoAcids[j] == sequence[i]:
                oneHot[j] = 1
        result += [oneHot]
    struct, disord, cont, finalSequence, finalConsensus = splitSequence(result, consensus)
    return struct, disord, cont, finalSequence, finalConsensus

def splitSequence(sequence, consensus):
    boundaries = [0]*20 + [1]
    encodedSeq = []
    encodedAnns = []
    struct = 0
    disord = 0
    cont = 0
    for i in range(0,len(consensus)):
        x = range(consensus[i]['start']-1,consensus[i]['end'])
        start = consensus[i]['start']-1
        end = consensus[i]['end']-1
        annotation = str(consensus[i]['ann']).upper()
        if (annotation == 'S'):
            annotation = 0
            struct += (end - start + 1)
        elif (annotation == 'D'):
            annotation = 1
            disord += (end - start + 1)
        else :
            annotation = 2
            cont += (end - start + 1)
        for j in x:
            if (j < ((windowSize-1)/2)):
                current = [boundaries]*(((windowSize-1)/2)-j) + sequence[0:min(j+((windowSize+1)/2),len(sequence)-1)]
                current = current + [boundaries]*(windowSize-len(current))
            elif (j > end-((windowSize-1)/2)):
                current = sequence[(j-((windowSize-1)/2)):end+1] + [boundaries]*(((windowSize-1)/2)-(end - j))
            else:
                current = sequence[(j-((windowSize-1)/2)):(j+((windowSize+1)/2))]
            current = np.array(current)
            current = np.ravel(current)
            encodedSeq += [current]
            encodedAnns += [annotation]
    return struct, disord, cont, encodedSeq , encodedAnns

def buildModel(X, y):
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
    model = MLPClassifier(hidden_layer_sizes=(windowSize*21,windowSize*21,1), max_iter=500)
    # model.fit(scaledTrainX,y)
    scores = cross_val_score(model, scaledTrainX,y,cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Fitted Model !\n" + "Now saving model and scaler")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

def predictModel(X,y):
    print("Loading model and scaler")
    model = joblib.load('model.pkl')
    scaler  = joblib.load('scaler.pkl')
    scaledTestX = scaler.transform(X)

    predictedY = model.predict(scaledTestX)
    print(confusion_matrix(y,predictedY))
    print(classification_report(y,predictedY))

if __name__ == "__main__":
    main()
