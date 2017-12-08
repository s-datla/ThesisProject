import sys
import io
import string
import json
from pprint import pprint
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

aminoAcids = 'ACDEGHILKMNFPQRSTVWY'

def main():
    print (sys.argv)
    if len(sys.argv) > 3:
        print("Invalid Entry!")
        print("Expect arguments in format <python sieveson.py JSONFILE function> ")
    else:
        if (sys.argv[2] == 'encode'):
            encodeJSON(sys.argv[1])
        elif (sys.argv[2] == 'filter'):
            filterJSON(sys.argv[1])
        else:
            print("Invalid Entry!")
            print("Expect arguments in format <python sieveson.py JSONFILE function> ")

def filterJSON(path):
    result = []
    with open(path, 'r') as r:
        raw = json.load(r)
    for i in range(0, len(raw)):
        current = {}
        current['sequence'] = raw[i]['protein']['sequence']
        current['consensus'] = raw[i]['protein']['mobidb']['consensus']['predictors']
        result.append(current)
    print i
    with open('filterJSON.json','w') as w:
        json.dump(result, w)
    print ("Finished, created filtered json file <filterJSON.json>")

def encodeJSON(path):
    seqs = []
    anns = []
    with open(path, 'r') as r:
        original = json.load(r)
    for i in range(0, len(original)):
        currentSeq, currentAnn = seqUnravel(original[i]['sequence'],original[i]['consensus'])
        seqs += currentSeq
        anns += currentAnn
    result = [seqs, anns]
    seqs = np.asarray(seqs)
    anns = np.asarray(anns)
    print(seqs.shape)
    buildModel(seqs,anns)
    # with open('encoded.json', 'w') as w:
    #     json.dump(result,w)

def seqUnravel(sequence, consensus):
    result = []
    for i in range(0,len(sequence)):
        oneHot = [0] * 21
        for j in range(0,len(aminoAcids)):
            if aminoAcids[j] == sequence[i]:
                oneHot[j] = 1
        result += [oneHot]
    finalSequence, finalConsensus = splitSequence(result, consensus)
    return finalSequence, finalConsensus

def splitSequence(sequence, consensus):
    boundaries = [0]*20 + [1]
    encodedSeq = []
    encodedAnns = []
    for i in range(0,len(consensus)):
        x = range(consensus[i]['start']-1,consensus[i]['end'])
        start = consensus[i]['start']-1
        end = consensus[i]['end']-1
        annotation = str(consensus[i]['ann']).upper()
        if (annotation == 'S'):
            annotation = 0
        elif (annotation == 'D'):
            annotation = 1
        else :
            annotation = 2
        for j in x:
            current = []
            if (j < 7):
                current = [boundaries]*(7-j) + sequence[start:j+8]
            elif (j > end-7):
                current = sequence[(j-7):end+1] + [boundaries]*(7-(end - j))
            else:
                current = sequence[(j-7):(j+8)]
            current = np.asmatrix(current)
            current = np.ravel(current)
            encodedSeq += [current]
            encodedAnns += [annotation]

    return encodedSeq , encodedAnns

def buildModel(X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)
    model = LogisticRegression()
    scores = cross_val_score(model, X, y, cv=10)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # model.fit(X_train,y_train)
    # print(model.score(X_test,y_test))


if __name__ == "__main__":
    main()
