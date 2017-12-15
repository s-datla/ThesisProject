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
windowSize = 21

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
    numStructured = 0
    numDisordered = 0
    numContext = 0
    with open(path, 'r') as r:
        original = json.load(r)
    for i in range(0, len(original)):
        struct, disord, cont, currentSeq, currentAnn = seqUnravel(original[i]['sequence'],original[i]['consensus'])
        seqs += currentSeq
        anns += currentAnn
        numStructured += struct
        numDisordered += disord
        numContext += cont
    result = [seqs, anns]
    seqs = np.asarray(seqs)
    anns = np.asarray(anns)
    print(seqs.shape)
    print(numDisordered, numStructured, numContext)
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
            current = []
            if (j < ((windowSize-1)/2)):
                current = [boundaries]*(((windowSize-1)/2)-j) + sequence[start:j+((windowSize+1)/2)]
            elif (j > end-((windowSize-1)/2)):
                current = sequence[(j-((windowSize-1)/2)):end+1] + [boundaries]*(((windowSize-1)/2)-(end - j))
            else:
                current = sequence[(j-((windowSize-1)/2)):(j+((windowSize+1)/2))]
            current = np.asmatrix(current)
            current = np.ravel(current)
            encodedSeq += [current]
            encodedAnns += [annotation]

    return struct, disord, cont, encodedSeq , encodedAnns

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
