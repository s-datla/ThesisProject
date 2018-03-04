import sys
import os
import subprocess
import numpy as np
from pprint import pprint
import math

def main():
    if len(sys.argv) > 4:
        print("Invalid Entry!")
        print("Expect arguments in format <python sieveson.py JSONFILE function> ")
    else:
        crawl_files(sys.argv[1],sys.argv[2],sys.argv[3])

def crawl_files(path,anns,save):
    baseFileName = '../Data_Files/DisorderHunter-master/DISOPRED_RUNS/'
    PSSMvalues = []
    seqLengths = []
    with open(path, 'r') as r:
        for lines in r:
            if lines[0] == '>':
                identifiers = lines.split(' ')
                accessionID = identifiers[0].strip('>')
                # print accessionID
                currentPSSM,length = process_mtx(baseFileName + str(accessionID) + '/' + str(accessionID) + '.mtx')
                PSSMvalues += currentPSSM
                seqLengths += [length]
    annotations = load_compressed_file(anns)
    pssm, limits = linear_scaler(PSSMvalues)
    pssmNP = np.array(pssm)
    limitsNP = np.array(limits)
    seqLengthsNP = np.array(seqLengths)
    np.savez_compressed(save,pssm=pssmNP,limits=limitsNP,length=seqLengthsNP, labels=annotations)
    print "Sorted through MTX files (#files: " + str(len(seqLengths)) + ")"
    print "Saved compressed PSSM in " + save

def process_mtx(file):
    # The amino acid values are encoded in the mtx files as XAXCDEFGHIKLMNPQRSTVWXYXXXX, where X denotes the unknown amino acids / terminating characters
    mtxPositions = [1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22];
    with open(file,'r') as r:
        raw = r.readlines()
    counter = 0
    pssm = []
    for i,line in enumerate(raw):
        if i == 0:
            seqLength = line.strip()
        elif i == 1:
            seq = line.strip()
        elif i >= 14:
            values = line.split()
            currentPSS = []
            for j in range(0,len(mtxPositions)):
                currentPSS += [int(values[mtxPositions[j]])]
            pssm += [currentPSS]
            counter += 1
    # print counter,seqLength,len(PSSM),len(PSSM[0])
    if counter > seqLength:
        print "File: " + str(file) + " has invalid sequence length!"
        print "Expected: " + str(seqLength) + " , Actual: " + str(counter)
    return pssm, seqLength


def load_compressed_file(path):
    compressed = np.load(path)
    annotations = compressed['labels']
    print str(len(annotations)) + " annotations"
    return annotations

def linear_scaler(pssm):
    finalLimits = [[0,0]]*20
    for i in range(0,len(finalLimits)):
        finalLimits[i] = [min(el[i] for el in pssm),max(el[i] for el in pssm)]
    for j in range(0, len(pssm)):
        for k in range(0,len(finalLimits)):
            pssm[j][k] = math.log(min(float(pssm[j][k] - finalLimits[k][0]+1)/float(finalLimits[k][1] - finalLimits[k][0]),1))
    return pssm, finalLimits

if __name__ == "__main__":
    main()
