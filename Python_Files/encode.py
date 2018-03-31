import sys
import io
import string
import json
import csv
from pprint import pprint
import numpy as np
import math
windowSize = 7
aminoAcids = 'ACDEGHILKMNFPQRSTVWY'


def main():
    global windowSize
    if len(sys.argv) > 5:
        print("Invalid Entry!")
        print("Expect arguments in format <python encode.py JSONFILE function OUTPUT> ")
    else:
        if (sys.argv[2] == 'filter'):
            filter_JSON(sys.argv[1])
        elif(sys.argv[2] == 'fasta1' and len(sys.argv) == 4):
            fasta_format(sys.argv[1], 1, str(sys.argv[3]))
        elif(sys.argv[2] == 'fasta2' and len(sys.argv) == 4):
            fasta_format(sys.argv[1], 2, str(sys.argv[3]))
        elif(sys.argv[2] == 'tab'):
            filter_TAB(sys.argv[1])
        elif(sys.argv[2] == 'json' and len(sys.argv) == 5):
            windowSize = int(sys.argv[4])
            encode_JSON(sys.argv[1], str(sys.argv[3]))
        elif(sys.argv[2] == 'mtx' and len(sys.argv) == 5):
            windowSize = int(sys.argv[4])
            encode_MTX(sys.argv[1], str(sys.argv[3]))
        else:
            print("Invalid Entry!")
            print("Expect arguments in format <python encode.py JSONFILE function OUTPUT> ")

def filter_JSON(path):
    result = []
    with open(path, 'r') as r:
        raw = json.load(r)
    for i in range(0, len(raw)):
        current = {}
        current['sequence'] = raw[i]['protein']['sequence']
        if not raw[i]['protein']['mobidb']['consensus']['predictors']:
            current['consensus'] = raw[i]['protein']['mobidb']['consensus']['pdb']
        else:
            current['consensus'] = raw[i]['protein']['mobidb']['consensus']['predictors']
        result.append(current)
    print i+1
    with open('../Data_Files/filterJSON.json','w') as w:
        json.dump(result, w)
    print ("Finished, created filtered json file <filterJSON.json>")


def fasta_format(path, state, newfile):
    result = []
    n = 80
    with open(path, 'r') as r:
        raw = json.load(r)
    for i in range(0, len(raw)):
        if (state == 1):
            ident = "> " + str(raw[i]['protein']['disprot_id']) + " | " + str(raw[i]['protein']['protein_name'])
            current = [ident]
            seq = raw[i]['protein']['sequence']
            current += [seq[i:i+n] for i in range(0,len(seq),n)]
            result += current
        elif (state == 2):
            ident = "> " + str(raw[i]['id']) + " | Protein No: " + str(i)
            current = [ident]
            seq = raw[i]['sequence']
            current += [seq[i:i+n] for i in range(0,len(seq),n)]
            result += current
    with open(newfile,'w') as w:
        for line in result:
            w.write(line + "\n")
    print ("Finished, created fasta format file <" + newfile + ">")

def filter_TAB(path):
    result = []
    with open(path, 'r') as r:
        current = {}
        sequence = ''
        i = 0
        state = 's'
        consensus = []
        start = 1
        end = 1
        ident = ''
        for line in csv.reader(r, dialect="excel-tab"):
            i += 1
            if (line[1] == '1'):
                current['sequence'] = sequence
                consensus.append({
                    'start': start,
                    'ann': state,
                    'end': end
                })
                current['consensus'] = consensus
                current['id'] = ident
                result.append(current)
                current = {}
                sequence = ''
                i = 1
                start = 1
                state = 's'
                consensus = []
                ident = line[0]
            if (state == 's'):
                if (line[3] == '1'):
                    consensus.append({
                        'start': start,
                        'ann': state,
                        'end': end
                    })
                    state = 'd'
                    start = i
            elif (state == 'd'):
                if(line[3] == '0'):
                    consensus.append({
                        'start': start,
                        'ann': state,
                        'end': end
                    })
                    state = 's'
                    start = i
            end = i
            sequence += line[2]
    current['sequence'] = sequence
    current['id'] = ident
    consensus.append({
        'start': start,
        'ann': state,
        'end': end
    })
    current['consensus'] = consensus
    current['id'] = ident
    result.append(current)
    result.pop(0)
    with open('../Data_Files/test_files/disopredJSON.json','w') as w:
        json.dump(result, w)
    print ("Finished, created filtered json file <disopredJSON.json>")
    pprint(result)

def encode_JSON(path,save):
    seqs = []
    anns = []
    numStructured = 0
    numDisordered = 0
    numContext = 0
    with open(path, 'r') as r:
        original = json.load(r)
    print(len(original))
    for i in range(0, len(original)):
        struct, disord, cont, currentSeq, currentAnn = sequence_unravel(original[i]['sequence'],original[i]['consensus'],i)
        seqs += currentSeq
        anns += currentAnn
        numStructured += struct
        numDisordered += disord
        numContext += cont
    result = [seqs, anns]
    seqs1 = np.asarray(seqs)
    anns1 = np.asarray(anns)
    print('Shape of data array: ' + str(seqs1.shape))
    np.savez_compressed(save, inputs=seqs1, labels=anns)
    print('Number of disordered positions: ' + str(numDisordered) + ' , Number of structured positions: ' + str(numStructured)
    + ' , Number of context dependent positions: ' + str(numContext))

def sequence_unravel(sequence, consensus,num):
    result = []
    for i in range(0,len(sequence)):
        oneHot = [0] * 21
        for j in range(0,len(aminoAcids)):
            if aminoAcids[j] == sequence[i]:
                oneHot[j] = 1
        result += [oneHot]
    struct, disord, cont, finalSequence, finalConsensus = split_sequence(result, consensus,num)
    return struct, disord, cont, finalSequence, finalConsensus

def split_sequence(sequence, consensus,num):
    boundaries = [0]*20 + [1]
    encodedSeq = []
    encodedAnns = []
    struct = 0
    disord = 0
    cont = 0
    total = 0
    for i in range(0,len(consensus)):
        x = range(consensus[i]['start']-1,consensus[i]['end'])
        start = consensus[i]['start']-1
        end = consensus[i]['end']-1
        total += end - start  + 1
        annotation = str(consensus[i]['ann']).upper()
        if (annotation == 'S'):
            annotation = 0
            struct += (end - start + 1)
        elif (annotation == 'D'):
            annotation = 1
            disord += (end - start + 1)
        else :
            annotation = 0
            struct += (end - start + 1)
        for j in x:
            if (j < ((windowSize-1)/2)):
                current = [boundaries]*(((windowSize-1)/2)-j) + sequence[0:min(j+((windowSize+1)/2),len(sequence)-1)]
                current += [boundaries]*(windowSize-len(current))
            elif (j > end-((windowSize-1)/2)):
                current = sequence[(j-((windowSize-1)/2)):end+1] + [boundaries]*(((windowSize-1)/2)-(end - j))
            else:
                current = sequence[(j-((windowSize-1)/2)):(j+((windowSize+1)/2))]
            current = np.array(current)
            current = np.ravel(current)
            encodedSeq += [current]
            encodedAnns += [annotation]
    if total < len(sequence) or len(encodedAnns) < len(sequence):
        print "Differences found ! "
        print total - len(sequence)
        print total, len(sequence), len(encodedSeq), len(encodedAnns)
        print num
    return struct, disord, cont, encodedSeq , encodedAnns

def encode_MTX(path,save):
    mtxCompressed = np.load(path)
    labels = mtxCompressed['labels']
    pssm = mtxCompressed['pssm']
    lengths = mtxCompressed['length']
    print len(pssm), len(labels), sum([int(i) for i in lengths])
    X = create_windows(pssm,lengths)
    Y = labels.tolist()
    savedX = np.array(X)
    savedY = np.array(Y)
    print(savedX.shape,savedY.shape)
    np.savez_compressed(save,savedX=savedX,savedY=savedY)


def create_windows(old_pssm,lengths):
    pssm = old_pssm.tolist()
    background = 10e-07
    boundaries = [math.log(background)]*20 + [math.log(1)]
    windows = []
    pssmPosition = 0
    for i in lengths:
        for j in range(0,int(i)):
            lowerLim = (pssmPosition + j) - ((windowSize-1)/2)
            upperLim = (pssmPosition + j) + ((windowSize+1)/2)
            current = [boundaries]*(((windowSize-1)/2)-j) + pssm[max(lowerLim,pssmPosition):min(pssmPosition+int(i),upperLim)]
            current += [boundaries]*(windowSize-len(current))
            windows += [current]
        pssmPosition += int(i)
    print "Created Windows !"
    print "Length: " + str(len(windows))
    return windows

if __name__ == "__main__":
    main()
