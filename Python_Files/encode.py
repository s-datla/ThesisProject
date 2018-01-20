import sys
import io
import string
import json
import csv
from pprint import pprint
import numpy as np


def main():
    print (sys.argv)
    if len(sys.argv) > 3:
        print("Invalid Entry!")
        print("Expect arguments in format <python sieveson.py JSONFILE function> ")
    else:
        if (sys.argv[2] == 'filter'):
            filterJSON(sys.argv[1])
        elif(sys.argv[2] == 'fasta'):
            fastaFormat(sys.argv[1])
        elif(sys.argv[2] == 'tab'):
            filterTAB(sys.argv[1])
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


def fastaFormat(path):
    result = []
    n = 80
    with open(path, 'r') as r:
        raw = json.load(r)
    for i in range(0, len(raw)):
        ident = ">" + str(raw[i]['protein']['disprot_id']) + " | " + str(raw[i]['protein']['protein_name'])
        current = [ident]
        seq = raw[i]['protein']['sequence']
        current += [seq[i:i+n] for i in range(0,len(seq),n)]
        result += current
    with open('disprotFasta.fa','w') as w:
        for line in result:
            w.write(line + "\n")
    print ("Finished, created fasta format file <disprotFasta.fa>")

def filterTAB(path):
    result = []
    with open(path, 'r') as r:
        current = {}
        sequence = ''
        i = 0
        state = 's'
        consensus = []
        start = 1
        end = 1
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
                result.append(current)
                current = {}
                sequence = ''
                i = 1
                start = 1
                state = 's'
                consensus = []
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
    consensus.append({
        'start': start,
        'ann': state,
        'end': end
    })
    current['consensus'] = consensus
    result.append(current)
    result.pop(0)
    with open('../JSON_Files/test_files/disopredJSON.json','w') as w:
        json.dump(result, w)
    print ("Finished, created filtered json file <disopredJSON.json>")

if __name__ == "__main__":
    main()
