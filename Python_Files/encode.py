import sys
import io
import string
import json
import csv
from pprint import pprint
import numpy as np


def main():
    print (sys.argv)
    if len(sys.argv) > 4:
        print("Invalid Entry!")
        print("Expect arguments in format <python encode.py JSONFILE function OUTPUT> ")
    else:
        if (sys.argv[2] == 'filter'):
            filterJSON(sys.argv[1])
        elif(sys.argv[2] == 'fasta1' and len(sys.argv) == 4):
            fastaFormat(sys.argv[1], 1, str(sys.argv[3]))
        elif(sys.argv[2] == 'fasta2' and len(sys.argv) == 4):
            fastaFormat(sys.argv[1], 2, str(sys.argv[3]))
        elif(sys.argv[2] == 'tab'):
            filterTAB(sys.argv[1])
        else:
            print("Invalid Entry!")
            print("Expect arguments in format <python encode.py JSONFILE function OUTPUT> ")

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


def fastaFormat(path, state, newfile):
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
    with open('../JSON_Files/test_files/disopredJSON.json','w') as w:
        json.dump(result, w)
    print ("Finished, created filtered json file <disopredJSON.json>")
    pprint(result)

if __name__ == "__main__":
    main()
