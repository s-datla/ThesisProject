import sys, os, json, glob
import csv

from Bio import SeqIO

cwd = os.getcwd()

json_file = []

total = 0
for drFile in glob.glob("{}/*.dr".format(cwd)):
    print drFile
    with open(drFile,"r") as r:
        counter = 0
        reader = csv.reader(r,delimiter='\t')
        current = {}
        sequence = ""
        state = ''
        consensus = []
        start = 1
        end = 1
        label = ''
        for row in reader:
            if len(row) == 3:
                counter += 1
                sequence += row[0]
                if row[1] == 'D':
                    label = 'd'
                else: 
                    label = 's'
                if row[2] == '1':
                    state = label
                if not (state == label):
                    end = int(row[2]) - 1
                    consensus.append({
                        'start': start,
                        'ann': state,
                        'end': end
                    })
                    state = label
                    start = int(row[2])
                    end = int(row[2])                    
        consensus.append({
            'start': start,
            'ann': state,
            'end': counter
        })
        json_file.append({
            'consensus': consensus,
            'sequence': sequence
        })
        print counter
        total += counter
with open('casp10.json','w') as w:
    json.dump(json_file,w)
print total

