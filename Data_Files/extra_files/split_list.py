
import sys
import os
import subprocess

from Bio import SeqIO

count = 1
part = 1
#fh_out = open("part1_" + sys.argv[1], "w")
print("Part 1")
split_number = int(sys.argv[2])

for seq_record in SeqIO.parse(sys.argv[1], "fasta"):

    if (count > split_number):
        part += 1
        #fh_out.close()
        #fh_out = open("part" + str(part) + "_" + sys.argv[1], "w")
        print("Part " + str(part))
        count = 1

    print(seq_record.id)
    count += 1;


