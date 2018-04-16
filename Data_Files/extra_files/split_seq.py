
import sys
import os
import subprocess

from Bio import SeqIO

count = 1
part = 1
fh_out = open("part1_" + sys.argv[1], "w")
split_number = int(sys.argv[2])

for seq_record in SeqIO.parse(sys.argv[1], "fasta"):

    if (count > split_number):
        part += 1
        fh_out.close()
        fh_out = open("part" + str(part) + "_" + sys.argv[1], "w")
        count = 1

    SeqIO.write(seq_record, fh_out, "fasta")
    count += 1;


