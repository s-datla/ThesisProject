
import sys
import os
import subprocess

from Bio import SeqIO

for seq_record in SeqIO.parse(sys.argv[2], "fasta"):

    print("PROCESSING ... " + seq_record)
    os.makedirs('./' + seq_record.id)
    os.chdir(seq_record.id)
    seq_file = seq_record.id + ".fa"
    SeqIO.write(seq_record, seq_file, "fasta")
    subprocess.call(['../' + sys.argv[1], seq_file])
    os.chdir('..')


