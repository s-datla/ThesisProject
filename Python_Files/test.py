import sys
import organelle
from Bio import SeqIO

path = "../Data_Files/test_files/casp10.fa"

length = 0
for seq_record in SeqIO.parse(path,"fasta"):
	length += len(str(seq_record.seq))
print length
