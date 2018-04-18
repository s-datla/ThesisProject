import sys
import organelle
from Bio import SeqIO

path = "../Data_Files/test_files/disopred3test.fa"

seqs = []
for seq_record in SeqIO.parse(path,'fasta'):
    seqs += [str(seq_record.seq)]
print seqs
vals = organelle.buildPredict(seqs)

print vals
