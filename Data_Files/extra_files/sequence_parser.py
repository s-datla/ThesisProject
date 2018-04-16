

import sys

from Bio import SeqIO
for seq_record in SeqIO.parse(sys.argv[1], "fasta"):

    print("Writing " + seq_record.id)
    SeqIO.write(seq_record, seq_record.id + ".fa", "fasta")
