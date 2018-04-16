import os, glob, time, sys, shutil
from Bio import SeqIO

if not os.path.exists("../mtx_files"):
    os.makedirs("../mtx_files")

counter = 0

for seq_record in SeqIO.parse(sys.argv[1],"fasta"):
    current = seq_record.id
    print current
    file_path = "../DISOPRED_RUNS/{}/*.*".format(current)
    for cleanFile in glob.glob(file_path):
        print cleanFile
        if not cleanFile.endswith('.mtx'):
            os.remove(cleanFile)
        if cleanFile.endswith('.mtx'):
            shutil.move(cleanFile, "../mtx_files/")
            counter += 1
print counter
