import sys
import subprocess

import sys
import os
from subprocess import Popen, PIPE

output_file = "result6.txt"

read_mtx = [['read_mtx.py'],
            ['../Data_Files/train_files/disprot.fa','../Data_Files/test_files/disopred3.fa','../Data_Files/test_files/casp10.fa'],
            ['../Data_Files/temp_files/compressedDisprot.npz','../Data_Files/temp_files/compressedDisopred.npz','../Data_Files/temp_files/compressedCASP.npz'],
            ['../Data_Files/temp_files/pssmTrain','../Data_Files/temp_files/pssmTest','../Data_Files/temp_files/pssmCASP']]
json = [
            ['encode.py'],
            ['../Data_Files/train_files/disprotfiltered.json','../Data_Files/test_files/disopred3filtered.json','../Data_Files/test_files/casp10.json'],
            ['../Data_Files/temp_files/compressedTrain','../Data_Files/temp_files/compressedTest','../Data_Files/temp_files/compressedCASP']]
encoded = [
            ['encode.py'],
            ['../Data_Files/temp_files/pssmTrain.npz','../Data_Files/temp_files/pssmTest.npz','../Data_Files/temp_files/pssmCASP.npz'],
            ['../Data_Files/temp_files/encTrain','../Data_Files/temp_files/encTest','../Data_Files/temp_files/encCASP']]

predict = [
            ['predict_model.py'],
            ['../Data_Files/temp_files/encTrain.npz','../Data_Files/temp_files/encTest.npz','../Data_Files/temp_files/encCASP.npz'],
            ['../Data_Files/test_files/disopred3.fa','../Data_Files/test_files/casp10.fa']]
#
# for i in range(7,23,2):
#     cmd = 'python'
#     with open('result.txt','a') as a:
#         a.writelines("----WINDOW SIZE---\n")
#         a.writelines("---- " + str(i) + "----\n")
#     for j in range(0,2):
#         p = Popen([cmd,encoded[0][0],encoded[1][j],'mtx',encoded[2][j],str(i)], stdout=PIPE,stderr=PIPE)
#         out,err = p.communicate()
#         with open('result.txt','a') as a:
#             a.writelines(out)
#             a.writelines(err)
#     p = Popen([cmd,predict[0][0],predict[1][0],'mtxBuild',str(i)], stdout=PIPE,stderr=PIPE)
#     out,err = p.communicate()
#     with open('result.txt','a') as a:
#         a.writelines(out)
#         a.writelines(err)
#     p = Popen([cmd,predict[0][0],predict[1][1],'predictMTX'], stdout=PIPE,stderr=PIPE)
#     out,err = p.communicate()
#     with open('result.txt','a') as a:
#         a.writelines(out)
#         a.writelines(err)

i = 21
cmd = 'python'
with open(output_file,'w') as a:
    a.writelines("----WINDOW SIZE---\n")
    a.writelines("---- " + str(i) + "----\n")
for j in range(0,3):
    p = Popen([cmd,json[0][0],json[1][j],'json',json[2][j],str(i)], stdout=PIPE,stderr=PIPE)
    out,err = p.communicate()
    with open(output_file,'a') as a:
        a.writelines(out)
for k in range(0,3):
    p = Popen([cmd,read_mtx[0][0],read_mtx[1][k],read_mtx[2][k],read_mtx[3][k]], stdout=PIPE,stderr=PIPE)
    out,err = p.communicate()
    with open(output_file,'a') as a:
        a.writelines(out)
        a.writelines(err)
for j in range(0,3):
    p = Popen([cmd,encoded[0][0],encoded[1][j],'mtx',encoded[2][j],str(i)], stdout=PIPE,stderr=PIPE)
    out,err = p.communicate()
    with open(output_file,'a') as a:
        a.writelines(out)
        a.writelines(err)
# p = Popen([cmd,predict[0][0],predict[1][0],'bagClassify',str(i)], stdout=PIPE,stderr=PIPE)
p = Popen([cmd,predict[0][0],predict[1][0],'organelle',str(i)], stdout=PIPE,stderr=PIPE)
out,err = p.communicate()
with open(output_file,'a') as a:
    a.writelines(out)
    a.writelines(err)
    a.writelines("----PREDICTIONS---\n")    
for l in range(0,2):
    # p = Popen([cmd,predict[0][0],predict[1][l+1],'predictMTX'], stdout=PIPE,stderr=PIPE)
    p = Popen([cmd,predict[0][0],predict[1][l+1],'predictOrganelle',predict[2][l]], stdout=PIPE,stderr=PIPE)
    out,err = p.communicate()
    with open(output_file,'a') as a:
        a.writelines(out)
        a.writelines(err)
