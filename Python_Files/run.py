import sys
import subprocess

import sys
import os
from subprocess import Popen, PIPE


encoded = [
            ['encode.py'],
            ['../Data_Files/test_files/pssmTrain.npz','../Data_Files/test_files/pssmTest.npz'],
            ['../Data_Files/test_files/encTrain','../Data_Files/test_files/encTest']]

predict = [
            ['predict_model.py'],
            ['../Data_Files/test_files/encTrain.npz','../Data_Files/test_files/encTest.npz']]
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
with open('result3.txt','a') as a:
    a.writelines("----WINDOW SIZE---\n")
    a.writelines("---- " + str(i) + "----\n")
for j in range(0,2):
    p = Popen([cmd,encoded[0][0],encoded[1][j],'mtx',encoded[2][j],str(i)], stdout=PIPE,stderr=PIPE)
    out,err = p.communicate()
    with open('result3.txt','a') as a:
        a.writelines(out)
        a.writelines(err)
p = Popen([cmd,predict[0][0],predict[1][0],'bagClassify',str(i)], stdout=PIPE,stderr=PIPE)
out,err = p.communicate()
with open('result3.txt','a') as a:
    a.writelines(out)
    a.writelines(err)
