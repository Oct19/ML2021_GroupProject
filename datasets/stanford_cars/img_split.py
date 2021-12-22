import numpy as np
import os
from scipy.io import loadmat

list_path = 'cars_test_annos_withlabels.mat'
list_mat = loadmat(list_path)
num_inst = len(list_mat['annotations']['fname'][0])
clstable = [[] for _ in range(196)]
test_list = []

for i in range(num_inst):
    imgfname = list_mat['annotations']['fname'][0][i].item()
    label = list_mat['annotations']['class'][0][i].item() - 1
    test_list.append(imgfname + ' ' + str(int(label))+'\n')

val_list = []
train_list = []
'''
for i in range(196):
    for j, ss in enumerate(clstable[i]):
        test_list.append(ss)
     
        if j < 6:
            val_list.append(ss)
        else:
            train_list.append(ss)
            '''

with open('test_list.txt','w') as f:
    f.writelines(test_list)
    f.close()



'''
test_list = []
for i in range(num_inst):
    imgfname = list_mat['annotations']['fname'][0][i].item()
    test_list.append(imgfname+'\n')
with open('test_list.txt','w') as f:
    f.writelines(test_list)
    f.close()
'''






