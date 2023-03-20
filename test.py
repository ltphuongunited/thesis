import numpy as np
import torch
import json
from tqdm import tqdm
# f = open('h36m/Human36M_subject1_smpl_param.json')
  
# returns JSON object as 
# a dictionary
# data = json.load(f)
# print(data['15'].keys())

# a = np.load('h36m_fits.npy')
# print(a[10000][:10])
# a = np.load('data_pymaf/static_Fits/h36m.npy')
# print(a[10000][:10])
# print(sum(a[0][:72] < 0))
data1 = np.load('data/dataset_extras/lsp_dataset_test.npz')
# data2 = np.load('/home/tienthinh/phuong/SPIN/data/dataset_extras/h36m_mosh_train.npz')
# data3 = np.load('data/dataset_extras/h36m_valid_protocol2_newpath.npz')
print(data1['imgname'][100])
for key in data1.keys():
    print("variable name:", key          , end="  ")
    print("type: "+ str(data1[key].dtype) , end="  ")
    print("shape:"+ str(data1[key].shape))


# print('='*50)
# print(data2['imgname'][100])
# for key in data2.keys():
#     print("variable name:", key          , end="  ")
#     print("type: "+ str(data2[key].dtype) , end="  ")
#     print("shape:"+ str(data2[key].shape))

# print('='*50)
# print(data3['imgname'][100])
# for key in data3.keys():
#     print("variable name:", key          , end="  ")
#     print("type: "+ str(data3[key].dtype) , end="  ")
#     print("shape:"+ str(data3[key].shape))

# np.savez('h36m_valid_protocol2_newpath.npz',
#                     imgname=np.array(list(map(lambda x: x.split('/')[-1], data1['imgname']))),
#                     center=data1['center'],
#                     scale=data1['scale'],
#                     S=data1['S'])

# e = []
# for i in tqdm(range(data1['pose'].shape[0])):
#     temp = np.concatenate((data1['pose'][i], data1['shape'][i]))
#     e.append(temp)
# np.save('h36m_fits.npy',e)