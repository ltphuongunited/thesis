import numpy as np
import torch
import json
from tqdm import tqdm
# f = open('h36m/Human36M_subject1_smpl_param.json')
  
# returns JSON object as 
# a dictionary
# data = json.load(f)
# print(data['15'].keys())

# a = np.load('data/static_fits/h36m_fits_right.npy')
# # print(a.shape)
# np.save('h36m_fits.npy',a[:32])
# print(a[10000][:32])
# a = np.load('data_pymaf/static_Fits/h36m.npy')
# print(a[10000][:32])
# print(sum(a[0][:72] < 0))
data1 = np.load('data/dataset_extras/3dpw_test.npz')
# data2 = np.load('/home/tienthinh/phuong/SPIN/data/dataset_extras/h36m_mosh_train.npz')
# data3 = np.load('data/dataset_extras/h36m_valid_protocol2_newpath.npz')
# print(data1['imgname'][100])
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

np.savez('3dpw_valid.npz',
                    imgname=data1['imgname'][:32],
                    center=data1['center'][:32],
                    scale=data1['scale'][:32],
                    pose=data1['pose'][:32],
                    shape=data1['shape'][:32],
                    gender=data1['gender'][:32])

# e = []
# for i in tqdm(range(data1['pose'].shape[0])):
#     temp = np.concatenate((data1['pose'][i], data1['shape'][i]))
#     e.append(temp)
# np.save('h36m_fits.npy',e)