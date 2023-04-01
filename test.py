import numpy as np
import torch
import json
from tqdm import tqdm
import pickle


# with open('data/gmm_08.pkl', 'rb') as f:
#     gmm = pickle.load(f, encoding='latin1')

# for key in gmm.keys():
#     print(gmm[key].shape)
# f = open('h36m/Human36M_subject1_smpl_param.json')
  
# returns JSON object as 
# a dictionary
# data = json.load(f)
# print(data['15'].keys())

# a = np.load('data/static_fits/h36m_fits_code.npy')
# print(a.shape)
# np.save('h36m_fits.npy',a[:32])
# print(a[10000][:32])
# a = np.load('data_pymaf/static_Fits/h36m.npy')
# print(a[10000][:32])
# print(sum(a[0][:72] < 0))
data1 = np.load('data/dataset_extras/coco_2014_train_cliff_no.npz')
# data2 = np.load('data/dataset_extras/mpi_inf_3dhp_train.npz')
# data3 = np.load('data/dataset_extras/h36m_valid_protocol2_newpath.npz')
# print(sum(data1['has_smpl']))
for key in data1.keys():
    print("variable name:", key, end="  ")
    print("type: "+ str(data1[key].dtype) , end="  ")
    print("shape:"+ str(data1[key].shape))


# print('='*50)
# print(sum(data2['has_smpl']))
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

# np.savez('3dpw_train.npz',
#                     imgname=data1['imgname'],
#                     center=data1['center'],
#                     scale=data1['scale'],
#                     pose=data1['pose'],
#                     shape=data1['shape'],
#                     part=data1['part'],
#                     S=data1['S'],
#                     gshape=data1['gshape'],
#                     gender=data1['gender'],
#                     has_smpl=np.ones(len(data1['gender'])))

# e = []
# for i in tqdm(range(data1['pose'].shape[0])):
#     temp = np.concatenate((data1['pose'][i], data1['shape'][i]))
#     e.append(temp)
# np.save('mpii_fits.npy',e)