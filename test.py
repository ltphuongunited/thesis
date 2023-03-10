import numpy as np
import torch
import json

# f = open('h36m/Human36M_subject1_smpl_param.json')
  
# returns JSON object as 
# a dictionary
# data = json.load(f)
# print(data['15'].keys())

# a = np.load('data/static_fits/coco_fits.npy')
# print(a.shape)
# print(sum(a[0][:72] < 0))
data1 = np.load('data/dataset_extras/3dpw_test.npz')
# data2 = np.load('data/dataset_extras/coco_2014_train.npz')
# data3 = np.load('data/dataset_extras/coco_2014_train_cliff.npz')

for key in data1.keys():
    print("variable name:", key          , end="  ")
    print("type: "+ str(data1[key].dtype) , end="  ")
    print("shape:"+ str(data1[key].shape))

# print('*'*50)

# for key in data2.keys():
#     print("variable name:", key          , end="  ")
#     print("type: "+ str(data2[key].dtype) , end="  ")
#     print("shape:"+ str(data2[key].shape))

# print('*'*50)

# for key in data3.keys():
#     print("variable name:", key          , end="  ")
#     print("type: "+ str(data3[key].dtype) , end="  ")
#     print("shape:"+ str(data3[key].shape))

# np.savez('data/dataset_extras/coco_2014_train_cliff.npz',
#                     imgname=data1['imgname'],
#                     center=data1['center'],
#                     scale=data1['scale'],
#                     part=data1['part'],
#                     annot_id=data1['annot_id'],
#                     pose=data1['pose'],
#                     shape=data1['shape'],
#                     has_smpl=data1['has_smpl'],
#                     global_t=data1['global_t'],
#                     focal_l=data1['focal_l'],
#                     S=data1['S'],
#                     openpose=data2['openpose'])
