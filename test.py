import numpy as np

data = np.load('cliffGT_v1/coco2014part_cliffGT.npz')
a=5
# print(data['S'][a:a+3])
for key in data.keys():
    print("variable name:", key          , end="  ")
    print("type: "+ str(data[key].dtype) , end="  ")
    print("shape:"+ str(data[key].shape))

print('*'*50)
data = np.load('data/dataset_extras/coco_2014_train.npz')
# print(data['S'][a:a+3])
for key in data.keys():
    print("variable name:", key          , end="  ")
    print("type: "+ str(data[key].dtype) , end="  ")
    print("shape:"+ str(data[key].shape))
# # print(data.shape)
# a = data['imgname']
# a = list(map(lambda x: 'datasets/all/mpi_inf_3dhp/' + x, a))
# np.save('image_mpi.npy', a)

# a = np.load('image_mpi.npy')
# temp = 'datasets/all/mpi_inf_3dhp/S1/Seq1/imageFrames/video_0/frame_000001.jpg'

# print(a[0])
# print(temp)
# print(temp in a)