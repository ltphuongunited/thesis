import numpy as np
import torch
import random
import config
random.seed(10)
torch.manual_seed(10)
np.random.seed(10)

from models.tokenpose import Token3d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
model = Token3d(smpl_mean_params=config.SMPL_MEAN_PARAMS, pretrained=True).to(device)

x = torch.randn(32, 3, 224, 224).to(device)

pred_rotmat, pred_shape, pred_cam = model(x)
# print(pred_rotmat.shape,pred_shape.shape,pred_cam.shape)

