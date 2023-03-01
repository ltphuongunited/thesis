import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import numpy as np
import math
from utils.geometry import rot6d_to_rotmat

ANCESTOR_INDEX = [
    [],
    [0], 
    [0], 
    [0],
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 1, 4],
    [0, 2, 5],
    [0, 3, 6],
    [0, 1, 4, 7],
    [0, 2, 5, 8],
    [0, 3, 6, 9], 
    [0, 3, 6, 9], 
    [0, 3, 6, 9],
    [0, 3, 6, 9, 12],
    [0, 3, 6, 9, 13],
    [0, 3, 6, 9, 14],
    [0, 3, 6, 9, 13, 16],
    [0, 3, 6, 9, 14, 17],
    [0, 3, 6, 9, 13, 16, 18],
    [0, 3, 6, 9, 14, 17, 19],
    [0, 3, 6, 9, 13, 16, 18, 20],
    [0, 3, 6, 9, 14, 17, 19, 21]
]

class Bottleneck(nn.Module):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HMR_KTD(nn.Module):
    """ SMPL Iterative Regressor with ResNet50 backbone
    """

    def __init__(self, block, layers, smpl_mean_params):
        self.inplanes = 64
        super(HMR_KTD, self).__init__()
        npose = 24 * 6
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        self.joint_regs = nn.ModuleList()
        for joint_idx, ancestor_idx in enumerate(ANCESTOR_INDEX):
            regressor = nn.Linear(1024 + 6 * len(ancestor_idx), 6)
            nn.init.xavier_uniform_(regressor.weight, gain=0.01)
            self.joint_regs.append(regressor)

        self.decshape = nn.Linear(1024, 10)
        self.deccam = nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3):

        batch_size = x.shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        xf = self.avgpool(x4)
        xf = xf.view(xf.size(0), -1)

        xc = self.fc1(xf)
        xc = self.drop1(xc)
        xc = self.fc2(xc)
        xc = self.drop2(xc)
        pred_shape = self.decshape(xc)
        pred_cam = self.deccam(xc)
        
        pose = []
        for ancestor_idx, reg in zip(ANCESTOR_INDEX, self.joint_regs):
            ances = torch.cat([xc] + [pose[i] for i in ancestor_idx], dim=1)
            pose.append(reg(ances))

        pred_pose = torch.cat(pose, dim=1)
        
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        return pred_rotmat, pred_shape, pred_cam

def hmr_ktd(smpl_mean_params, pretrained=True, **kwargs):
    """ Constructs an HMR model with ResNet50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HMR_KTD(Bottleneck, [3, 4, 6, 3],  smpl_mean_params, **kwargs)
    if pretrained:
        resnet_imagenet = resnet.resnet50(pretrained=True)
        model.load_state_dict(resnet_imagenet.state_dict(),strict=False)
    return model

