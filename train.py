import numpy as np
import torch
import random

random.seed(10)
torch.manual_seed(10)
np.random.seed(10)

from utils import TrainOptions
from train import Trainer

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    trainer = Trainer(options)
    trainer.train()
