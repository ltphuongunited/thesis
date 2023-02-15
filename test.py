import config
import constants
import numpy as np

joint = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
print(np.nonzero(joint))