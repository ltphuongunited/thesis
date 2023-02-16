"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

H36M_ROOT = 'datasets/all/H36M'
LSP_ROOT = 'datasets/all/lsp_dataset'
LSP_ORIGINAL_ROOT = 'datasets/all/lsp_dataset_original'
LSPET_ROOT = 'datasets/all/hr-lspet'
MPII_ROOT = 'datasets/all/mpii_human_pose_v1'
COCO_ROOT = 'datasets/all/coco'
MPI_INF_3DHP_ROOT = 'datasets/all/mpi_inf_3dhp'
PW3D_ROOT = 'datasets/all/3dpw'
UPI_S1H_ROOT = 'datasets/all/upi-s1h'

# Output folder to save test/train npz files
DATASET_NPZ_PATH = 'data/dataset_extras'

# Output folder to store the openpose detections
# This is requires only in case you want to regenerate 
# the .npz files with the annotations.
OPENPOSE_PATH = 'datasets/openpose'

# Path to test/train npz files
DATASET_FILES = [ {'h36m-p1': join(DATASET_NPZ_PATH, 'h36m_valid_protocol1.npz'),
                   'h36m-p2': join(DATASET_NPZ_PATH, 'h36m_valid_protocol2.npz'),
                   'lsp': join(DATASET_NPZ_PATH, 'lsp_dataset_test.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_valid.npz'),
                   '3dpw': join(DATASET_NPZ_PATH, '3dpw_test.npz'),
                  },

                  {'h36m': join(DATASET_NPZ_PATH, 'h36m_train.npz'),
                   'lsp-orig': join(DATASET_NPZ_PATH, 'lsp_dataset_original_train.npz'),
                   'mpii': join(DATASET_NPZ_PATH, 'mpii_train.npz'),
                   'coco': join(DATASET_NPZ_PATH, 'coco_2014_train.npz'),
                   'lspet': join(DATASET_NPZ_PATH, 'hr-lspet_train.npz'),
                   'mpi-inf-3dhp': join(DATASET_NPZ_PATH, 'mpi_inf_3dhp_train.npz')
                  }
                ]

DATASET_FOLDERS = {'h36m': H36M_ROOT,
                   'h36m-p1': H36M_ROOT,
                   'h36m-p2': H36M_ROOT,
                   'lsp-orig': LSP_ORIGINAL_ROOT,
                   'lsp': LSP_ROOT,
                   'lspet': LSPET_ROOT,
                   'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'mpii': MPII_ROOT,
                   'coco': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   'upi-s1h': UPI_S1H_ROOT,
                }

CUBE_PARTS_FILE = 'data/cube_parts.npy' # for evaluation LSP
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/J_regressor_extra.npy' # Joints regressor for joints or landmarks that are not included in the standard set of SMPL joints.
JOINT_REGRESSOR_H36M = 'data/J_regressor_h36m.npy' # Joints regressor reflecting the Human3.6M joints. Used for evaluation.
VERTEX_TEXTURE_FILE = 'data/vertex_texture.npy' # for evaluation LSP
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/smpl_mean_params.npz'
SMPL_MODEL_DIR = 'data/smpl'
