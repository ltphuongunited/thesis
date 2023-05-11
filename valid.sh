#!/bin/bash

# HMR
# echo "Running HMR..."
# echo "H36m"
# python3 eval.py --checkpoint=logs/hmr/checkpoints/epoch_28_136556_56-30.pt --dataset=h36m-p1 --log_freq=1000
# echo "3DPW"
# python3 eval.py --checkpoint=logs/hmr/checkpoints/epoch_28_136556_56-30.pt --dataset=3dpw --log_freq=1000
# echo "MPI"
# python3 eval.py --checkpoint=logs/hmr/checkpoints/epoch_28_136556_56-30.pt --dataset=mpi-inf-3dhp --log_freq=1000

# # KTD
# echo "Running KTD..."
# echo "H36m"
# python3 eval.py --checkpoint=logs/hmr_ktd/checkpoints/epoch_28_136556_54-41_1e-05.pt --dataset=h36m-p1 --log_freq=1000
# echo "3DPW"
# python3 eval.py --checkpoint=logs/hmr_ktd/checkpoints/epoch_28_136556_54-41_1e-05.pt --dataset=3dpw --log_freq=1000
# echo "MPI"
# python3 eval.py --checkpoint=logs/hmr_ktd/checkpoints/epoch_28_136556_54-41_1e-05.pt --dataset=mpi-inf-3dhp --log_freq=1000

# # TFM
# echo "Running TFM..."
# echo "H36m"
# python3 eval.py --checkpoint=logs/hmr_tfm/checkpoints/epoch_17_80000_56-40_5e-07.pt --dataset=h36m-p1 --log_freq=1000
# echo "3DPW"
# python3 eval.py --checkpoint=logs/hmr_tfm/checkpoints/epoch_17_80000_56-40_5e-07.pt --dataset=3dpw --log_freq=1000
# echo "MPI"
# python3 eval.py --checkpoint=logs/hmr_tfm/checkpoints/epoch_17_80000_56-40_5e-07.pt --dataset=mpi-inf-3dhp --log_freq=1000

# # ViT
# echo "Running ViT..."
# echo "H36m"
# python3 eval.py --checkpoint=logs/hmr_vit/checkpoints/epoch_37_288575_52-23_5e-06.pt --dataset=h36m-p1 --log_freq=1000
# echo "3DPW"
# python3 eval.py --checkpoint=logs/hmr_vit/checkpoints/epoch_37_288575_52-23_5e-06.pt --dataset=3dpw --log_freq=1000
# echo "MPI"
# python3 eval.py --checkpoint=logs/hmr_vit/checkpoints/epoch_37_288575_52-23_5e-06.pt --dataset=mpi-inf-3dhp --log_freq=1000

# HR
echo "Running HR..."
echo "H36m"
python3 eval.py --checkpoint=logs/hmr_hr_old/checkpoints/epoch_44_429220_59-83_1e-07.pt --dataset=h36m-p1 --log_freq=1000
echo "3DPW"
python3 eval.py --checkpoint=logs/hmr_hr_old/checkpoints/epoch_44_429220_59-83_1e-07.pt --dataset=3dpw --log_freq=1000
echo "MPI"
python3 eval.py --checkpoint=logs/hmr_hr_old/checkpoints/epoch_44_429220_59-83_1e-07.pt --dataset=mpi-inf-3dhp --log_freq=1000
