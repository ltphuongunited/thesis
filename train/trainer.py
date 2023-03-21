import torch
import torch.nn as nn
import numpy as np
from torchgeometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis
import cv2

from datasets import MixedDataset, BaseDataset
from models import hmr, SMPL, hmr_ktd, hmr_hr, hmr_tfm, ktd
from smplify import SMPLify
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation
from utils.renderer import Renderer
from utils import BaseTrainer
from utils.pose_utils import reconstruction_error

import config
import constants
from .fits_dict import FitsDict
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchgeometry as tgm
from torch.optim.lr_scheduler import StepLR


class Trainer(BaseTrainer):
    
    def init_fn(self):
        self.train_ds = MixedDataset(self.options, ignore_3d=self.options.ignore_3d, is_train=True)
        self.model = hmr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        # self.model = hmr_ktd(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        # self.model = hmr_hr(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        # self.model = hmr_tfm(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        # self.model = ktd(config.SMPL_MEAN_PARAMS, pretrained=True).to(self.device)
        
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=self.options.lr,
                                          weight_decay=0)
        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=1)
        
        self.smpl = SMPL(config.SMPL_MODEL_DIR,
                         batch_size=self.options.batch_size,
                         create_transl=False).to(self.device)
        self.smpl_male = SMPL(model_path=config.SMPL_MODEL_DIR,
                                gender='male',
                                create_transl=False).to(self.device)
        self.smpl_female = SMPL(model_path=config.SMPL_MODEL_DIR,
                                gender='female',
                                create_transl=False).to(self.device)
        # Per-vertex loss on the shape
        self.criterion_shape = nn.L1Loss().to(self.device)
        # Keypoint (2D and 3D) loss
        # No reduction because confidence weighting needs to be applied
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        # Loss for SMPL parameter regression
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.models_dict = {'model': self.model}
        self.optimizers_dict = {'optimizer': self.optimizer}
        self.focal_length = constants.FOCAL_LENGTH

        # Initialize SMPLify fitting module
        self.smplify = SMPLify(step_size=1e-2, batch_size=self.options.batch_size, num_iters=self.options.num_smplify_iters, focal_length=self.focal_length)
        if self.options.pretrained_checkpoint is not None:
            self.load_pretrained(checkpoint_file=self.options.pretrained_checkpoint)

        # Load dictionary of fits
        self.fits_dict = FitsDict(self.options, self.train_ds)

        # Create renderer
        self.renderer = Renderer(focal_length=self.focal_length, img_res=self.options.img_res, faces=self.smpl.faces)

    def finalize(self):
        self.fits_dict.save()

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """ Compute 2D reprojection loss on the keypoints.
        The loss is weighted by the confidence.
        The available keypoints are different for each dataset.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
        The loss is weighted by the confidence.
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def shape_loss(self, pred_vertices, gt_vertices, has_smpl):
        """Compute per-vertex loss on the shape for the examples that SMPL annotations are available."""
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return self.criterion_shape(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas, has_smpl):
        pred_rotmat_valid = pred_rotmat[has_smpl == 1]
        gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
        pred_betas_valid = pred_betas[has_smpl == 1]
        gt_betas_valid = gt_betas[has_smpl == 1]
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas

    def train_step(self, input_batch):
        self.model.train()

        # Get data from the batch
        images = input_batch['img'] # input image
        gt_keypoints_2d = input_batch['keypoints'] # 2D keypoints
        gt_pose = input_batch['pose'] # SMPL pose parameters
        gt_betas = input_batch['betas'] # SMPL beta parameters
        gt_joints = input_batch['pose_3d'] # 3D pose
        has_smpl = input_batch['has_smpl'].byte() # flag that indicates whether SMPL parameters are valid
        has_pose_3d = input_batch['has_pose_3d'].byte() # flag that indicates whether 3D pose is valid
        is_flipped = input_batch['is_flipped'] # flag that indicates whether image was flipped during data augmentation
        rot_angle = input_batch['rot_angle'] # rotation angle used for data augmentation
        dataset_name = input_batch['dataset_name'] # name of the dataset the image comes from
        indices = input_batch['sample_index'] # index of example inside its dataset
        batch_size = images.shape[0]

        # Get GT vertices and model joints
        # Note that gt_model_joints is different from gt_joints as it comes from SMPL
        gt_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:,3:], global_orient=gt_pose[:,:3])
        gt_model_joints = gt_out.joints
        gt_vertices = gt_out.vertices

        # Get current best fits from the dictionary
        opt_pose, opt_betas = self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu())]
        opt_pose = opt_pose.to(self.device)
        opt_betas = opt_betas.to(self.device)
        opt_output = self.smpl(betas=opt_betas, body_pose=opt_pose[:,3:], global_orient=opt_pose[:,:3])
        opt_vertices = opt_output.vertices
        opt_joints = opt_output.joints


        # De-normalize 2D keypoints from [-1,1] to pixel space
        gt_keypoints_2d_orig = gt_keypoints_2d.clone()
        gt_keypoints_2d_orig[:, :, :-1] = 0.5 * self.options.img_res * (gt_keypoints_2d_orig[:, :, :-1] + 1)

        # Estimate camera translation given the model joints and 2D keypoints
        # by minimizing a weighted least squares loss
        gt_cam_t = estimate_translation(gt_model_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)

        opt_cam_t = estimate_translation(opt_joints, gt_keypoints_2d_orig, focal_length=self.focal_length, img_size=self.options.img_res)


        opt_joint_loss = self.smplify.get_fitting_loss(opt_pose, opt_betas, opt_cam_t,
                                                       0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                                       gt_keypoints_2d_orig).mean(dim=-1)

        # Feed images in the network to predict camera and SMPL parameters
        pred_rotmat, pred_betas, pred_camera = self.model(images)

        pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints

        # Convert Weak Perspective Camera [s, tx, ty] to camera translation [tx, ty, tz] in 3D given the bounding box size
        # This camera translation can be used in a full perspective projection
        pred_cam_t = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length/(self.options.img_res * pred_camera[:,0] +1e-9)],dim=-1)


        camera_center = torch.zeros(batch_size, 2, device=self.device)
        pred_keypoints_2d = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length,
                                                   camera_center=camera_center)
        # Normalize keypoints to [-1,1]
        pred_keypoints_2d = pred_keypoints_2d / (self.options.img_res / 2.)

        if self.options.run_smplify:

            # Convert predicted rotation matrices to axis-angle
            pred_rotmat_hom = torch.cat([pred_rotmat.detach().view(-1, 3, 3).detach(), torch.tensor([0,0,1], dtype=torch.float32,
                device=self.device).view(1, 3, 1).expand(batch_size * 24, -1, -1)], dim=-1)
            pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(batch_size, -1)
            # tgm.rotation_matrix_to_angle_axis returns NaN for 0 rotation, so manually hack it
            pred_pose[torch.isnan(pred_pose)] = 0.0

            # Run SMPLify optimization starting from the network prediction
            new_opt_vertices, new_opt_joints,\
            new_opt_pose, new_opt_betas,\
            new_opt_cam_t, new_opt_joint_loss = self.smplify(
                                        pred_pose.detach(), pred_betas.detach(),
                                        pred_cam_t.detach(),
                                        0.5 * self.options.img_res * torch.ones(batch_size, 2, device=self.device),
                                        gt_keypoints_2d_orig)
            new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)

            # Will update the dictionary for the examples where the new loss is less than the current one
            update = (new_opt_joint_loss < opt_joint_loss)
            

            opt_joint_loss[update] = new_opt_joint_loss[update]
            opt_vertices[update, :] = new_opt_vertices[update, :]
            opt_joints[update, :] = new_opt_joints[update, :]
            opt_pose[update, :] = new_opt_pose[update, :]
            opt_betas[update, :] = new_opt_betas[update, :]
            opt_cam_t[update, :] = new_opt_cam_t[update, :]


            self.fits_dict[(dataset_name, indices.cpu(), rot_angle.cpu(), is_flipped.cpu(), update.cpu())] = (opt_pose.cpu(), opt_betas.cpu())

        else:
            update = torch.zeros(batch_size, device=self.device).byte()

        # Replace extreme betas with zero betas
        opt_betas[(opt_betas.abs() > 3).any(dim=-1)] = 0.

        # Replace the optimized parameters with the ground truth parameters, if available
        # opt_vertices[has_smpl, :, :] = gt_vertices[has_smpl, :, :]
        # opt_cam_t[has_smpl, :] = gt_cam_t[has_smpl, :]
        # opt_joints[has_smpl, :, :] = gt_model_joints[has_smpl, :, :]
        # opt_pose[has_smpl, :] = gt_pose[has_smpl, :]
        # opt_betas[has_smpl, :] = gt_betas[has_smpl, :]
        opt_vertices[has_smpl.bool(), :, :] = gt_vertices[has_smpl.bool(), :, :]
        opt_cam_t[has_smpl.bool(), :] = gt_cam_t[has_smpl.bool(), :]
        opt_joints[has_smpl.bool(), :, :] = gt_model_joints[has_smpl.bool(), :, :]
        opt_pose[has_smpl.bool(), :] = gt_pose[has_smpl.bool(), :]
        opt_betas[has_smpl.bool(), :] = gt_betas[has_smpl.bool(), :]

        # Assert whether a fit is valid by comparing the joint loss with the threshold
        valid_fit = (opt_joint_loss < self.options.smplify_threshold).to(self.device)
        # Add the examples with GT parameters to the list of valid fits
        valid_fit = valid_fit | has_smpl

        opt_keypoints_2d = perspective_projection(opt_joints,
                                                  rotation=torch.eye(3, device=self.device).unsqueeze(0).expand(batch_size, -1, -1),
                                                  translation=opt_cam_t,
                                                  focal_length=self.focal_length,
                                                  camera_center=camera_center)


        opt_keypoints_2d = opt_keypoints_2d / (self.options.img_res / 2.)


        # Compute loss on SMPL parameters
        loss_regr_pose, loss_regr_betas = self.smpl_losses(pred_rotmat, pred_betas, opt_pose, opt_betas, valid_fit)

        # Compute 2D reprojection loss for the keypoints
        loss_keypoints = self.keypoint_loss(pred_keypoints_2d, gt_keypoints_2d,
                                            self.options.openpose_train_weight,
                                            self.options.gt_train_weight)

        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_joints, gt_joints, has_pose_3d)

        # Per-vertex loss for the shape
        loss_shape = self.shape_loss(pred_vertices, opt_vertices, valid_fit)

        # Compute total loss
        # The last component is a loss that forces the network to predict positive depth values
        loss = self.options.shape_loss_weight * loss_shape +\
               self.options.keypoint_loss_weight * loss_keypoints +\
               self.options.keypoint_loss_weight * loss_keypoints_3d +\
               self.options.pose_loss_weight * loss_regr_pose + self.options.beta_loss_weight * loss_regr_betas +\
               ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()
        loss *= 60


        # Do backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Pack output arguments for tensorboard logging
        output = {'pred_vertices': pred_vertices.detach(),
                  'opt_vertices': opt_vertices,
                  'pred_cam_t': pred_cam_t.detach(),
                  'opt_cam_t': opt_cam_t}
        losses = {'loss': loss.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regr_betas': loss_regr_betas.detach().item(),
                  'loss_shape': loss_shape.detach().item()}

        return output, losses

    def train_summaries(self, input_batch, output, losses):
        images = input_batch['img']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)

        pred_vertices = output['pred_vertices']
        opt_vertices = output['opt_vertices']
        pred_cam_t = output['pred_cam_t']
        opt_cam_t = output['opt_cam_t']
        images_pred = self.renderer.visualize_tb(pred_vertices, pred_cam_t, images)
        images_opt = self.renderer.visualize_tb(opt_vertices, opt_cam_t, images)
        self.summary_writer.add_image('pred_shape', images_pred, self.step_count)
        self.summary_writer.add_image('opt_shape', images_opt, self.step_count)
        for loss_name, val in losses.items():
            self.summary_writer.add_scalar(loss_name, val, self.step_count)
    

    def evaluate(self):
        self.model.eval()
        # Regressor for H36m joints
        J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
        
        shuffle = False
        batch_size = self.options.batch_size
        batch_size = 32
        dataset_name = self.options.eval_dataset
        result_file = None
        num_workers = self.options.num_workers
        device = self.device
        
        # Regressor for H36m joints
        J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()

        save_results = result_file is not None
        # Disable shuffling if you want to save the results
        if save_results:
            shuffle=False
        # Create dataloader for the dataset
        dataset = BaseDataset(self.options, dataset_name, is_train=False)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        
        # Pose metrics
        # MPJPE and Reconstruction error for the non-parametric and parametric shapes
        mpjpe = np.zeros(len(dataset))
        recon_err = np.zeros(len(dataset))
        pve = np.zeros(len(dataset))


        # Store SMPL parameters
        smpl_pose = np.zeros((len(dataset), 72))
        smpl_betas = np.zeros((len(dataset), 10))
        smpl_camera = np.zeros((len(dataset), 3))
        pred_joints = np.zeros((len(dataset), 17, 3))
        action_idxes = {}
        idx_counter = 0


        eval_pose = False

        # Choose appropriate evaluation for each dataset
        if dataset_name == 'h36m-p1' or dataset_name == 'h36m-p2' or dataset_name == 'h36m-p2-mosh' \
        or dataset_name == '3dpw' or dataset_name == 'mpi-inf-3dhp' or dataset_name == '3doh50k':
            eval_pose = True

        joint_mapper_h36m = constants.H36M_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.H36M_TO_J14
        joint_mapper_gt = constants.J24_TO_J17 if dataset_name == 'mpi-inf-3dhp' else constants.J24_TO_J14
        # Iterate over the entire dataset
        cnt = 0
        results_dict = {'id': [], 'pred': [], 'pred_pa': [], 'gt': []}
        for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=len(data_loader))):
            # Get ground truth annotations from the batch
            gt_pose = batch['pose'].to(device)
            gt_betas = batch['betas'].to(device)
            gt_smpl_out = self.smpl(betas=gt_betas, body_pose=gt_pose[:, 3:], global_orient=gt_pose[:, :3])
            gt_vertices_nt = gt_smpl_out.vertices
            images = batch['img'].to(device)
            gender = batch['gender'].to(device)
            curr_batch_size = images.shape[0]

            if save_results:
                s_id = np.array([int(item.split('/')[-3][-1]) for item in batch['imgname']]) * 10000
                s_id += np.array([int(item.split('/')[-1][4:-4]) for item in batch['imgname']])
                results_dict['id'].append(s_id)

            if dataset_name == 'h36m-p2':
                action = [im_path.split('/')[-1].split('.')[0].split('_')[1] for im_path in batch['imgname']]
                for act_i in range(len(action)):

                    if action[act_i] in action_idxes:
                        action_idxes[action[act_i]].append(idx_counter + act_i)
                    else:
                        action_idxes[action[act_i]] = [idx_counter + act_i]
                idx_counter += len(action)

            with torch.no_grad():
                pred_rotmat, pred_betas, pred_camera = self.model(images)

                pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
                pred_vertices = pred_output.vertices

            if save_results:
                rot_pad = torch.tensor([0,0,1], dtype=torch.float32, device=device).view(1,3,1)
                rotmat = torch.cat((pred_rotmat.view(-1, 3, 3), rot_pad.expand(curr_batch_size * 24, -1, -1)), dim=-1)
                pred_pose = tgm.rotation_matrix_to_angle_axis(rotmat).contiguous().view(-1, 72)
                smpl_pose[step * batch_size:step * batch_size + curr_batch_size, :] = pred_pose.cpu().numpy()
                smpl_betas[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_betas.cpu().numpy()
                smpl_camera[step * batch_size:step * batch_size + curr_batch_size, :]  = pred_camera.cpu().numpy()

            # 3D pose evaluation
            if eval_pose:
                # Regressor broadcasting
                J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(device)
                # Get 14 ground truth joints
                if 'h36m' in dataset_name or 'mpi-inf' in dataset_name or '3doh50k' in dataset_name:
                    gt_keypoints_3d = batch['pose_3d'].cuda()
                    gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_gt, :-1]
                # For 3DPW get the 14 common joints from the rendered shape
                else:
                    gt_vertices = self.smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                    gt_vertices_female = self.smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                    gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                    gt_keypoints_3d = torch.matmul(J_regressor_batch, gt_vertices)
                    gt_pelvis = gt_keypoints_3d[:, [0],:].clone()
                    gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
                    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis

                if '3dpw' in dataset_name:
                    per_vertex_error = torch.sqrt(((pred_vertices - gt_vertices) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                else:
                    per_vertex_error = torch.sqrt(((pred_vertices - gt_vertices_nt) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                pve[step * batch_size:step * batch_size + curr_batch_size] = per_vertex_error

                # Get 14 predicted joints from the mesh
                pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
                if save_results:
                    pred_joints[step * batch_size:step * batch_size + curr_batch_size, :, :]  = pred_keypoints_3d.cpu().numpy()
                pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
                pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
                pred_keypoints_3d = pred_keypoints_3d - pred_pelvis

                # Absolute error (MPJPE)
                error = torch.sqrt(((pred_keypoints_3d - gt_keypoints_3d) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                mpjpe[step * batch_size:step * batch_size + curr_batch_size] = error

                # Reconstuction_error
                r_error, pred_keypoints_3d_pa = reconstruction_error(pred_keypoints_3d.cpu().numpy(), gt_keypoints_3d.cpu().numpy(), reduction=None)
                recon_err[step * batch_size:step * batch_size + curr_batch_size] = r_error


        pa_mpjpe = 1000 * recon_err.mean()
        tqdm.write('PA-MPJPE on 3DPW: ' + str(pa_mpjpe))
        return pa_mpjpe