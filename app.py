import gradio as gr
import torch

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import json
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
import os.path as osp
from matplotlib.image import imsave
from skimage.transform import resize
from torchvision.transforms import Normalize

# from core.cfgs import cfg, parse_args
from models import hmr, SMPL, hmr_tfm, hmr_hr, Token3d, hmr_ktd
import config
import constants
from datasets.inference import Inference
from utils.renderer_add import PyRenderer
from utils.imutils import crop
from utils.pose_tracker import run_posetracker
from utils.demo_utils import (
    convert_crop_cam_to_orig_img,
    prepare_rendering_results
)

MIN_NUM_FRAMES = 1

def run_image_demo(input_image):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    image_folder = 'input_image/'
    cv2.imwrite(os.path.join(image_folder, f'input.png'), input_image)
    num_frames = len(os.listdir(image_folder))
    img_shape = cv2.imread(osp.join(image_folder, os.listdir(image_folder)[0])).shape

    output_path = os.path.join('output/', osp.split(image_folder)[-1])

    os.makedirs(output_path, exist_ok=True)

    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]

    total_time = time.time()

    # ========= Run tracking ========= #
    bbox_scale = 1.0
    # run multi object tracker
    mot = MPT(
        device=device,
        batch_size=12,
        display=False,
        detector_type='yolo',
        output_format='dict',
        yolo_img_size=416,
    )
    tracking_results = mot(image_folder)

    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Define model ========= #
    name = 'logs/hmr_vit/checkpoints/epoch_37_288575_52-23_5e-06.pt'
    model = Token3d(config.SMPL_MEAN_PARAMS)

    model = model.to(device)
    checkpoint = torch.load(name)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{name}\"')

    # ========= Run pred on each person ========= #
    image_file_names = None
    print(f'Running reconstruction on each tracklet...')
    pred_time = time.time()
    pred_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None
        
        bboxes = tracking_results[person_id]['bbox']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )


        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=8, num_workers=8)
        smpl = SMPL(config.SMPL_MODEL_DIR,
                    batch_size=8,
                    create_transl=False).to(device)
        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))
                batch = batch.to(device)
                batch_size = batch.shape[0]
                seqlen = 1
                rotmat, betas, camera = model(batch)
                output = smpl(betas=betas, body_pose=rotmat[:,1:], global_orient=rotmat[:,0].unsqueeze(1), pose2rot=False)
                out_vertices = output.vertices
                out_joints = output.joints
                
                pred_cam.append(camera.reshape(batch_size * seqlen, -1))
                pred_verts.append(out_vertices.reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(rotmat.reshape(batch_size * seqlen, -1))
                pred_betas.append(betas.reshape(batch_size * seqlen, -1))
                pred_joints3d.append(out_joints.reshape(batch_size * seqlen, -1, 3))

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        pred_results[person_id] = output_dict

    del model

    renderer = PyRenderer(resolution=(orig_width, orig_height))

    output_img_folder = os.path.join(output_path, osp.split(image_folder)[-1] + '_output')
    os.makedirs(output_img_folder, exist_ok=True)

    print(f'Rendering output video, writing frames to {output_img_folder}')

    # prepare results for rendering
    frame_results = prepare_rendering_results(pred_results, num_frames)

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])
    color_type = 'pink'

    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        render_ratio = 1
        if render_ratio != 1:
            img = resize(img, (int(img.shape[0] * render_ratio), int(img.shape[1] * render_ratio)), anti_aliasing=True)
            img = (img * 255).astype(np.uint8)

        raw_img = img.copy()

        side_img = np.zeros_like(img)


        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mesh_filename = None

            # file 3D
            mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:04d}')
            os.makedirs(mesh_folder, exist_ok=True)
            mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

            img = renderer(
                frame_verts,
                img=img,
                cam=frame_cam,
                color_type=color_type,
                mesh_filename=mesh_filename
            )

            side_img = renderer(
                frame_verts,
                img=side_img,
                cam=frame_cam,
                color_type=color_type,
                angle=270,
                axis=[0,1,0],
            )
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mesh_filename
        
input_image = gr.inputs.Image(label="Input Image")
output_image = gr.outputs.Image(type="numpy", label="Output Image").style(height=300)
output_3d_model = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model").style(height=200)

interface = gr.Interface(
    fn=run_image_demo,
    inputs=input_image,
    outputs=[output_image, output_3d_model]
)

interface.launch()
