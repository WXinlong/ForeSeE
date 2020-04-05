import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from tools.parse_arg_test import TestOptions
from data.load_dataset import CustomerDataLoader
from lib.models.depth_normal_model import DepthNormal
from lib.utils.net_tools import load_ckpt
from lib.utils.logging import setup_logging, SmoothedValue

logger = setup_logging(__name__)



# Add by users
pcd_folder = os.path.join(ROOT_DIR, 'output')
calib_fold = os.path.join(ROOT_DIR, 'datasets/KITTI_object/training/calib')
if not os.path.exists(pcd_folder):
    os.makedirs(pcd_folder)

def main():
    test_args = TestOptions().parse()
    test_args.thread = 1   # test code only supports thread = 1
    test_args.batchsize = 1  # test code only supports batchSize = 1

    data_loader = CustomerDataLoader(test_args)
    test_datasize = len(data_loader)
    logger.info('{:>15}: {:<30}'.format('test_data_size', test_datasize))
    # load model
    model = DepthNormal()
    # evaluate mode
    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    model.cuda()
    model = torch.nn.DataParallel(model)

    for i, data in enumerate(data_loader):
        out = model.module.inference(data)
        pred_depth = np.squeeze(out['b_fake']) * 80. # [h, w]
        pred_conf = np.squeeze(out['b_fake_conf']) # [c, h, w]

        # the image size has been padded to the size (385, 1243)
        pred_depth_crop = pred_depth[data['pad_raw'][0][0]:, data['pad_raw'][0][2]:]
        pred_conf_crop = pred_conf[:, data['pad_raw'][0][0]:, data['pad_raw'][0][2]:]

        sample_th = 0.15
        sample_mask = get_sample_mask(pred_conf_crop.cpu().numpy(), threshold=sample_th) # [h, w]

        #######################################################################################
        # add by users
        img_name = data['A_paths'][0].split('/')[-1][:-4]
        calib_name = img_name + '.txt'
        calib_dir = os.path.join(calib_fold, calib_name)
        camera_para = np.genfromtxt(calib_dir, delimiter=' ', skip_footer= 3, dtype=None)
        P3_0 = camera_para[3]
        P2_0 = camera_para[2]
        P3_2 = P3_0
        P3_2[4] -= P2_0[4]
        R0_rect = np.genfromtxt(calib_dir, delimiter=' ', skip_header=4, skip_footer=2)
        Tr_velo_to_cam0 = np.genfromtxt(calib_dir, delimiter=' ', skip_header=5, skip_footer=1)

        pcd_cam2 = reconstruct_3D(pred_depth_crop.cpu().numpy(), P3_2[3], P3_2[7], P3_2[1], P3_2[6])
        # Transfer points in cam2 coordinate to cam0 coordinate
        pcd_cam0 = pcd_cam2 - np.array([[[P2_0[4] / P2_0[1]]], [[P2_0[8] / P2_0[1]]], [[P2_0[12] / P2_0[1]]]])

        # Transfer points in cam0 coordinate to velo coordinate
        pcd_velo = transfer_points_in_cam0_to_velo(pcd_cam0, R0_rect, Tr_velo_to_cam0)

        rgb = data['A_raw'][0].cpu().numpy()

        save_ply(pcd_velo, rgb, os.path.join(pcd_folder, img_name) + '_sample.ply', sample_mask=sample_mask)
        #save_ply(pcd_cam2, rgb, os.path.join(pcd_folder, img_name) + '.ply')
        print('saved', img_name)
        #######################################################################################


##########################################################################
# others
def get_sample_mask(conf, threshold):
    max_conf = np.amax(conf, axis=0) # [h, w]
    print(max_conf.shape)
    return max_conf >= threshold

def transfer_points_in_cam0_to_velo(pcd_cam0, R_rect0, T_velo_cam0):
    pcd_cam0_3n = pcd_cam0.reshape((3, -1))
    R_rect0 = np.array(R_rect0[1:], dtype=np.float64).reshape((3, 3))
    R_rect0_inv = np.linalg.inv(R_rect0)

    # X_cam0_raw = (R_rect0)^-1 * X_cam0
    pcd_cam0_raw = np.matmul(R_rect0_inv, pcd_cam0_3n)

    T_velo_cam0 = np.array(T_velo_cam0[1:], dtype=np.float64).reshape((3, 4))
    R_velo_cam0 = T_velo_cam0[:, 0:3]
    T_velo_cam0 = T_velo_cam0[:, 3]
    R_cam0_velo = np.linalg.inv(R_velo_cam0)
    T_cam0_velo = -np.matmul(R_cam0_velo, T_velo_cam0)

    # X_velo = R*X_cam0 + T
    T_cam0_velo = T_cam0_velo[:, np.newaxis]
    pcd_velo_3n = np.matmul(R_cam0_velo, pcd_cam0_raw) + T_cam0_velo
    pcd_velo = pcd_velo_3n.reshape(3, pcd_cam0.shape[1], pcd_cam0.shape[2])
    return pcd_velo


def reconstruct_3D(depth, cu, cv, fx, fy):
    width = depth.shape[1]
    height = depth.shape[0]
    row = np.arange(0, width, 1)
    u = np.array([row for _ in np.arange(height)])
    col = np.arange(0, height, 1)
    v = np.array([col for _ in np.arange(width)])
    v = v.transpose(1, 0)

    x = (u - cu) * depth / fx
    y = (v - cv) * depth / fy
    z = depth

    x = x[np.newaxis, :, :]
    y = y[np.newaxis, :, :]
    z = z[np.newaxis, :, :]
    return np.concatenate([x, y, z], axis=0)

def save_ply(pcd, rgb, path, sample_mask=None):
    width = rgb.shape[1]
    height = rgb.shape[0]
    x = np.reshape(pcd[0], width * height)
    y = np.reshape(pcd[1], width * height)
    z = np.reshape(pcd[2], width * height)

    rgb = np.reshape(rgb, (width * height, 3))

    if sample_mask is not None:
        sample_mask = np.reshape(sample_mask, width * height)
        x = x[sample_mask]
        y = y[sample_mask]
        z = z[sample_mask]
        
        rgb = rgb[sample_mask, :]

    r = rgb[:, 2]
    g = rgb[:, 1]
    b = rgb[:, 0]
    r = np.squeeze(r)
    g = np.squeeze(g)
    b = np.squeeze(b)

    ply_head = 'ply\n' \
               'format ascii 1.0\n' \
               'element vertex %d\n' \
               'property float x\n' \
               'property float y\n' \
               'property float z\n' \
               'property uchar red\n' \
               'property uchar green\n' \
               'property uchar blue\n' \
               'end_header' % r.shape[0]
    # ---- Save ply data to disk
    np.savetxt(path, np.column_stack((x, y, z, r, g, b)), fmt="%f %f %f %d %d %d", header=ply_head, comments='')
##########################################################################

if __name__ == '__main__':
   main()
