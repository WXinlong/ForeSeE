import os
import os.path
import sys
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.core.config import cfg
from lib.utils.logging import setup_logging
from lib.utils.obj_utils import read_labels, rois2mask, rois2mask_shrink, rois2boxlist
logger = setup_logging(__name__)

from IPython import embed

class KITTIGtDataset():
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_anno = os.path.join(opt.dataroot, 'annotations', opt.phase + '_annotations.json')
        self.A_paths, self.B_paths, self.AB_anno, self.rois_paths = self.getData()
        self.data_size = len(self.AB_anno)
        self.depth_normalize = 255. * 80.
        self.ignore_cate_list = ['Person_sitting', 'Misc', 'DontCare']
        self.uniform_size = (385, 1243)

    def getData(self):
        with open(self.dir_anno, 'r') as load_f:
            AB_anno = json.load(load_f)
        A_list = [os.path.join(self.opt.dataroot, AB_anno[i]['rgb_path']) for i in range(len(AB_anno))]
        B_list = [os.path.join(self.opt.dataroot, AB_anno[i]['depth_path']) for i in range(len(AB_anno))]
        rois_list = [os.path.join(self.opt.dataroot, AB_anno[i]['rois_path']) for i in range(len(AB_anno))]
        logger.info('Loaded Kitti data!')
        return A_list, B_list, AB_anno, rois_list

    def __getitem__(self, anno_index):

        data = self.online_aug_val_test(anno_index)
        return data

    def online_aug_val_test(self, idx):
        A_path = self.A_paths[idx]
        B_path = self.B_paths[idx]
        rois_path = self.rois_paths[idx]

        A = cv2.imread(A_path, -1)  # [H, W, C] C:bgr

        #B = np.zeros((A.shape[0], A.shape[1]), dtype=np.float32 )
        B = cv2.imread(B_path, -1) / self.depth_normalize

        rois = read_labels(rois_path, ignore_cate=self.ignore_cate_list) # list of instances of class ObjectLabel, see obj_utils.py
        raw_boxlist = rois2boxlist(rois, (A.shape[1], A.shape[0]))

        flip_flg, resize_size, crop_size, pad, resize_ratio = self.set_flip_pad_reshape_crop(A)

        A_crop = np.pad(A, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant', constant_values=(0, 0))
        B_crop = np.pad(B, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant', constant_values=(0, 0))

        raw_boxlist.bbox[:, 0::2] += pad[2]
        raw_boxlist.bbox[:, 1::2] += pad[0]
        boxes = raw_boxlist.bbox        

        A_crop = A_crop.transpose((2, 0, 1))
        B_crop = B_crop[np.newaxis, :, :]
        # change the color channel, bgr->rgb
        A_crop = A_crop[::-1, :, :]
        # to torch, normalize
        A_crop = self.scale_torch(A_crop, 255.)
        B_crop = self.scale_torch(B_crop, 1.0)

        data = {'A': A_crop, 'B': B_crop, 'bbox': boxes,
                'A_raw': A, 'B_raw': B, 'A_paths': A_path, 'B_paths': B_path, 'pad_raw': np.array(pad)}

        return data

    def set_flip_pad_reshape_crop(self, A):
        flip_flg = False

        # pad
        pad_height = self.uniform_size[0] - A.shape[0]
        pad_width = self.uniform_size[1] - A.shape[1]
        pad = [pad_height, 0, pad_width, 0] # [up, down, left, right]

        # reshape
        resize_ratio = 1.0
        resize_size = [int((A.shape[0]+pad[0]+pad[1]) * resize_ratio + 0.5),
                       int((A.shape[1]+pad[2]+pad[3]) * resize_ratio + 0.5)]

        # crop
        start_y = 0 if resize_size[0] < (50 + pad[0] + pad[1]) * resize_ratio + cfg.CROP_SIZE[0]\
            else np.random.randint(int((50 + pad[0]) * resize_ratio), resize_size[0] - cfg.CROP_SIZE[0] - pad[1] * resize_ratio)
        start_x = np.random.randint(pad[2] * resize_ratio, resize_size[1] - cfg.CROP_SIZE[1] - pad[3] * resize_ratio)
        crop_height = cfg.CROP_SIZE[0]
        crop_width = cfg.CROP_SIZE[1]
        crop_size = [start_x, start_y, crop_width, crop_height]
        return flip_flg, resize_size, crop_size, pad, resize_ratio

    def flip_pad_reshape_crop(self, img, flip, resize_size, crop_size, pad, pad_value=0):
        if len(img.shape) == 1:
            return img
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                       constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                             constant_values=(pad_value, pad_value))
        # Resize the raw image
        img_resize = cv2.resize(img_pad, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
        # Crop the resized image
        img_crop = img_resize[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]
        return img_crop

    def depth_to_class(self, depth):
        """
        Transfer 1-channel depth to 1-channel depth in n depth ranges
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: classes [1, h, w]
        """
        invalid_mask = depth < 0.
        depth[depth < cfg.DATA.DATA_MIN] = cfg.DATA.DATA_MIN
        depth[depth > cfg.DATA.DATA_MAX] = cfg.DATA.DATA_MAX
        classes = ((torch.log10(depth) - cfg.DATA.DATA_MIN_LOG) / cfg.DATA.DEPTH_RANGE_INTERVAL).to(torch.int)
        classes[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        classes[classes == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        return classes

    def scale_torch(self, img, scale):
        # scale image
        img = img.astype(np.float32)
        img /= scale
        img = torch.from_numpy(img.copy())
        if img.size(0) == 3:
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        else:
            img = transforms.Normalize((0,), (1,))(img)
        return img


    def __len__(self):
        return self.data_size

    def name(self):
        return 'NYUDepthV2Dataset'

if __name__ == "__main__":
    class test_opt:
        def __init__(self):
            self.phase = "tongji"
            self.dataroot = "../datasets/KITTI_object"
        
    opt = test_opt()
    dataset = KITTIPredictionDataset()
    dataset.initialize(opt) 
    
    #idx = 0
    #data = dataset.__getitem__(idx)

    #embed()

    # tongji
    num_fg = 0
    num_bg = 0
    num_all = 0
    num_iter =  len(dataset)
    #num_iter =  20

    fg_list = []
    bg_list = []

    fg_hist_cnt = 0
    bg_hist_cnt = 0

    def cal_grad(B):
        B = B[0,...]
        H, W = B.shape
        #for i in range(H):
        #    for j in range(W):

        B_cv2 = cv2.fromarray(B)
        B_lap = cv2.Laplacian(img,cv2.CV_64F)    

        return np.array(B_lap)

    for i in range(num_iter):
        print(i)
        #idx = np.random.randint(0, len(dataset)) 
        data = dataset.__getitem__(i)
        
        bbox = data['bbox']
        B = data['B'] * 80
        
        num_box = bbox.shape[0]
        rois_mask = np.zeros_like(B)
        for j in range(num_box):
            
            box = bbox[j]
            x1, y1, x2, y2 = map(int, box)
            rois_mask[0, y1:y2, x1:x2] = 1
        rois_mask = torch.from_numpy(rois_mask.astype(np.uint8))
       
        mask_0 = B != 0 
        cur_fg_list = torch.masked_select(B, mask_0 & rois_mask) 
        cur_bg_list = torch.masked_select(B, mask_0 & (1 - rois_mask))

        cur_fg_list = list(cur_fg_list.numpy())
        cur_bg_list = list(cur_bg_list.numpy())

        cur_fg_hist_cnt, bins = np.histogram(cur_fg_list, bins=10, range=(0,80))
        cur_bg_hist_cnt, bins = np.histogram(cur_bg_list, bins=10, range=(0,80))

        fg_hist_cnt += cur_fg_hist_cnt
        bg_hist_cnt += cur_bg_hist_cnt

        #fg_list.extend(list(cur_fg_list.numpy()))
        #bg_list.extend(list(cur_bg_list.numpy()))

    print(fg_hist_cnt)
    print(bg_hist_cnt)

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)
    
    center = (bins[:-1] + bins[1:]) / 2
    width = 0.3 * (bins[1] - bins[0])

    fg_hist_frq = 1. * fg_hist_cnt / fg_hist_cnt.sum()
    bg_hist_frq = 1. * bg_hist_cnt / bg_hist_cnt.sum()

    print("fg_hist_frq: {}".format(fg_hist_frq))
    print("bg_hist_frq: {}".format(bg_hist_frq))

    labels = [str(a) for a in range(8, 88, 8)]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rect1 = ax.bar(x-width/2, fg_hist_frq, color='salmon', width=width, label="Foreground")
    rect2 = ax.bar(x+width/2, bg_hist_frq, color="darkseagreen", width=width, label="Background")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()

    embed()    

    """
    plt.subplot(1, 2, 1)
    plt.bar(center, fg_hist_frq, align='center', color='salmon')
    plt.xlim((0,80))
    plt.ylim((0,0.2))

    plt.subplot(1, 2, 2)
    plt.bar(center, bg_hist_frq, align='center', color='salmon')
    plt.xlim((0,80))
    plt.ylim((0,0.2))

    plt.show()
    
    embed()
    """
