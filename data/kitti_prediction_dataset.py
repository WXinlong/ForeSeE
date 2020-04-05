import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
from lib.core.config import cfg
import cv2
import json
from lib.utils.logging import setup_logging
from lib.utils.obj_utils import read_labels, rois2mask, rois2mask_shrink, rois2boxlist
logger = setup_logging(__name__)


class KITTIPredictionDataset():
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

        B = np.zeros((A.shape[0], A.shape[1]), dtype=np.float32 )

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
