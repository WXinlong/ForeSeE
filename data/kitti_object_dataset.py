import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
from lib.core.config import cfg
import cv2
import json
from lib.utils.logging import setup_logging
from lib.utils.obj_utils import read_labels, rois2mask, rois2mask_shrink
logger = setup_logging(__name__)

class KITTIObjectDataset():
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_anno = os.path.join(opt.dataroot, 'annotations', opt.phase + '_annotations.json')
        self.A_paths, self.B_paths, self.rois_paths = self.getData()
        self.data_size = len(self.A_paths)
        self.depth_normalize = 255. * 80.
        self.uniform_size = (385, 1243)
        self.ignore_cate_list = ['Person_sitting', 'Misc', 'DontCare']

    def getData(self):
        with open(self.dir_anno, 'r') as load_f:
            AB_anno = json.load(load_f)
        A_list = [os.path.join(self.opt.dataroot, AB_anno[i]['rgb_path']) for i in range(len(AB_anno))]
        B_list = [os.path.join(self.opt.dataroot, AB_anno[i]['depth_path']) for i in range(len(AB_anno))]
        rois_list = [os.path.join(self.opt.dataroot, AB_anno[i]['rois_path']) for i in range(len(AB_anno))]

        logger.info('Loaded Kitti data!')
        return A_list, B_list, rois_list

    def __getitem__(self, anno_index):
        if 'train' in self.opt.phase:
            data = self.online_aug_train(anno_index)
        else:
            data = self.online_aug_val_test(anno_index)
        return data

    def online_aug_train(self, idx):
        A_path = self.A_paths[idx]
        B_path = self.B_paths[idx]
        rois_path = self.rois_paths[idx]

        A = cv2.imread(A_path, -1)  # [H, W, C] C:bgr
        B = cv2.imread(B_path, -1) / self.depth_normalize  #[0.0, 1.0]
        #B = np.load(B_path) / (self.depth_normalize / 255.)  #[0.0, 1.0]
        #B[B<0] = -1

        rois = read_labels(rois_path, ignore_cate=self.ignore_cate_list) # list of instances of class ObjectLabel, see obj_utils.py
        rois_mask = rois2mask(rois, B.shape) # use a mask which has same shape to lidepth  to prepresent the rois. rois have value 1., else 0.
        B_rois = B * rois_mask

        flip_flg, resize_size, crop_size, pad, resize_ratio = self.set_flip_pad_reshape_crop(A)

        A_crop = self.flip_pad_reshape_crop(A, flip_flg, resize_size, crop_size, pad, 128)
        B_crop = self.flip_pad_reshape_crop(B, flip_flg, resize_size, crop_size, pad, -1)
        B_rois_crop = self.flip_pad_reshape_crop(B_rois, flip_flg, resize_size, crop_size, pad, -1)
        rois_mask_crop = self.flip_pad_reshape_crop(rois_mask, flip_flg, resize_size, crop_size, pad, 0)

        A_crop = A_crop.transpose((2, 0, 1))
        B_crop = B_crop[np.newaxis, :, :]
        B_rois_crop = B_rois_crop[np.newaxis, :, :]
        rois_mask_crop = rois_mask_crop[np.newaxis, :, :]

        # change the color channel, bgr->rgb
        A_crop = A_crop[::-1, :, :]

        # to torch, normalize
        A_crop = self.scale_torch(A_crop, 255.)
        B_crop = self.scale_torch(B_crop, resize_ratio)
        B_rois_crop = self.scale_torch(B_rois_crop, resize_ratio)

        rois_mask_crop = np.ceil(rois_mask_crop).astype(np.uint8)
        rois_mask_crop = torch.from_numpy(rois_mask_crop.copy())

        B_classes = self.depth_to_class(B_crop)
        B_rois_classes = self.depth_to_class(B_rois_crop)

        invalid_side = [0, 0, 0, 0] if crop_size[1] != 0 else [int((pad[0] + 50)*resize_ratio), 0, 0, 0]

        A = np.pad(A, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant', constant_values=(0, 0))
        B = np.pad(B, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant', constant_values=(0, 0))

        data = {'A': A_crop, 'B': B_crop, 'A_raw': A, 'B_raw': B, 'B_classes': B_classes, 'B_rois_classes': B_rois_classes, 'A_paths': A_path,
                    'B_paths': B_path, 'invalid_side': np.array(invalid_side), 'pad_raw': np.array(pad), 'rois_mask': rois_mask_crop}
        return data

    def online_aug_val_test(self, idx):
        A_path = self.A_paths[idx]
        B_path = self.B_paths[idx]
        rois_path = self.rois_paths[idx]

        A = cv2.imread(A_path, -1)  # [H, W, C] C:bgr

        B = cv2.imread(B_path, -1) / (self.depth_normalize)  #[0.0, 1.0]
        #B = np.load(B_path) / (self.depth_normalize / 255.)  #[0.0, 1.0]
        #B[B<0] = 0
        rois = read_labels(rois_path, ignore_cate=self.ignore_cate_list) # list of instances of class ObjectLabel, see obj_utils.py
        rois_mask = rois2mask_shrink(rois, B.shape) # use a mask which has same shape to lidepth  to prepresent the rois. rois have value 1., else 0.
        B_rois = B * rois_mask

        flip_flg, resize_size, crop_size, pad, resize_ratio = self.set_flip_pad_reshape_crop(A)

        crop_size_l = [pad[2], 0, cfg.CROP_SIZE[1], cfg.CROP_SIZE[0]]
        crop_size_m = [cfg.CROP_SIZE[1] + pad[2] - 20, 0, cfg.CROP_SIZE[1], cfg.CROP_SIZE[0]]
        crop_size_r = [self.uniform_size[1] - cfg.CROP_SIZE[1], 0, cfg.CROP_SIZE[1], cfg.CROP_SIZE[0]]

        A_crop_l = self.flip_pad_reshape_crop(A, flip_flg, resize_size, crop_size_l, pad, 128)
        A_crop_l = A_crop_l.transpose((2, 0, 1))
        A_crop_l = A_crop_l[::-1, :, :]

        A_crop_m = self.flip_pad_reshape_crop(A, flip_flg, resize_size, crop_size_m, pad, 128)
        A_crop_m = A_crop_m.transpose((2, 0, 1))
        A_crop_m = A_crop_m[::-1, :, :]

        A_crop_r = self.flip_pad_reshape_crop(A, flip_flg, resize_size, crop_size_r, pad, 128)
        A_crop_r = A_crop_r.transpose((2, 0, 1))
        A_crop_r = A_crop_r[::-1, :, :]

        A_crop_l = self.scale_torch(A_crop_l, 255.)
        A_crop_m = self.scale_torch(A_crop_m, 255.)
        A_crop_r = self.scale_torch(A_crop_r, 255.)
        A = np.pad(A, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant', constant_values=(0, 0))
        B = np.pad(B, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant', constant_values=(0, 0))
        B_rois = np.pad(B_rois, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant', constant_values=(0, 0))
        crop_lmr = np.array((crop_size_l, crop_size_m, crop_size_r))

        A_crop = A.transpose((2, 0, 1))
        B_crop = B[np.newaxis, :, :]
        # change the color channel, bgr->rgb
        A_crop = A_crop[::-1, :, :]
        # to torch, normalize
        A_crop = self.scale_torch(A_crop, 255.)
        B_crop = self.scale_torch(B_crop, 1.0)

        data = {'A': A_crop, 'B': B_crop,'A_l': A_crop_l, 'A_m': A_crop_m, 'A_r': A_crop_r,
                'A_raw': A, 'B_raw': B, 'B_raw_rois': B_rois, 'A_paths': A_path, 'B_paths': B_path, 'pad_raw': np.array(pad), 'crop_lmr': crop_lmr}
        return data

    def set_flip_pad_reshape_crop(self, A):
        """
        Set flip, padding, reshaping and cropping flags.
        :param A: Input image, [H, W, C]
        :return: Data augamentation parameters
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        # pad
        pad_height = self.uniform_size[0] - A.shape[0]
        pad_width = self.uniform_size[1] - A.shape[1]
        pad = [pad_height, 0, pad_width, 0] #[up, down, left, right]

        # reshape
        ratio_list = [1.0, 1.2, 1.5, 1.8, 2.0]#
        resize_ratio = ratio_list[np.random.randint(len(ratio_list))] if 'train' in self.opt.phase else 1.0
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
        """
        Preprocessing input image or ground truth depth.
        :param img: RGB image or depth image
        :param flip: Flipping flag, True or False
        :param resize_size: Resizing size
        :param crop_size: Cropping size
        :param pad: Padding region
        :param pad_value: Padding value
        :return: Processed image
        """
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
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth <= 0.
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
        return 'KITTI'
