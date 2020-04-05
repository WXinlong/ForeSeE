import torch
import cv2
from lib.core.config import cfg
import numpy as np
import torch.nn.functional as F



def fg_bg_maxpooling(clses_bg, clses_fg):
    "[b,c,h,w]"
    B,C,H,W = clses_bg.shape
    clses_bg = clses_bg.permute(0, 2, 3, 1) #[b, h, w, c]        
    clses_bg = clses_bg.reshape((B, -1, C)).unsqueeze(-1)

    clses_fg = clses_fg.permute(0, 2, 3, 1) #[b, h, w, c]        
    clses_fg = clses_fg.reshape((B, -1, C)).unsqueeze(-1)

    clses_cat = torch.cat((clses_fg, clses_bg), -1) # [b, hxw, c, 2]

    clses_final = F.max_pool2d(clses_cat, kernel_size=(1, 2)) #[b, hxw, c, 1]
    clses_final = clses_final.squeeze(-1).reshape((B, H, W, C))

    clses_final = clses_final.permute(0, 3, 1, 2) # [b, c, h, w]

    clses_final = F.softmax(clses_final, dim=1)

    return clses_final


def class_depth(classes):
    """
    Transfer n-channel output of the network in classes to 1-channel depth
    @classes: n-channel output of the network, [b, c, h, w]
    :return: 1-channel depth, [b, 1, h, w]
    """
    if type(classes).__module__ != torch.__name__:
        classes = torch.tensor(classes, dtype=torch.float32).cuda()
    classes = classes.permute(0, 2, 3, 1) #[b, h, w, c]
    if type(cfg.DATA.DEPTH_CLASSES).__module__ != torch.__name__:
        cfg.DATA.DEPTH_CLASSES = torch.tensor(cfg.DATA.DEPTH_CLASSES, dtype=torch.float32).cuda()
    depth = classes * cfg.DATA.DEPTH_CLASSES
    depth = torch.sum(depth, dim=3, dtype=torch.float32, keepdim=True)
    depth = 10 ** depth
    depth = depth.permute(0, 3, 1, 2) #[b, 1, h, w]
    return depth

def class_depth_hard(classes):
    """
    Transfer n-channel output of the network in classes to 1-channel depth
    @classes: n-channel output of the network, [b, c, h, w]
    :return: 1-channel depth, [b, 1, h, w]
    """
    if type(classes).__module__ != torch.__name__:
        classes = torch.tensor(classes, dtype=torch.float32).cuda()
    classes = classes.permute(0, 2, 3, 1) #[b, h, w, c]
    if type(cfg.DATA.DEPTH_CLASSES).__module__ != torch.__name__:
        cfg.DATA.DEPTH_CLASSES = torch.tensor(cfg.DATA.DEPTH_CLASSES, dtype=torch.float32).cuda()

    # softmax to one-hot
    max_idx = torch.argmax(classes, -1, keepdim=True)
    one_hot = torch.FloatTensor(classes.shape).zero_().to(device=max_idx.device)
    one_hot.scatter_(-1, max_idx, 1)
    classes = one_hot

    depth = classes * cfg.DATA.DEPTH_CLASSES
    depth = torch.sum(depth, dim=3, dtype=torch.float32, keepdim=True)
    depth = 10 ** depth
    depth = depth.permute(0, 3, 1, 2) #[b, 1, h, w]
    return depth

def depth_class(depth):
    """
    Transfer 1-channel depth to 1-channel depth in n depth ranges
    :param depth: 1-channel depth, [b, 1, h, w]
    :return: classes [b, 1, h, w]
    """
    depth[depth < cfg.DATA.DATA_MIN] = cfg.DATA.DATA_MIN
    depth[depth > cfg.DATA.DATA_MAX] = cfg.DATA.DATA_MAX
    classes = torch.round((torch.log10(depth) - cfg.DATA.DATA_MIN_LOG) / cfg.DATA.DEPTH_RANGE_INTERVAL)
    classes = classes.to(torch.long)
    classes[classes == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
    return classes


def resize_image(img, size):
    if type(img).__module__ != np.__name__:
        img = img.cpu().numpy()
    img = cv2.resize(img, (size[1], size[0]))
    return img


def kitti_merge_imgs(left, middle, right, img_shape, crops):
    left = torch.squeeze(left)
    right = torch.squeeze(right)
    middle = torch.squeeze(middle)
    out = torch.zeros(img_shape, dtype=left.dtype, device=left.device)
    crops = torch.squeeze(crops)
    band = 5

    out[:, crops[0][0]:crops[0][0] + crops[0][2] - band] = left[:, 0:left.size(1)-band]
    out[:, crops[1][0]+band:crops[1][0] + crops[1][2] - band] += middle[:, band:middle.size(1)-band]
    out[:, crops[1][0] + crops[1][2] - 2*band:crops[2][0] + crops[2][2]] += right[:, crops[1][0] + crops[1][2] - 2*band-crops[2][0]:]

    out[:, crops[1][0]+band:crops[0][0] + crops[0][2] - band] /= 2.0
    out[:, crops[1][0] + crops[1][2] - 2*band:crops[1][0] + crops[1][2] - band] /= 2.0
    out = out.cpu().numpy()

    return out
