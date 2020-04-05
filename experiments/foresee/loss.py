import numpy as np
from lib.core.config import cfg
import torch
import torch.nn.functional as F


def cross_entropy_loss(pred_nosoftmax, gt_class):
    """
    Standard cross-entropy loss
    :param pred_nosoftmax: predicted label
    :param gt_class: target label
    :return:
    """
    gt_class = torch.squeeze(gt_class)
    gt_class = gt_class.to(device=pred_nosoftmax.device, dtype=torch.int64)
    entropy = torch.nn.CrossEntropyLoss(ignore_index=cfg.MODEL.DECODER_OUTPUT_C+1)
    loss = entropy(pred_nosoftmax, gt_class)
    return loss


def weight_crossentropy_loss(pred_nosoftmax, gt, data):
    """
    Weighted Cross-entropy Loss
    :param pred_nosoftmax: predicted label
    :param gt: target label
    """
    invalid_side = data['invalid_side']
    cfg.DATA.WCE_LOSS_WEIGHT = torch.tensor(cfg.DATA.WCE_LOSS_WEIGHT, dtype=torch.float32, device=pred_nosoftmax.device)
    weight = cfg.DATA.WCE_LOSS_WEIGHT
    weight /= torch.sum(weight, 1, keepdim=True)
    classes_range = torch.arange(cfg.MODEL.DECODER_OUTPUT_C, device=gt.device, dtype=gt.dtype)
    log_pred = torch.nn.functional.log_softmax(pred_nosoftmax, 1)
    log_pred = torch.t(torch.transpose(log_pred, 0, 1).reshape(log_pred.size(1), -1))

    gt_reshape = gt.reshape(-1, 1)
    one_hot = (gt_reshape == classes_range).to(dtype=torch.float, device=pred_nosoftmax.device)
    weight = torch.matmul(one_hot, weight)
    weight_log_pred = weight * log_pred

    valid_pixes = torch.tensor([0], device=pred_nosoftmax.device, dtype=torch.float)
    for i in range(gt.size(0)):
        valid_gt = gt[i, :,  int(invalid_side[i][0]):gt.size(2)-int(invalid_side[i][1]), :]
        valid_pixes += valid_gt.size(1) * valid_gt.size(2)
    loss_sum = -1 * torch.sum(weight_log_pred)
    return loss_sum, valid_pixes

def rois_weight_crossentropy_loss(pred_nosoftmax, gt, data):
    """
    Weighted Cross-entropy Loss
    :param pred_nosoftmax: predicted label
    :param gt: target label
    """
    invalid_side = data['invalid_side']
    rois_mask = data['rois_mask']
    cfg.DATA.WCE_LOSS_WEIGHT = torch.tensor(cfg.DATA.WCE_LOSS_WEIGHT, dtype=torch.float32, device=pred_nosoftmax.device)
    weight = cfg.DATA.WCE_LOSS_WEIGHT
    weight /= torch.sum(weight, 1, keepdim=True)
    classes_range = torch.arange(cfg.MODEL.DECODER_OUTPUT_C, device=gt.device, dtype=gt.dtype)
    log_pred = torch.nn.functional.log_softmax(pred_nosoftmax, 1)
    log_pred = torch.t(torch.transpose(log_pred, 0, 1).reshape(log_pred.size(1), -1))

    gt_reshape = gt.reshape(-1, 1)
    one_hot = (gt_reshape == classes_range).to(dtype=torch.float, device=pred_nosoftmax.device)
    weight = torch.matmul(one_hot, weight)
    weight_log_pred = weight * log_pred

    valid_pixels = max(rois_mask.sum(), 1)
    loss_sum = -1 * torch.sum(weight_log_pred)
    return loss_sum, valid_pixels

def rois_scale_invariant_loss(pred_depth, data):
    """
    Follow Eigen paper, add silog loss, for KITTI benchmark
    :param pred_depth:
    :param data:
    :return:
    """
    invalid_side = data['invalid_side']
    gt_depth = data['B'].cuda()

    rois_mask = data['rois_mask'].to(device=gt_depth.device)

    loss_mean = torch.tensor([0.]).cuda()
    for j in range(pred_depth.size(0)):
        valid_pred = pred_depth[j, :, int(invalid_side[j][0]): pred_depth.size(2) - int(invalid_side[j][1]), :]
        valid_gt = gt_depth[j, :, int(invalid_side[j][0]): gt_depth.size(2) - int(invalid_side[j][1]), :]
        valid_rois_mask = rois_mask[j, :, int(invalid_side[j][0]): rois_mask.size(2) - int(invalid_side[j][1]), :]

        diff_log = torch.log(valid_pred) - torch.log(valid_gt)
        diff_log = diff_log * valid_rois_mask.to(dtype=diff_log.dtype)

        #size = torch.numel(diff_log)
        size = torch.sum(valid_rois_mask)
        if size == 0:
            continue

        loss_mean += torch.sum(diff_log ** 2) / size - 0.5 * torch.sum(diff_log) ** 2 / (size ** 2)
    loss = loss_mean / pred_depth.size(0)
    return loss


def scale_invariant_loss(pred_depth, data):
    """
    Follow Eigen paper, add silog loss, for KITTI benchmark
    :param pred_depth:
    :param data:
    :return:
    """
    invalid_side = data['invalid_side']
    gt_depth = data['B'].cuda()


    loss_mean = torch.tensor([0.]).cuda()
    for j in range(pred_depth.size(0)):
        valid_pred = pred_depth[j, :, int(invalid_side[j][0]): pred_depth.size(2) - int(invalid_side[j][1]), :]
        valid_gt = gt_depth[j, :, int(invalid_side[j][0]): gt_depth.size(2) - int(invalid_side[j][1]), :]

        diff_log = torch.log(valid_pred) - torch.log(valid_gt)

        size = torch.numel(diff_log)
        #size = torch.sum(valid_rois_mask)
        #if size == 0:
        #    continue

        loss_mean += torch.sum(diff_log ** 2) / size - 0.5 * torch.sum(diff_log) ** 2 / (size ** 2)
    loss = loss_mean / pred_depth.size(0)
    return loss


def berhu_loss(pred_depth, data, scale=80.):
    """
    :param pred_depth:
    :param data:
    :return:
    """
    huber_threshold = 0.2

    invalid_side = data['invalid_side']
    gt_depth = data['B'].cuda()

    mask = gt_depth > 0

    pred_depth = pred_depth * mask.to(dtype=pred_depth.dtype)
    gt_depth = gt_depth * mask.to(dtype=gt_depth.dtype)

    diff = torch.abs(gt_depth - pred_depth)
    delta = huber_threshold * torch.max(diff).data.cpu()

    part1 = -F.threshold(-diff, -delta, 0.)
    part2 = F.threshold(diff**2 + delta**2, 2*delta**2, 0.) 
    part2 = part2 / (2.*delta)

    loss = part1 + part2

    loss = loss[mask]
    loss = torch.mean(loss)

    return loss


def rmse_log_loss(pred_depth, data, scale=80.):
    """
    :param pred_depth:
    :param data:
    :return:
    """

    gt_depth = data['B'].cuda()
    mask = gt_depth > 0

    pred_depth = pred_depth * scale
    gt_depth = gt_depth * scale

    diff = torch.log(gt_depth) - torch.log(pred_depth)
    diff = diff[mask]
    
    loss = torch.sqrt(torch.mean(diff**2))
    return loss


def rmse_loss(pred_depth, data, scale=80.):
    """
    :param pred_depth:
    :param data:
    :return:
    """

    gt_depth = data['B'].cuda()
    mask = gt_depth > 0

    pred_depth = pred_depth
    gt_depth = gt_depth

    diff = gt_depth - pred_depth
    diff = diff[mask]
    
    loss = torch.sqrt(torch.mean(diff**2))
    return loss

def mse_loss(pred_depth, data, scale=80.):
    """
    :param pred_depth:
    :param data:
    :return:
    """

    gt_depth = data['B'].cuda()
    mask = gt_depth > 0

    pred_depth = pred_depth
    gt_depth = gt_depth

    diff = gt_depth - pred_depth
    diff = diff[mask]
    
    loss = torch.mean(diff**2)
    return loss


def rois_rmse_log_loss(pred_depth, data, scale=80.):
    """
    :param pred_depth:
    :param data:
    :return:
    """

    gt_depth = data['B'].cuda()

    mask = gt_depth > 0
    rois_mask = data['rois_mask'].to(device=gt_depth.device)
    mask = mask & rois_mask

    pred_depth = pred_depth
    gt_depth = gt_depth

    diff = torch.log(gt_depth) - torch.log(pred_depth)
    diff = diff[mask]
    
    loss = torch.sqrt(torch.mean(diff**2))
    return loss
