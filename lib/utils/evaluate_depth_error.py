import logging
import torch
import numpy as np
logger = logging.getLogger(__name__)


def validate_err(pred, gt, smoothed_criteria, mask=None, scale=10.):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    if mask is not None:
        gt = gt[mask[0]:mask[1], mask[2]:mask[3]]
        pred = pred[mask[0]:mask[1], mask[2]:mask[3]]
    if pred.shape != gt.shape:
        logger.info('The shapes of dt and gt are not same!')
        return -1
    mask2 = gt > 0
    gt = gt[mask2]
    pred = pred[mask2]
    n_pxl = gt.size
    gt_scale = gt * scale
    pred_scale = pred * scale

    # Mean Absolute Relative Error
    rel = np.abs(gt_scale - pred_scale) / gt_scale  # compute errors
    abs_rel_sum = np.sum(rel)
    smoothed_criteria['err_absRel'].AddValue(np.float64(abs_rel_sum), n_pxl)
    return smoothed_criteria


def validate_err_kitti(pred, gt, smoothed_criteria, mask=None, scale=256.*80.):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    if mask is not None:
        gt = gt[mask[0]:mask[1], mask[2]:mask[3]]
        pred = pred[mask[0]:mask[1], mask[2]:mask[3]]
    if pred.shape != gt.shape:
        logger.info('The shapes of dt and gt are not same!')
        return -1

    mask2 = gt > 0
    gt = gt[mask2]
    pred = pred[mask2]
    n_pxl = gt.size
    gt_scale = gt * scale
    pred_scale = pred * scale

    # Mean Absolute Relative Error
    rel = np.abs(gt_scale - pred_scale) / gt_scale  # compute errors
    abs_rel_sum = np.sum(rel)
    smoothed_criteria['err_absRel'].AddValue(np.float64(abs_rel_sum), n_pxl)

    # Scale invariant error, silog is an evaluation metric of KITTI benchmark
    diff_log = np.log(pred_scale) - np.log(gt_scale)
    diff_log_sum = np.sum(diff_log)
    smoothed_criteria['err_silog'].AddValue(np.float64(diff_log_sum), n_pxl)
    diff_log_2 = diff_log ** 2
    diff_log_2_sum = np.sum(diff_log_2)
    smoothed_criteria['err_silog2'].AddValue(np.float64(diff_log_2_sum), n_pxl)
    return smoothed_criteria


def evaluate_err(pred, gt, smoothed_criteria, mask = None, scale=10.0 ):
    if type(pred).__module__ != np.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.cpu().numpy()

    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    if mask is not None:
        gt = gt[mask[0]:mask[1], mask[2]:mask[3]]
        pred = pred[mask[0]:mask[1], mask[2]:mask[3]]
    if pred.shape != gt.shape:
        logger.info('The shapes of dt and gt are not same!')
        return -1

    mask2 = gt > 0
    gt = gt[mask2]
    pred = pred[mask2]
    n_pxl = gt.size
    gt_scale = gt * scale
    pred_scale = pred * scale

    #Mean Absolute Relative Error
    rel = np.abs(gt - pred) / gt# compute errors
    abs_rel_sum = np.sum(rel)
    smoothed_criteria['err_absRel'].AddValue(np.float64(abs_rel_sum), n_pxl)

    #Square Mean Relative Error
    s_rel = ((gt_scale - pred_scale) * (gt_scale - pred_scale)) / (gt_scale * gt_scale)# compute errors
    squa_rel_sum = np.sum(s_rel)
    smoothed_criteria['err_squaRel'].AddValue(np.float64(squa_rel_sum), n_pxl)

    #Root Mean Square error
    square = (gt_scale - pred_scale) ** 2
    rms_squa_sum = np.sum(square)
    smoothed_criteria['err_rms'].AddValue(np.float64(rms_squa_sum), n_pxl)

    #Log Root Mean Square error
    log_square = (np.log(gt_scale) - np.log(pred_scale)) **2
    log_rms_sum = np.sum(log_square)
    smoothed_criteria['err_logRms'].AddValue(np.float64(log_rms_sum), n_pxl)

    # Scale invariant error
    diff_log = np.log(pred_scale) - np.log(gt_scale)
    diff_log_sum = np.sum(diff_log)
    smoothed_criteria['err_silog'].AddValue(np.float64(diff_log_sum), n_pxl)
    diff_log_2 = diff_log ** 2
    diff_log_2_sum = np.sum(diff_log_2)
    smoothed_criteria['err_silog2'].AddValue(np.float64(diff_log_2_sum), n_pxl)

    # Mean log10 error
    log10_sum = np.sum(np.abs(np.log10(gt) - np.log10(pred)))
    smoothed_criteria['err_log10'].AddValue(np.float64(log10_sum), n_pxl)

    #Delta
    gt_pred = gt_scale / pred_scale
    pred_gt = pred_scale / gt_scale
    gt_pred = np.reshape(gt_pred, (1, -1))
    pred_gt = np.reshape(pred_gt, (1, -1))
    gt_pred_gt = np.concatenate((gt_pred, pred_gt), axis=0)
    ratio_max = np.amax(gt_pred_gt, axis=0)

    delta_1_sum = np.sum(ratio_max < 1.25)
    smoothed_criteria['err_delta1'].AddValue(np.float64(delta_1_sum), n_pxl)
    delta_2_sum = np.sum(ratio_max < 1.25**2)
    smoothed_criteria['err_delta2'].AddValue(np.float64(delta_2_sum), n_pxl)
    delta_3_sum = np.sum(ratio_max < 1.25**3)
    smoothed_criteria['err_delta3'].AddValue(np.float64(delta_3_sum), n_pxl)

    return smoothed_criteria

