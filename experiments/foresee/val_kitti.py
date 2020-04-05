import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from tools.parse_arg_val import ValOptions
from data.load_dataset import CustomerDataLoader
from lib.utils.net_tools import load_ckpt
from lib.utils.evaluate_depth_error import evaluate_err
from lib.utils.net_tools import save_images
from lib.utils.logging import setup_logging, SmoothedValue

from depth_normal_model import DepthNormal
logger = setup_logging(__name__)


if __name__ == '__main__':
    test_args = ValOptions().parse()
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

    # global
    smoothed_absRel = SmoothedValue(test_datasize)
    smoothed_rms = SmoothedValue(test_datasize)
    smoothed_logRms = SmoothedValue(test_datasize)
    smoothed_squaRel = SmoothedValue(test_datasize)
    smoothed_silog = SmoothedValue(test_datasize)
    smoothed_silog2 = SmoothedValue(test_datasize)
    smoothed_log10 = SmoothedValue(test_datasize)
    smoothed_delta1 = SmoothedValue(test_datasize)
    smoothed_delta2 = SmoothedValue(test_datasize)
    smoothed_delta3 = SmoothedValue(test_datasize)
    smoothed_criteria = {'err_absRel':smoothed_absRel, 'err_squaRel': smoothed_squaRel, 'err_rms': smoothed_rms,
                         'err_silog': smoothed_silog, 'err_logRms': smoothed_logRms, 'err_silog2': smoothed_silog2,
                         'err_delta1': smoothed_delta1, 'err_delta2': smoothed_delta2, 'err_delta3': smoothed_delta3,
                         'err_log10': smoothed_log10}

    # rois
    rois_smoothed_absRel = SmoothedValue(test_datasize)
    rois_smoothed_rms = SmoothedValue(test_datasize)
    rois_smoothed_logRms = SmoothedValue(test_datasize)
    rois_smoothed_squaRel = SmoothedValue(test_datasize)
    rois_smoothed_silog = SmoothedValue(test_datasize)
    rois_smoothed_silog2 = SmoothedValue(test_datasize)
    rois_smoothed_log10 = SmoothedValue(test_datasize)
    rois_smoothed_delta1 = SmoothedValue(test_datasize)
    rois_smoothed_delta2 = SmoothedValue(test_datasize)
    rois_smoothed_delta3 = SmoothedValue(test_datasize)
    rois_smoothed_criteria = {'err_absRel':rois_smoothed_absRel, 'err_squaRel': rois_smoothed_squaRel, 'err_rms': rois_smoothed_rms,
                         'err_silog': rois_smoothed_silog, 'err_logRms': rois_smoothed_logRms, 'err_silog2': rois_smoothed_silog2,
                         'err_delta1': rois_smoothed_delta1, 'err_delta2': rois_smoothed_delta2, 'err_delta3': rois_smoothed_delta3,
                         'err_log10': rois_smoothed_log10}

    # bg
    bg_smoothed_absRel = SmoothedValue(test_datasize)
    bg_smoothed_rms = SmoothedValue(test_datasize)
    bg_smoothed_logRms = SmoothedValue(test_datasize)
    bg_smoothed_squaRel = SmoothedValue(test_datasize)
    bg_smoothed_silog = SmoothedValue(test_datasize)
    bg_smoothed_silog2 = SmoothedValue(test_datasize)
    bg_smoothed_log10 = SmoothedValue(test_datasize)
    bg_smoothed_delta1 = SmoothedValue(test_datasize)
    bg_smoothed_delta2 = SmoothedValue(test_datasize)
    bg_smoothed_delta3 = SmoothedValue(test_datasize)
    bg_smoothed_criteria = {'err_absRel':bg_smoothed_absRel, 'err_squaRel': bg_smoothed_squaRel, 'err_rms': bg_smoothed_rms,
                         'err_silog': bg_smoothed_silog, 'err_logRms': bg_smoothed_logRms, 'err_silog2': bg_smoothed_silog2,
                         'err_delta1': bg_smoothed_delta1, 'err_delta2': bg_smoothed_delta2, 'err_delta3': bg_smoothed_delta3,
                         'err_log10': bg_smoothed_log10}
    for i, data in enumerate(data_loader):
        out = model.module.inference_kitti(data)
        pred_depth = np.squeeze(out['b_fake'])
        img_path = data['A_paths']

        if len(data['B_raw'].shape) != 2:
            smoothed_criteria = evaluate_err(pred_depth, data['B_raw'], smoothed_criteria, scale=80.)
            rois_smoothed_criteria = evaluate_err(pred_depth, data['B_raw_rois'], rois_smoothed_criteria, scale=80.)
            bg_smoothed_criteria = evaluate_err(pred_depth, data['B_raw_bg'], bg_smoothed_criteria, scale=80.)
            print('processing (%04d)-th image... %s' % (i, img_path))
            print(smoothed_criteria['err_absRel'].GetGlobalAverageValue())
        save_images(data, pred_depth, scale=256.*80.)


    LOG_FOUT = open(os.path.join('object_val_results.txt'), 'w')
    def log_string(out_str):
        LOG_FOUT.write(out_str+'\n')
        LOG_FOUT.flush()
        print(out_str)


    if len(data['B_raw'].shape) != 2:
        log_string("---image-level----")
        log_string("###############absREL ERROR: {}".format(smoothed_criteria['err_absRel'].GetGlobalAverageValue()))
        log_string("###############silog ERROR: {}".format(np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (smoothed_criteria['err_silog'].GetGlobalAverageValue())**2)))
        log_string("###############log10 ERROR: {}".format(smoothed_criteria['err_log10'].GetGlobalAverageValue()))
        log_string("###############RMS ERROR: {}".format(np.sqrt(smoothed_criteria['err_rms'].GetGlobalAverageValue())))
        log_string("###############delta_1 ERROR: {}".format(smoothed_criteria['err_delta1'].GetGlobalAverageValue()))
        log_string("###############delta_2 ERROR: {}".format(smoothed_criteria['err_delta2'].GetGlobalAverageValue()))
        log_string("###############delta_3 ERROR: {}".format(smoothed_criteria['err_delta3'].GetGlobalAverageValue()))
        log_string("###############squaRel ERROR: {}".format(smoothed_criteria['err_squaRel'].GetGlobalAverageValue()))
        log_string("###############logRms ERROR: {}".format(np.sqrt(smoothed_criteria['err_logRms'].GetGlobalAverageValue())))

        
        log_string("---rois-level----")
        log_string("###############absREL ERROR: {}".format(rois_smoothed_criteria['err_absRel'].GetGlobalAverageValue()))
        log_string("###############silog ERROR: {}".format(np.sqrt(rois_smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (rois_smoothed_criteria['err_silog'].GetGlobalAverageValue())**2)))
        log_string("###############log10 ERROR: {}".format(rois_smoothed_criteria['err_log10'].GetGlobalAverageValue()))
        log_string("###############RMS ERROR: {}".format(np.sqrt(rois_smoothed_criteria['err_rms'].GetGlobalAverageValue())))
        log_string("###############delta_1 ERROR: {}".format(rois_smoothed_criteria['err_delta1'].GetGlobalAverageValue()))
        log_string("###############delta_2 ERROR: {}".format(rois_smoothed_criteria['err_delta2'].GetGlobalAverageValue()))
        log_string("###############delta_3 ERROR: {}".format(rois_smoothed_criteria['err_delta3'].GetGlobalAverageValue()))
        log_string("###############squaRel ERROR: {}".format(rois_smoothed_criteria['err_squaRel'].GetGlobalAverageValue()))
        log_string("###############logRms ERROR: {}".format(np.sqrt(rois_smoothed_criteria['err_logRms'].GetGlobalAverageValue())))

        log_string("---bg-level----")
        log_string("###############absREL ERROR: {}".format(bg_smoothed_criteria['err_absRel'].GetGlobalAverageValue()))
        log_string("###############silog ERROR: {}".format(np.sqrt(bg_smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (bg_smoothed_criteria['err_silog'].GetGlobalAverageValue())**2)))
        log_string("###############log10 ERROR: {}".format(bg_smoothed_criteria['err_log10'].GetGlobalAverageValue()))
        log_string("###############RMS ERROR: {}".format(np.sqrt(bg_smoothed_criteria['err_rms'].GetGlobalAverageValue())))
        log_string("###############delta_1 ERROR: {}".format(bg_smoothed_criteria['err_delta1'].GetGlobalAverageValue()))
        log_string("###############delta_2 ERROR: {}".format(bg_smoothed_criteria['err_delta2'].GetGlobalAverageValue()))
        log_string("###############delta_3 ERROR: {}".format(bg_smoothed_criteria['err_delta3'].GetGlobalAverageValue()))
        log_string("###############squaRel ERROR: {}".format(bg_smoothed_criteria['err_squaRel'].GetGlobalAverageValue()))
        log_string("###############logRms ERROR: {}".format(np.sqrt(bg_smoothed_criteria['err_logRms'].GetGlobalAverageValue())))
