import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from tools.parse_arg_val import ValOptions
from data.load_dataset import CustomerDataLoader
from lib.models.depth_normal_model import DepthNormal
from lib.utils.net_tools import load_ckpt
from lib.utils.evaluate_depth_error import evaluate_err
from lib.utils.net_tools import save_images
from lib.utils.logging import setup_logging, SmoothedValue
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

    # test
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

    for i, data in enumerate(data_loader):
        out = model.module.inference_kitti(data)
        pred_depth = np.squeeze(out['b_fake'])
        img_path = data['A_paths']

        if len(data['B_raw'].shape) != 2:
            smoothed_criteria = evaluate_err(pred_depth, data['B_raw'], smoothed_criteria, scale=80.)
            print('processing (%04d)-th image... %s' % (i, img_path))
            print(smoothed_criteria['err_absRel'].GetGlobalAverageValue())
        #save_images(data, pred_depth, scale=256.*80.)


    if len(data['B_raw'].shape) != 2:
        print("###############absREL ERROR: %f", smoothed_criteria['err_absRel'].GetGlobalAverageValue())
        print("###############silog ERROR: %f", np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (smoothed_criteria['err_silog'].GetGlobalAverageValue())**2))
        print("###############log10 ERROR: %f", smoothed_criteria['err_log10'].GetGlobalAverageValue())
        print("###############RMS ERROR: %f", np.sqrt(smoothed_criteria['err_rms'].GetGlobalAverageValue()))
        print("###############delta_1 ERROR: %f", smoothed_criteria['err_delta1'].GetGlobalAverageValue())
        print("###############delta_2 ERROR: %f", smoothed_criteria['err_delta2'].GetGlobalAverageValue())
        print("###############delta_3 ERROR: %f", smoothed_criteria['err_delta3'].GetGlobalAverageValue())
        print("###############squaRel ERROR: %f", smoothed_criteria['err_squaRel'].GetGlobalAverageValue())
        print("###############logRms ERROR: %f", np.sqrt(smoothed_criteria['err_logRms'].GetGlobalAverageValue()))

