import os
import sys
import math
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from data.load_dataset import CustomerDataLoader
from lib.utils.training_stats import TrainingStats
from lib.utils.evaluate_depth_error import validate_err_kitti
from lib.models.depth_normal_model import *
from lib.core.config import cfg, train_args, val_args, merge_cfg_from_file
from lib.utils.net_tools import save_ckpt, load_ckpt
from lib.utils.logging import setup_logging, SmoothedValue
logger = setup_logging(__name__)


def train(train_dataloader, model, epoch, loss_func,
          optimizer, scheduler, training_stats, val_dataloader=None, val_err=[], ignore_step=-1):
    model.train()
    epoch_steps = math.ceil(len(train_dataloader) / cfg.TRAIN.BATCH_SIZE)
    base_steps = epoch_steps * epoch + ignore_step if ignore_step != -1 else epoch_steps * epoch
    for i, data in enumerate(train_dataloader):
        if ignore_step != -1 and i > epoch_steps - ignore_step:
            return
        scheduler.step()  # decay lr every iteration
        training_stats.IterTic()
        out = model(data)
        losses = loss_func.criterion(out['b_fake'], out['b_fake_nosoftmax'], data, epoch)
        optimizer.optim(losses)

        step = base_steps + i + 1
        training_stats.UpdateIterStats(losses)
        training_stats.IterToc()
        training_stats.LogIterStats(step, epoch, optimizer.optimizer, val_err[0])

        # validate the model
        if step % cfg.TRAIN.VAL_STEP == 0 and step != 0 and val_dataloader is not None:#
            model.eval()
            val_err[0] = val_kitti(val_dataloader, model)
            print("val_erro: {}".format(val_err[0]))
            # training mode
            model.train()
        # save checkpoint
        if step % cfg.TRAIN.SNAPSHOT_ITERS == 0 and step != 0:
            save_ckpt(train_args, step, epoch, model, optimizer.optimizer, scheduler, val_err[0])


def val_kitti(val_dataloader, model):
    """
    Validate the model.
    """
    smoothed_absRel = SmoothedValue(len(val_dataloader))
    smoothed_silog = SmoothedValue(len(val_dataloader))
    smoothed_silog2 = SmoothedValue(len(val_dataloader))
    smoothed_criteria = {'err_absRel': smoothed_absRel, 'err_silog': smoothed_silog, 'err_silog2': smoothed_silog2}
    for i, data in enumerate(val_dataloader):
        pred_depth = model.module.inference_kitti(data)
        smoothed_criteria = validate_err_kitti(pred_depth['b_fake'], data['B_raw'], smoothed_criteria)
        print(np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (smoothed_criteria['err_silog'].GetGlobalAverageValue())**2))
    val_metrics = {'abs_rel': smoothed_criteria['err_absRel'].GetGlobalAverageValue(),
            'silog': np.sqrt(smoothed_criteria['err_silog2'].GetGlobalAverageValue() - (smoothed_criteria['err_silog'].GetGlobalAverageValue())**2)}
    print(val_metrics)
    return val_metrics

if __name__=='__main__':
    train_dataloader = CustomerDataLoader(train_args)
    train_datasize = len(train_dataloader)
    gpu_num = torch.cuda.device_count()
    merge_cfg_from_file(train_datasize, gpu_num)

    val_dataloader = CustomerDataLoader(val_args)
    val_datasize = len(val_dataloader)

    # tensorboard logger
    if train_args.use_tfboard:
        from tensorboardX import SummaryWriter
        tblogger = SummaryWriter(cfg.TRAIN.LOG_DIR)

    # training status for logging
    training_stats = TrainingStats(train_args, cfg.TRAIN.LOG_INTERVAL,
                                   tblogger if train_args.use_tfboard else None)

    # total iterations
    total_iters = math.ceil(train_datasize / train_args.batchsize) * train_args.epoch[-1]

    # load model
    model = DepthNormal()

    if gpu_num != -1:
        logger.info('{:>15}: {:<30}'.format('GPU_num', gpu_num))
        logger.info('{:>15}: {:<30}'.format('train_data_size', train_datasize))
        logger.info('{:>15}: {:<30}'.format('val_data_size', val_datasize))
        logger.info('{:>15}: {:<30}'.format('total_iterations', total_iters))
        model.cuda()
    #optimizer
    optimizer = ModelOptimizer(model)
    #loss function
    loss_func = ModelLoss()

    val_err = [{'abs_rel': 0, 'silog': 0}]

    ignore_step = -1

    # Lerning strategy
    lr_optim_lambda = lambda iter: (1.0 - iter / (float(total_iters))) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer.optimizer, lr_lambda=lr_optim_lambda)

    # load checkpoint
    if train_args.load_ckpt:
        load_ckpt(train_args, model, optimizer.optimizer, scheduler, val_err)
        ignore_step = train_args.start_step - train_args.start_epoch * math.ceil(train_datasize / train_args.batchsize)

    if gpu_num != -1:
        model = torch.nn.DataParallel(model)
    try:
        for epoch in range(train_args.start_epoch, cfg.TRAIN.EPOCH[-1]):
            # training
            train(train_dataloader, model, epoch, loss_func, optimizer, scheduler, training_stats,
                  val_dataloader, val_err, ignore_step)
            ignore_step = -1

    except (RuntimeError, KeyboardInterrupt):
        logger.info('Save ckpt on exception ...')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if train_args.use_tfboard:
            tblogger.close()
