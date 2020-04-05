import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='Path to images')
        parser.add_argument('--batchsize', type=int, default=2, help='Batch size')
        parser.add_argument('--dataset', type=str, default='nyudv2', help='Dataset for training, [kitti | nyudv2]')
        parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
        parser.add_argument('--scale_decoder_lr', type=float, default=1, help='Scale factor for the decoder learning rate')
        parser.add_argument('--load_ckpt', help='Checkpoint path to load')
        parser.add_argument('--resume', action='store_true', help='Resume to train')
        parser.add_argument('--encoder', default='ResNeXt101_32x4d_body_stride16', type=str,
                                help='Set encoder model, [ResNeXt50_32x4d_body_stride16, MobileNetV2_body_stride8, ResNeXt101_32x4d_body_stride16]')
        parser.add_argument('--pretrained_model', default='resnext101_32x4d.pth', type=str, help='Pretrained model')
        parser.add_argument('--decoder_out_c', type=int, default=150, help='Output channel of the decoder')
        parser.add_argument('--epoch', default=[0, 30], nargs='+', type=int, help='Decaying epoch milestone: 0 30 40 50')
        parser.add_argument('--thread', default=2, type=int, help='Thread for loading data')
        parser.add_argument('--use_tfboard', action='store_true', help='Tensorboard to log training info')
        parser.add_argument('--results_dir', type=str, default='./evaluation', help='Output dir')
        parser.add_argument('--pcd_dir', type=str, default='output/', help='point cloud output dir')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch for training')
        parser.add_argument('--start_step', default=0, type=int,
                            help='set starting step for training, especially for resuming training')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt
