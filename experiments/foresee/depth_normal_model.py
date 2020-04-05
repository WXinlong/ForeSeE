import lateral_net
from lib.utils.net_tools import *
from lib.models.image_transfer import *
from loss import weight_crossentropy_loss, rois_weight_crossentropy_loss 
from lib.core.config import cfg



class DepthNormal(nn.Module):
    def __init__(self):
        super(DepthNormal, self).__init__()
        self.loss_names = ['Weighted_Cross_Entropy', 'Global_Normal']
        self.depth_normal_model = DepthModel()

    def forward(self, data):
        # Input data is a_real, predicted data is b_fake, groundtruth is b_real
        self.a_real = data['A'].cuda()
        self.boxes = data['bbox'].to(device=self.a_real.device)
        self.b_fake, self.b_roi_fake  = self.depth_normal_model(self.a_real, self.boxes)
        return {'b_fake': self.b_fake[1], 'b_fake_nosoftmax': self.b_fake[0], 'b_fake_roi': self.b_roi_fake[1], 'b_fake_roi_nosoftmax': self.b_roi_fake[0]}

    def inference(self, data):
        with torch.no_grad():
            out = self.forward(data)
        
            class_conf_final = fg_bg_maxpooling(out['b_fake_nosoftmax'], out['b_fake_roi_nosoftmax'])
            out_depth_final = class_depth(class_conf_final)

            class_conf = out['b_fake']
            return {'b_fake': out_depth_final, 'b_fake_conf': class_conf}

    def inference_kitti(self, data):
        #crop kitti images into 3 parts
        with torch.no_grad():
            self.a_l_real = data['A_l'].cuda()
            self.boxes_l = data['bbox_l'].to(device=self.a_l_real.device)
            [b_l_classes_nosoftmax, b_l_classes], [b_l_roi_classes_nosoftmax, b_l_roi_classes] = self.depth_normal_model(self.a_l_real, self.boxes_l)
            b_l_classes_final = fg_bg_maxpooling(b_l_classes_nosoftmax, b_l_roi_classes_nosoftmax)
            self.b_l_fake_final = class_depth(b_l_classes_final)

            self.a_m_real = data['A_m'].cuda()
            self.boxes_m = data['bbox_m'].to(device=self.a_m_real.device)
            [b_m_classes_nosoftmax, b_m_classes], [b_m_roi_classes_nosoftmax, b_m_roi_classes] = self.depth_normal_model(self.a_m_real, self.boxes_m)
            b_m_classes_final = fg_bg_maxpooling(b_m_classes_nosoftmax, b_m_roi_classes_nosoftmax)
            self.b_m_fake_final = class_depth(b_m_classes_final)

            self.a_r_real = data['A_r'].cuda()
            self.boxes_r = data['bbox_r'].to(device=self.a_r_real.device)
            [b_r_classes_nosoftmax, b_r_classes], [b_r_roi_classes_nosoftmax, b_r_roi_classes] = self.depth_normal_model(self.a_r_real, self.boxes_r)
            b_r_classes_final = fg_bg_maxpooling(b_r_classes_nosoftmax, b_r_roi_classes_nosoftmax)
            self.b_r_fake_final = class_depth(b_r_classes_final)

            out = kitti_merge_imgs(self.b_l_fake_final, self.b_m_fake_final, self.b_r_fake_final, torch.squeeze(data['B_raw']).shape, data['crop_lmr'])
            return {'b_fake': out}


class ModelLoss(object):
    def __init__(self):
        super(ModelLoss, self).__init__()
        self.weight_cross_entropy_loss =weight_crossentropy_loss
        self.rois_weight_cross_entropy_loss =rois_weight_crossentropy_loss


    def criterion(self, pred_softmax, pred_nosoftmax, pred_softmax_roi, pred_nosoftmax_roi, data, epoch):
        loss = {}
        # transfer output and gt
        pred_depth = class_depth(pred_softmax)

        #alpha = 0.99
        add_alpha = 0.2
        add_beta = 0.2

        # bg
        loss_entropy, valid_num = self.weight_cross_entropy_loss(pred_nosoftmax, data['B_classes'], data)
        loss_entropy_rois, valid_num_roi = self.rois_weight_cross_entropy_loss(pred_nosoftmax, data['B_rois_classes'], data)

        loss['bg_wcel_loss_fg'] = loss_entropy_rois / valid_num_roi
        loss['bg_wcel_loss_bg'] = (loss_entropy - loss_entropy_rois) / (valid_num - valid_num_roi)
        loss['bg_wcel_loss'] = (1 - add_beta) * loss['bg_wcel_loss_bg'] +  add_beta * loss['bg_wcel_loss_fg']

        # fg
        fg_loss_entropy, fg_valid_num = self.weight_cross_entropy_loss(pred_nosoftmax_roi, data['B_classes'], data)
        fg_loss_entropy_rois, fg_valid_num_roi = self.rois_weight_cross_entropy_loss(pred_nosoftmax_roi, data['B_rois_classes'], data)

        loss['fg_wcel_loss_bg'] = (fg_loss_entropy - fg_loss_entropy_rois) / (fg_valid_num - fg_valid_num_roi)
        loss['fg_wcel_loss_fg'] = fg_loss_entropy_rois / fg_valid_num_roi
        loss['fg_wcel_loss'] = (1 - add_alpha) * loss['fg_wcel_loss_fg'] + add_alpha * loss['fg_wcel_loss_bg']

        loss['total_loss'] = loss['bg_wcel_loss'] + loss['fg_wcel_loss']
        return loss


class ModelOptimizer(object):
    def __init__(self, model):
        super(ModelOptimizer, self).__init__()
        backbone_params = []
        backbone_params_names = []
        nonbackbone_others_params = []
        nonbackbone_others_params_names = []
        nograd_param_names = []

        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'res' in key:
                    backbone_params.append(value)
                    backbone_params_names.append(key)
                else:
                    nonbackbone_others_params.append(value)
                    nonbackbone_others_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_resnet = cfg.TRAIN.BASE_LR
        lr_fcn = cfg.TRAIN.BASE_LR * cfg.TRAIN.DIFF_LR
        weight_decay = 0.0005

        net_params = [
            {'params': backbone_params,
             'lr': lr_resnet,
             'weight_decay': weight_decay},
            {'params': nonbackbone_others_params,
             'lr': lr_fcn,
             'weight_decay': weight_decay},
            ]
        self.optimizer = torch.optim.SGD(net_params, momentum=0.9)
    def optim(self, loss):
        self.optimizer.zero_grad()
        loss_all = loss['total_loss']
        loss_all.backward()
        self.optimizer.step()


class DepthModel(nn.Module):
    def __init__(self):
        super(DepthModel, self).__init__()
        bottom_up_model = 'lateral_net.lateral_' + cfg.MODEL.ENCODER
        self.lateral_modules = get_func(bottom_up_model)()
        self.topdown_modules = lateral_net.fcn_topdown(cfg.MODEL.ENCODER)

    def forward(self, x, boxlist):
        lateral_out, backbone_stage_size = self.lateral_modules(x)
        # out: [nosoftmax, softmax]
        out, out_roi = self.topdown_modules(lateral_out, backbone_stage_size, boxlist)
        return out, out_roi

def cal_params(model):
    model_dict = model.state_dict()
    paras = np.sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum = 0

    for key in model_dict.keys():
        print(key)
        if 'layer5' not in key:
            if 'running' not in key:
                print(key)
                ss = model_dict[key].size()
                temp = 1
                for s in ss:
                    temp = temp * s
                print(temp)
                sum = sum + temp
    print(sum)
    print(paras)
