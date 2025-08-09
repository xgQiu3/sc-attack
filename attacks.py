import os
from pickle import FALSE
import sys
import numpy as np
from collections import Iterable
import importlib
import open3d as o3d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from utils.set_distance import ChamferDistance, HausdorffDistance
from baselines import *
from scipy.ndimage import gaussian_filter
from torch.nn.utils import clip_grad_norm_

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))


class PointCloudAttack(object):
    def __init__(self, args):
        """Shape-invariant Adversarial Attack for 3D Point Clouds.
        """
        self.args = args
        self.device = args.device

        self.eps = args.eps
        self.normal = args.normal
        self.step_size = args.step_size
        self.num_class = args.num_class
        self.max_steps = args.max_steps
        self.top5_attack = args.top5_attack

        self.build_models()
        self.defense_method = args.defense_method
        if not args.defense_method is None:
            self.pre_head = self.get_defense_head(args.defense_method)

    def build_models(self):
        """Build models and register hooks for intermediate layers."""
        ################################################
        # 加载白盒模型1
        MODEL = importlib.import_module(self.args.surrogate_model_1)
        wb_classifier_1 = MODEL.get_model(self.num_class, normal_channel=self.normal)
        wb_classifier_1 = wb_classifier_1.to(self.device)
        # 加载权重
        wb_classifier_1 = self.load_models(wb_classifier_1, self.args.surrogate_model_1)
        # 设置为评估模式
        self.wb_classifier_1 = wb_classifier_1.eval()
        #############################################
        # 加载白盒模型1
        MODEL = importlib.import_module(self.args.surrogate_model_2)
        wb_classifier_2 = MODEL.get_model(self.num_class, normal_channel=self.normal)
        wb_classifier_2 = wb_classifier_2.to(self.device)
        # 加载权重
        wb_classifier_2 = self.load_models(wb_classifier_2, self.args.surrogate_model_2)
        # 设置为评估模式
        self.wb_classifier_2 = wb_classifier_2.eval()
        ###################################################
        # 加载黑盒模型
        MODEL = importlib.import_module(self.args.target_model)
        classifier = MODEL.get_model(self.num_class, normal_channel=self.normal)
        classifier = classifier.to(self.args.device)
        # 加载权重
        classifier = self.load_models(classifier, self.args.target_model)
        self.classifier = classifier.eval()


    def load_models(self, classifier, model_name):
        """Load white-box surrogate model and black-box target model.
        """
        model_path = os.path.join('./checkpoint/' + self.args.dataset, model_name)
        if os.path.exists(model_path + '.pth'):
            checkpoint = torch.load(model_path + '.pth')
        elif os.path.exists(model_path + '.t7'):
            checkpoint = torch.load(model_path + '.t7')
        elif os.path.exists(model_path + '.tar'):
            checkpoint = torch.load(model_path + '.tar')
        else:
            raise NotImplementedError

        try:
            if 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            elif 'model_state' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state'])
            else:
                classifier.load_state_dict(checkpoint)
        except:
            classifier = nn.DataParallel(classifier)
            classifier.load_state_dict(checkpoint)
        return classifier

    def CWLoss(self, logits, target, kappa=0, tar=False, num_classes=40):
        """Carlini & Wagner attack loss.

        Args:
            logits (torch.cuda.FloatTensor): the predicted logits, [1, num_classes].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
        target_one_hot = Variable(torch.eye(num_classes).type(torch.cuda.FloatTensor)[target.long()].cuda())

        real = torch.sum(target_one_hot * logits, 1)
        if not self.top5_attack:
            ### top-1 attack
            other = torch.max((1 - target_one_hot) * logits - (target_one_hot * 10000), 1)[0]
        else:
            ### top-5 attack
            other = torch.topk((1 - target_one_hot) * logits - (target_one_hot * 10000), 5)[0][:, 4]
        kappa = torch.zeros_like(other).fill_(kappa)

        if tar:
            return torch.sum(torch.max(other - real, kappa))
        else:
            return torch.sum(torch.max(real - other, kappa))

    def run(self, istrain, points, target, bert, ae, vae):
        """Main attack method.

        Args:
            points (torch.cuda.FloatTensor): the point cloud with N points, [1, N, 6].
            target (torch.cuda.LongTensor): the label for points, [1].
        """
        return self.shape_invariant_ifgm(istrain, points, target, bert, ae, vae)

    def get_defense_head(self, method):
        """Set the pre-processing based defense module.

        Args:
            method (str): defense method name.
        """
        if method == 'sor':
            pre_head = SORDefense(k=2, alpha=1.1)
        elif method == 'srs':
            pre_head = SRSDefense(drop_num=500)
        elif method == 'dupnet':
            pre_head = DUPNet(sor_k=2, sor_alpha=1.1, npoint=1024, up_ratio=4)
        else:
            raise NotImplementedError
        return pre_head
#################### ############### ############### ############### ############### ############### ###############

    def shape_invariant_ifgm(self, istrain, points, target, bert, ae, vae):
        ori_v = points[:, :, -3:].data  # N, [1, N, 3]
        ori_v = ori_v / torch.sqrt(torch.sum(ori_v ** 2, dim=-1, keepdim=True))  # N, [1, N, 3]
        bert.eval()
        points = points[:, :, :3].data  # P, [1, N, 3]
        chamfer_loss = ChamferDistance()
        hau_loss = HausdorffDistance()
        clip_func = ClipPointsLinf(0.18)
        ###########################
        logits = self.classifier(points.transpose(1, 2))
        incorrect_indices = (logits.data.max(1)[1] != target).nonzero().squeeze()
        black_clean_cor = torch.ones_like(target, dtype=torch.int64)
        black_clean_cor[incorrect_indices] = 0
        ###########################

        z = ae.encode(points).cuda()
        adv_data = ae.decode(code=z, num_points=points.size(1), points=points, target=target, dist_fun=chamfer_loss,
                             bert=bert, CWLoss=self.CWLoss, victim=self.wb_classifier_1, flexibility=0.0).detach()

        #############################
        adv_data_2 = adv_data.clone().detach()

        logits = self.classifier(adv_data_2.transpose(1, 2))
        correct_indices = (logits.data.max(1)[1] == target).nonzero().squeeze()
        black_pert_cor = torch.ones_like(target, dtype=torch.int64)
        black_pert_cor[correct_indices] = 0
        clean_sum = torch.sum(black_clean_cor)
        both_success = black_clean_cor & black_pert_cor
        pert_sum = torch.sum(both_success)

        return adv_data_2, pert_sum.item(), clean_sum.item()

