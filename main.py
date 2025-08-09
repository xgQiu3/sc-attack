# -*- coding: utf-8 -*-


import os
import argparse
import sys
import numpy as np
from tqdm import tqdm
import torch
from data_utils.ModelNetDataLoader import ModelNetDataLoader
from data_utils.ShapeNetDataLoader import PartNormalDataset
from torch.utils.data import DataLoader
from utils.utils import set_seed
from attacks import PointCloudAttack
from utils.set_distance import ChamferDistance, HausdorffDistance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))
x_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'attention')
sys.path.append(x_path)
x_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusion')
sys.path.append(x_path)
from attention import conf, test_net
from ae import my_ae, get_ae



def load_data(args):
    """Load the dataset from the given path.
    """
    print('Start Loading Dataset...')
    if args.dataset == 'ModelNet40':
        TEST_DATASET = ModelNetDataLoader(
            root=args.data_path,
            npoint=args.input_point_nums,
            split='test',
            #category='bed',
            normal_channel=True
        )
    elif args.dataset == 'ShapeNetPart':
        TEST_DATASET = PartNormalDataset(
            root=args.data_path,
            npoints=args.input_point_nums,
            #class_choice=['Chair'],
            split='test',
            normal_channel=True
        )
    else:
        raise NotImplementedError

    testDataLoader = torch.utils.data.DataLoader(
        TEST_DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('Finish Loading Dataset...')
    return testDataLoader



def data_preprocess(data):
    """Preprocess the given data and label.
    """
    if args.dataset == 'ModelNet40':
        points, target = data
    elif args.dataset == 'ShapeNetPart':
        points, target, _ = data

    points = points # [B, N, C]
    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target


def save_tensor_as_txt(points, filename):
    """Save the torch tensor into a txt file.
    """
    points = points.squeeze(0).detach().cpu().numpy()
    with open(filename, "a") as file_object:
        for i in range(points.shape[0]):
            # msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2])
            msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]) + \
                ' ' + str(points[i][3].item()) +' ' + str(points[i][3].item()) + ' '+ str(1-points[i][3].item())
            file_object.write(msg+'\n')
        file_object.close()
    print('Have saved the tensor into {}'.format(filename))


def main():
    # load data
    test_loader = load_data(args)

    num_class = 0
    if args.dataset == 'ModelNet40':
        num_class = 40
    elif args.dataset == 'ShapeNetPart':
        num_class = 16
    assert num_class != 0
    args.num_class = num_class

    # load model
    attack = PointCloudAttack(args)

    # start attack
    atk_success = 0
    black_clean_sum = 0
    avg_chamfer_dist = 0.
    avg_hausdorff_dist = 0.
    avg_l2_dist = 0.

    chamfer_loss = ChamferDistance()
    hausdorff_loss = HausdorffDistance()

    bert = test_net(conf)
    bert = bert.cuda()
    for param in bert.parameters():
        param.requires_grad = False
    bert.eval()

    ae = my_ae(args.dataset)
    ae.eval()
    #vae = get_vae()
    vae = test_net(conf)
    vae.eval()

    for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        points, target = data_preprocess(data)
        target = target.long()
        points = points.cuda()    # [B,N,6]
        target = target.cuda()    # [B]

        ori_normal = points[:, :, -3:].data.transpose(1, 2).contiguous()
        ori_points = points[:, :, :3].data.transpose(1, 2).contiguous()

        points_np = points[0, :, :3].data.cpu().numpy() 
        np.savetxt("./txt/my_ori" + str(batch_id) + ".txt", points_np.reshape(-1, 3), fmt='%.6f')

        # start attack
        adv_points, black_pert, black_clean = attack.run(True, points, target, bert, ae, vae)

        adv_points_np = adv_points[0].cpu().detach().numpy()  
        np.savetxt("./txt/my_si" + str(batch_id) + ".txt", adv_points_np.reshape(-1, 3), fmt='%.6f')

        atk_success += black_pert
        black_clean_sum += black_clean

        points = points[:, :, :3].data
        avg_chamfer_dist += chamfer_loss(adv_points, points).mean()
        avg_hausdorff_dist += hausdorff_loss(adv_points, points).mean()
        avg_l2_dist += torch.norm(adv_points - points, dim=2).mean(dim=1).mean()


    Trans = atk_success / black_clean_sum
    print("正确率", Trans)
    print("黑盒在干净样本上本身正确分类：", black_clean_sum)
    print("黑盒被成功攻击：", atk_success)

    avg_chamfer_dist /= batch_id + 1
    print('Average Chamfer Dist:', avg_chamfer_dist.item())
    avg_hausdorff_dist /= batch_id + 1
    print('Average Hausdorff Dist:', avg_hausdorff_dist.item())
    avg_l2_dist /= batch_id + 1
    print('Average L2 Dist:', avg_l2_dist.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Shape-Consistent Adversarial Point Cloud Generation via Conditional Diffusion Model')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=2500,
                        help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2025, metavar='S',
                        help='random seed (default: 2025)')
    parser.add_argument('--dataset', type=str, default='ModelNet40',
                        choices=['ModelNet40', 'ShapeNetPart'])
    parser.add_argument('--data_path', type=str,
                        default='/data/modelnet40_normal_resampled/')
    parser.add_argument('--normal', action='store_true', default=False,
                        help='Whether to use normal information [default: False]')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Worker nums of data loading.')
    parser.add_argument('--surrogate_model_1', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
    parser.add_argument('--surrogate_model_2', type=str, default='pointnet2_cls_msg',
                        choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct',
                                 'curvenet', 'simple_view'])
    parser.add_argument('--target_model', type=str, default='pointnet_cls',
                        choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn', 'pointconv', 'pointcnn', 'paconv', 'pct', 'curvenet', 'simple_view'])
    parser.add_argument('--defense_method', type=str, default=None,
                        choices=['sor', 'srs', 'dupnet'])
    parser.add_argument('--top5_attack', action='store_true', default=False,
                        help='Whether to attack the top-5 prediction [default: False]')

    parser.add_argument('--max_steps', default=50, type=int,
                        help='max iterations for black-box attack')
    parser.add_argument('--eps', default=0.16, type=float,
                        help='epsilon of perturbation')
    parser.add_argument('--step_size', default=0.07, type=float,
                        help='step-size of perturbation')
    args = parser.parse_args()

    # basic configuration
    set_seed(args.seed)  #设置随机种子
    args.device = torch.device("cuda")

    # main loop
    main()
