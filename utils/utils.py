# *_*coding:utf-8 *_*
import os
import numpy as np
import torch
import random
# import matplotlib.pyplot as plt
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import torch.nn.functional as F
import math
from pytorch3d.ops import knn_points, knn_gather


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def show_example(x, y, x_reconstruction, y_pred,save_dir, figname):
    x = x.squeeze().cpu().data.numpy()
    x = x.permute(0,2,1)
    y = y.cpu().data.numpy()
    x_reconstruction = x_reconstruction.squeeze().cpu().data.numpy()
    _, y_pred = torch.max(y_pred, -1)
    y_pred = y_pred.cpu().data.numpy()

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x, cmap='Greys')
    ax[0].set_title('Input: %d' % y)
    ax[1].imshow(x_reconstruction, cmap='Greys')
    ax[1].set_title('Output: %d' % y_pred)
    plt.savefig(save_dir + figname + '.png')

def save_checkpoint(epoch, model, path, modelnet='checkpoint', use_multi_gpu=False):
    savepath = os.path.join(path, '%s.pth' % (modelnet))
    if use_multi_gpu:
        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
        }
    else:
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
    torch.save(state, savepath)

def test(model, loader):
    total_correct = 0.0
    total_seen = 0.0
    for j, data in enumerate(loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = model.eval()
        with torch.no_grad():
            pred = classifier(points[:, :3, :], points[:, 3:, :])
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        total_correct += correct.item()
        total_seen += float(points.size()[0])

    accuracy = total_correct / total_seen
    return accuracy

def compute_cat_iou(pred,target,iou_tabel):
    iou_list = []
    target = target.cpu().data.numpy()
    for j in range(pred.size(0)):
        batch_pred = pred[j]
        batch_target = target[j]
        batch_choice = batch_pred.data.max(1)[1].cpu().data.numpy()
        for cat in np.unique(batch_target):
            # intersection = np.sum((batch_target == cat) & (batch_choice == cat))
            # union = float(np.sum((batch_target == cat) | (batch_choice == cat)))
            # iou = intersection/union if not union ==0 else 1
            I = np.sum(np.logical_and(batch_choice == cat, batch_target == cat))
            U = np.sum(np.logical_or(batch_choice == cat, batch_target == cat))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            iou_tabel[cat,0] += iou
            iou_tabel[cat,1] += 1
            iou_list.append(iou)
    return iou_tabel,iou_list

def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx].max(1) == part, target_np[shape_idx] == part))
            if U == 0:
                iou = 1 #If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious

def test_partseg(model, loader, catdict, num_classes = 50,forpointnet2=False):
    ''' catdict = {0:Airplane, 1:Airplane, ...49:Table} '''
    iou_tabel = np.zeros((len(catdict),3))
    iou_list = []
    metrics = defaultdict(lambda:list())
    hist_acc = []
    # mean_correct = []
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point,_= points.size()
        points, label, target, norm_plt = Variable(points.float()),Variable(label.long()), Variable(target.long()),Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(), label.squeeze().cuda(), target.cuda(), norm_plt.cuda()
        if forpointnet2:
            seg_pred = model(points, norm_plt, to_categorical(label, 16))
        else:
            labels_pred, seg_pred, _  = model(points,to_categorical(label,16))
            # labels_pred_choice = labels_pred.data.max(1)[1]
            # labels_correct = labels_pred_choice.eq(label.long().data).cpu().sum()
            # mean_correct.append(labels_correct.item() / float(points.size()[0]))
        # print(pred.size())
        iou_tabel, iou = compute_cat_iou(seg_pred,target,iou_tabel)
        iou_list+=iou
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        seg_pred = seg_pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['inctance_avg_iou'] = np.mean(iou_list)
    # metrics['label_accuracy'] = np.mean(mean_correct)
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()
    metrics['class_avg_iou'] = np.mean(cat_iou)

    return metrics, hist_acc, cat_iou

def test_semseg(model, loader, catdict, num_classes = 13, pointnet2=False):
    iou_tabel = np.zeros((len(catdict),3))
    metrics = defaultdict(lambda:list())
    hist_acc = []
    for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        batchsize, num_point, _ = points.size()
        points, target = Variable(points.float()), Variable(target.long())
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        if pointnet2:
            pred = model(points[:, :3, :], points[:, 3:, :])
        else:
            pred, _ = model(points)
        # print(pred.size())
        iou_tabel, iou_list = compute_cat_iou(pred,target,iou_tabel)
        # shape_ious += compute_overall_iou(pred, target, num_classes)
        pred = pred.contiguous().view(-1, num_classes)
        target = target.view(-1, 1)[:, 0]
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item()/ (batchsize * num_point))
    iou_tabel[:,2] = iou_tabel[:,0] /iou_tabel[:,1]
    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(metrics['accuracy'])
    metrics['iou'] = np.mean(iou_tabel[:, 2])
    iou_tabel = pd.DataFrame(iou_tabel,columns=['iou','count','mean_iou'])
    iou_tabel['Category_IOU'] = [catdict[i] for i in range(len(catdict)) ]
    # print(iou_tabel)
    cat_iou = iou_tabel.groupby('Category_IOU')['mean_iou'].mean()

    return metrics, hist_acc, cat_iou


def compute_avg_curve(y, n_points_avg):
    avg_kernel = np.ones((n_points_avg,)) / n_points_avg
    rolling_mean = np.convolve(y, avg_kernel, mode='valid')
    return rolling_mean

def plot_loss_curve(history,n_points_avg,n_points_plot,save_dir):
    curve = np.asarray(history['loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-g')

    curve = np.asarray(history['margin_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-b')

    curve = np.asarray(history['reconstruction_loss'])[-n_points_plot:]
    avg_curve = compute_avg_curve(curve, n_points_avg)
    plt.plot(avg_curve, '-r')

    plt.legend(['Total Loss', 'Margin Loss', 'Reconstruction Loss'])
    plt.savefig(save_dir + '/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M')) + '_total_result.png')
    plt.close()

def plot_acc_curve(total_train_acc,total_test_acc,save_dir):
    plt.plot(total_train_acc, '-b',label = 'train_acc')
    plt.plot(total_test_acc, '-r',label = 'test_acc')
    plt.legend()
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('Accuracy of training and test')
    plt.savefig(save_dir +'/'+ str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M'))+'_total_acc.png')
    plt.close()

def show_point_cloud(tuple,seg_label=[],title=None):
    import matplotlib.pyplot as plt
    if seg_label == []:
        x = [x[0] for x in tuple]
        y = [y[1] for y in tuple]
        z = [z[2] for z in tuple]
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    else:
        category = list(np.unique(seg_label))
        color = ['b','r','g','y','w','b','p']
        ax = plt.subplot(111, projection='3d')
        for categ_index in range(len(category)):
            tuple_seg = tuple[seg_label == category[categ_index]]
            x = [x[0] for x in tuple_seg]
            y = [y[1] for y in tuple_seg]
            z = [z[2] for z in tuple_seg]
            ax.scatter(x, y, z, c=color[categ_index], cmap='spectral')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
    plt.title(title)
    plt.show()



def set_seed(seed = 0):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



# 计算两个点集欧几里得距离的平方
def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    #print("xyz shape:", xyz.shape)  # 应该是 [B, N, 3]
    #print("new_xyz shape:", new_xyz.shape)  # 应该是 [B, S, 3]
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def grouping_operation(points, idx):
    """
    input: points(b, c, n)
            idx(b, npoints, nsample)
    output: (b, c, npoints, nsample)
    """
    B, C, N = points.shape
    _, npoints, nsample = idx.shape

    # [B,C,npoint,nsample]
    output = torch.zeros(
        (B, C, npoints, nsample), device=points.device, dtype=torch.float32
    )

    # plan 1
    # 遍历每个样本
    for b in range(B):
        # 提取当前样本的点云和索引
        points_b = points[b]  # [C, N]
        idx_b = idx[b]  # [npoints, nsample]

        # 遍历每个查询点
        for j in range(npoints):
            # 遍历每个样本点
            for k in range(nsample):
                # 获取索引值
                ii = idx_b[j, k]
                # 提取点
                output[b, :, j, k] = points_b[:, ii]

    '''
    # plan 2
    for b in range(B):
        for l in range(C):
            for j in range(npoints):
                for k in range(nsample):  # 遍历邻居点
                    ii = idx[b, j, k]  # 取出索引
                    output[b, l, j, k] = points[b, l, ii]  # 选取 points 中对应索引的数据，填入 output

    # plan 3
    # 扩展 idx 以匹配 points 维度
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1)  # (B, C, npoints, nsample)

    # 选取点: PyTorch 的 gather 作用是根据 idx 选取 points 对应的值
    output = torch.gather(points, dim=2, index=idx)  # (B, C, npoints, nsample)

    # plan 4
    for b in range(B):
        for j in range(npoints):
            output[b, :, j, :] = points[b, :, idx[b, j, :]]
    '''
    return output


def gather_operation(points, idx):
    B, C, N = points.shape
    _, n = idx.shape

    # [B,C,npoint]
    output = torch.zeros(
        (B, C, n), device=points.device, dtype=torch.float32
    )
    # 遍历每个样本
    for b in range(B):
        # 遍历每个特征维度
        for c in range(C):
            # 遍历每个索引
            for m in range(n):
                # 获取索引值
                index = idx[b, m]
                # 提取点并存储到输出
                output[b, c, m] = points[b, c, index]
    return output


def uniform_loss_metric(adv_pc, percentages=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0, k=2):
    if adv_pc.size(1) == 3:
        adv_pc = adv_pc.permute(0, 2, 1).contiguous()
    b, n, _ = adv_pc.size()
    npoint = int(n * 0.05)
    for p in percentages:
        p = p * 4
        nsample = int(n * p)
        r = math.sqrt(p * radius)
        disk_area = math.pi * (radius ** 2) * p / nsample
        expect_len = torch.sqrt(torch.Tensor([disk_area])).cuda()
        adv_pc_flipped = adv_pc.transpose(1, 2).contiguous()
        new_xyz = gather_operation(adv_pc_flipped,
                                        farthest_point_sample(adv_pc, npoint)
                                         ).transpose(1,2).contiguous()  # (batch_size, npoint, 3)

        idx = query_ball_point(r, nsample, adv_pc, new_xyz)  # (batch_size, npoint, nsample)
        adv_pc_flipped = adv_pc_flipped.cuda()
        idx = idx.cuda()
        grouped_pcd = grouping_operation(adv_pc_flipped, idx
                                               ).permute(0, 2, 3,1).contiguous()  # (batch_size, npoint, nsample, 3)
        grouped_pcd = torch.cat(torch.unbind(grouped_pcd, axis=1), axis=0)

        grouped_pcd = grouped_pcd.permute(0, 2, 1).contiguous()
        # dis = torch.sqrt(((grouped_pcd.unsqueeze(3) - grouped_pcd.unsqueeze(2))**2).sum(1)+1e-12) # (batch_size*npoint, nsample, nsample)
        # dists, _ = torch.topk(dis, k+1, dim=2, largest=False, sorted=True) # (batch_size*npoint, nsample, k+1)
        inter_KNN = knn_points(grouped_pcd.permute(0, 2, 1), grouped_pcd.permute(0, 2, 1),
                               K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]

        # inter_KNN : [B*npoint, nsample, k+1]
        uniform_dis = inter_KNN.dists[:, :, 1:].contiguous()
        uniform_dis = torch.sqrt(torch.abs(uniform_dis) + 1e-12)
        uniform_dis = uniform_dis.mean(axis=[-1])
        uniform_dis = (uniform_dis - expect_len) ** 2 / (expect_len + 1e-12)
        uniform_dis = torch.reshape(uniform_dis, [-1])

        mean = uniform_dis.mean()
        mean = mean * math.pow(p * 100, 2)

        # nothing 4
        try:
            loss = loss + mean
        except:
            loss = mean
    return loss / len(percentages)
