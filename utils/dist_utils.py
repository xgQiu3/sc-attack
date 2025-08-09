import torch.nn as nn
import torch
from pytorch3d.ops import knn_points, knn_gather

class KNNDist(nn.Module):

    def __init__(self, k=5, alpha=1.05):
        """Compute kNN distance punishment within a point cloud.

        Args:
            k (int, optional): kNN neighbor num. Defaults to 5.
            alpha (float, optional): threshold = mean + alpha * std. Defaults to 1.05.
        """
        super(KNNDist, self).__init__()

        self.k = k
        self.alpha = alpha

    def forward(self, pc, weights=None, batch_avg=True):
        """KNN distance loss described in AAAI'20 paper.

        Args:
            adv_pc (torch.FloatTensor): [B, K, 3]
            weights (torch.FloatTensor, optional): [B]. Defaults to None.
            batch_avg: (bool, optional): whether to avg over batch dim
        """
        # build kNN graph
        B, K = pc.shape[:2]
        if pc.shape[1] != 3:
            pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K], l2^2
        # print(dist.min().item())

        # assert dist.min().item() >= -1e-6

        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)
        # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # d_p, [B, K]
        with torch.no_grad():
            mean = torch.mean(value, dim=-1)  # [B]
            std = torch.std(value, dim=-1)  # [B]
            # [B], penalty threshold for batch
            threshold = mean + self.alpha * std
            weight_mask = (value > threshold[:, None]). \
                float().detach()  # [B, K]
        loss = torch.mean(value * weight_mask, dim=1)  # [B]
        # accumulate loss
        if weights is None:
            weights = torch.ones((B,))
        weights = weights.float().cuda()
        loss = loss * weights
        if batch_avg:
            return loss.mean()
        return loss


class CurvStdDist(nn.Module):
    def __init__(self, k=5):
        super(CurvStdDist, self).__init__()
        self.k = k

    def forward(self, ori_data, adv_data, ori_normal):
        pdist = torch.nn.PairwiseDistance(p=2)
        # fixme adv_data 使用 ori_normal 的影响有多大
        ori_kappa_std = self._get_kappa_std_ori(ori_data, ori_normal, k=self.k)  # [b, n]
        adv_kappa_std = self._get_kappa_std_ori(adv_data, ori_normal, k=self.k)  # [b, n]
        curv_std_dist = pdist(ori_kappa_std, adv_kappa_std).mean()
        return curv_std_dist

    def _get_kappa_std_ori(self, pc, normal, k=10):
        b, _, n = pc.size()
        # inter_dis = ((pc.unsqueeze(3) - pc.unsqueeze(2))**2).sum(1)
        # inter_idx = torch.topk(inter_dis, k+1, dim=2, largest=False, sorted=True)[1][:, :, 1:].contiguous()
        # nn_pts = torch.gather(pc, 2, inter_idx.view(b,1,n*k).expand(b,3,n*k)).view(b,3,n,k)
        inter_KNN = knn_points(pc.permute(0, 2, 1), pc.permute(0, 2, 1), K=k + 1)  # [dists:[b,n,k+1], idx:[b,n,k+1]]
        nn_pts = knn_gather(pc.permute(0, 2, 1), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                 1:].contiguous()  # [b, 3, n ,k]
        vectors = nn_pts - pc.unsqueeze(3)
        vectors = self._normalize(vectors)

        kappa_ori = torch.abs((vectors * normal.unsqueeze(3)).sum(1)).mean(2)  # [b, n]
        nn_kappa = knn_gather(kappa_ori.unsqueeze(2), inter_KNN.idx).permute(0, 3, 1, 2)[:, :, :,
                   1:].contiguous()  # [b, 1, n ,k]
        std_kappa = torch.std(nn_kappa.squeeze(1), dim=2)
        return std_kappa

    def _normalize(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, keepdim=True).clamp(min=eps).expand_as(input)