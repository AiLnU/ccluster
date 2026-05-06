# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([2] + layers + [feature_dim])#NOTEsp 根据实际需求 3改成2 没有score
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        inputs = kpts.transpose(1, 2)
        # return self.encoder(torch.cat(inputs, dim=1))
        return self.encoder(inputs)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                raise ValueError
            else:  # if name == 'self':
                src0 = desc0
            delta0 = layer(desc0, src0)
            desc0 = (desc0 + delta0)
        return desc0


def collect_same_value_indices(tensor):
    """
    收集张量中每个相同值的所有索引位置
    返回：
        Dict[值(int或float), Tensor]：键为唯一值，值为对应的索引张量
    """
    unique_values = torch.unique(tensor)
    batch_groups = []

    for val in unique_values:
        # 找到所有等于当前值的坐标
        indices = torch.nonzero(tensor == val, as_tuple=True)

        # 处理一维张量的特殊情况（去掉多余的维度）
        if tensor.dim() == 1:
            indices = indices.squeeze(1)

        batch_groups.append(indices)

    return batch_groups


class MultiPositiveContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, base_temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        features: 输入特征 [180, d]
        labels: 目标标签 [180] (0-19)
        """
        device = features.device
        batch_size = features.size(0)

        # 归一化特征向量
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵 [180, 180]
        sim_matrix = torch.matmul(features, features.T)

        # 创建标签掩码 [180, 180]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 排除自身相似度
        self_mask = torch.eye(batch_size, device=device)
        mask = mask * (1 - self_mask)

        # 计算正样本对数概率
        logits = sim_matrix / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # 数值稳定

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # 计算每个样本的平均正样本对数概率
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)

        # 损失计算
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def group_constraint_loss(assignments, groups, temperature=0.1):
    """
    组约束损失函数
    assignments: 分配概率 [batch_size, num_clusters]
    groups: 当前批次中的约束组列表 [[idx1, idx2,...], ...]
    """
    must_loss = 0.0
    cannot_loss = 0.0
    for group in groups:
        if len(group) < 2:
            continue

        # 提取组内样本的分配概率
        group_assign = assignments[group]  # [group_size, num_clusters]

        # 计算组内两两之间的KL散度
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                p = group_assign[i]
                q = group_assign[j]
                must_loss += torch.norm(p - q, p=2) ** 2 #Notesp 不再用ＫＬ散度　为了和ｃａｎｎｏｔ损失平衡

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group_assign_in = assignments[groups[i]] #当前族的分配
            group_assign_out = assignments[groups[j]] #其他族的分配
            for m in range(len(groups[i])):
                for n in range(len(groups[j])):
                    p = group_assign_in[m]
                    q = group_assign_out[n]
                    cannot_loss += torch.sum(p * q)


            # 对称KL散度
                # kl_loss = 0.5 * (
                #         F.kl_div(torch.log(p + 1e-8), q, reduction='batchmean') +
                #         F.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')
                # )
                # loss += kl_loss
    return must_loss, cannot_loss


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def extract_values_method1(A, indices):
    # 创建第一个维度的索引 [100, 7]
    batch_idx = torch.arange(A.size(0))[:, None].expand(-1, indices.size(1))

    # 使用高级索引提取数据
    result = A[batch_idx, indices, :]
    return result


class GroupContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        支持多batch的组对比损失函数

        Args:
            temperature (float): 温度参数，控制相似度分布的尖锐程度
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, features, group_labels):
        """
        Args:
            features (Tensor): 形状为 [batch_size, num_vectors, feat_dim]
            group_labels (Tensor): 形状为 [batch_size, num_vectors]
        Returns:
            loss (Tensor): 对比损失值
        """
        device = features.device
        batch_size, num_vectors, feat_dim = features.shape

        # 1. 展平batch维度以便矩阵计算
        flat_features = features.reshape(-1, feat_dim)  # [batch_size * num_vectors, feat_dim]
        flat_labels = group_labels.view(-1)  # [batch_size * num_vectors]

        # 2. 归一化特征向量
        norm_features = F.normalize(flat_features, dim=1)

        # 3. 计算余弦相似度矩阵
        sim_matrix = torch.matmul(norm_features, norm_features.T)  # [B*N, B*N]
        sim_matrix /= self.temperature

        # 4. 创建batch掩码（仅同batch内向量比较）
        batch_range = torch.arange(batch_size, device=device)
        batch_idx = batch_range.view(-1, 1).repeat(1, num_vectors).view(-1)
        batch_mask = (batch_idx.unsqueeze(1) == batch_idx.unsqueeze(0))

        # 5. 创建组掩码（同组内向量）
        group_mask = (flat_labels.unsqueeze(1) == flat_labels.unsqueeze(0))

        # 6. 创建正样本掩码（同batch、同组、非自身）
        eye_mask = ~torch.eye(batch_size * num_vectors, device=device).bool()
        pos_mask = batch_mask & group_mask & eye_mask

        # 7. 创建负样本掩码（同batch、不同组）
        neg_mask = batch_mask & ~group_mask

        # 8. 计算分子（正样本相似度）
        # pos_sim = sim_matrix[pos_mask].view(batch_size * num_vectors, -1)  # [B*N, 8]

        # 9. 计算分母（负样本指数和）
        max_sim = sim_matrix.detach().max(dim=1, keepdim=True)[0]  # 数值稳定
        exp_sim = torch.exp(sim_matrix - max_sim)  # 减去最大值避免指数爆炸

        # 10. 计算每个样本的对比损失
        pos_sum = torch.sum(exp_sim * pos_mask, dim=1, keepdim=True)  # 分子
        neg_sum = torch.sum(exp_sim * neg_mask, dim=1, keepdim=True)  # 分母中的负样本部分

        # 11. 避免数值问题（确保分母不为零）
        denom = pos_sum + neg_sum + 1e-10

        # 12. 计算每个样本的损失
        # loss_per_sample = -torch.log(pos_sum / denom)
        loss_per_sample = 1-pos_sum / denom

        # 13. 计算有效样本的平均损失
        valid_mask = (pos_mask.sum(dim=1)) > 0  # 确保有正样本

        if torch.any(valid_mask):
            loss = loss_per_sample[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return loss


class Conv1DNet(nn.Module):
    def __init__(self, input_length=20, input_dim=4, output_dim=1):
        super(Conv1DNet, self).__init__()
        self.conv = nn.Sequential(
            # 保持长度不变: padding=(kernel_size-1)//2
            nn.Conv1d(in_channels=input_dim, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=output_dim, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x 形状: (batch_size, 20, 4) → 转置为 (batch_size, 4, 20)
        # x = x.transpose(2, 3)
        # 卷积处理: (batch_size, 4, 20) → (batch_size, 1, 20)
        x_reshape = x.reshape(-1, x.size(2), x.size(3))
        out = self.conv(x_reshape)
        # 转置回: (batch_size, 20, dim)
        return out.reshape(x.size(0), x.size(1), x.size(3))


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 4, #NOTEsp 依据实际需求修改
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128],
        # 'GNN_layers': ['self', 'cross'] * 9,
        'GNN_layers': ['self'] * 9, #NOTEsp 仅用了self
        'sinkhorn_iterations': 100,
        'match_threshold': 0.5,
        'use_other': False,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        # if self.config['descriptor_dim']<2:
        #     self.config['descriptor_dim'] = 4

        self.num_classes = 20
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        self.temperature = 1.0
        self.criterion = GroupContrastiveLoss()

        if self.config['use_other']:
            self.conv = Conv1DNet()

    def _init_centers(self, x):
        """可微的聚类中心初始化"""
        batch_size = x.size(0)

        # 方案1：基于特征统计量的初始化
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        centers = mean + torch.randn(
            self.num_clusters, x.size(-1),
            device=x.device
        ) * std

        # 方案2：随机选择初始化（保持可微）
        # indices = torch.randperm(batch_size)[:self.num_clusters]
        # centers = x[indices] + torch.randn_like(x[indices])*0.01
        return centers

    def constrained_kmeans(self, features, UAV_num, num_classes=20, max_iters=100):
        """
        对单个样本的特征向量进行约束 K-means 聚类。

        参数:
            features: 形状为 [180, 32] 的张量，特征向量按无人机顺序排列。
            num_classes: 类数, 默认 20。
            max_iters: 最大迭代次数, 默认 100。

        返回:
            assignments: 形状为 [180] 的张量，每个向量的类 ID (0-19)。
        """
        num_drones = UAV_num  # 无人机数量
        vectors_per_drone = 20  # 每个无人机的向量数
        if self.config['part_covis']:
            vectors_per_drone = 14 # 30%漏检

        # L2 归一化特征向量（重要：使欧氏距离等价于余弦距离）
        features = F.normalize(features, p=2, dim=1)

        # 初始化分配：每个无人机随机分配其向量到不同类
        assignments = torch.zeros(features.size(0), dtype=torch.long)
        for u in range(num_drones):
            start_idx = u * vectors_per_drone
            end_idx = (u + 1) * vectors_per_drone
            perm = torch.randperm(num_classes)  # 随机排列类 ID
            if self.config['part_covis']:
                perm = perm[:14]  # 30%漏检 取前１４个，也是随机

            assignments[start_idx:end_idx] = perm

        # 迭代优化
        centroids = torch.zeros(num_classes, features.size(1)).cuda()
        for iter in range(max_iters):
            prev_assignments = assignments.clone()

            # 步骤 1: 计算当前类质心
            for j in range(num_classes):
                mask = (assignments == j)
                if torch.any(mask): # 如果没有Ｔｒｕｅ　就用上一次的中心
                    centroids[j] = features[mask].mean(dim=0)  # 类质心

            # 步骤 2: 对每个无人机重新分配向量
            for u in range(num_drones):
                start_idx = u * vectors_per_drone
                end_idx = (u + 1) * vectors_per_drone
                drone_vecs = features[start_idx:end_idx]  # 当前无人机的向量, [20, 32]

                # 计算成本矩阵: 行是向量, 列是类, 成本为欧氏距离平方
                # 公式: ||v - c||^2 = ||v||^2 + ||c||^2 - 2 * v·c (归一化后 ||v||^2=1)
                dot_products = torch.mm(drone_vecs, centroids.t())  # [20, 20]
                norms_cent = torch.sum(centroids ** 2, dim=1)  # [20]
                cost_matrix = 1 + norms_cent.unsqueeze(0) - 2 * dot_products  # [20, 20]

                # 使用匈牙利算法解决分配问题 (最小化总成本)
                cost_np = cost_matrix.detach().cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(cost_np)
                # col_ind 给出每个向量分配的新类 ID
                new_assign = torch.tensor(col_ind, dtype=torch.long)

                # 更新当前无人机的分配
                assignments[start_idx:end_idx] = new_assign

            # 检查收敛: 如果分配不再变化, 则停止
            if torch.all(assignments == prev_assignments):
                break

        return assignments

    def forward(self, data, temp=None,lambda_constraint=0.5):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        # TODO 后续可以放到数据集部分
        if self.config['use_other']:
            batch, _, _, featuredim = data[0]['descriptors'].shape
        else:
            batch, _, featuredim = data[0]['descriptors'].shape

        UAV_num = len(data)
        inputfeature_ = torch.empty((batch, featuredim, 0))
        labels_ = torch.empty((batch, 0))
        inputfeature, labels = inputfeature_.cuda(), labels_.cuda()
        if not self.training:
            kpts_all = torch.empty((batch, 0, 2)).cuda()

        for feature in data:
            kpts, desc = feature['keypoints'], feature['descriptors']#.transpose(1, 2)
            # if self.config['use_other']:
            #     desc = self.conv(desc)

            label = feature['label_cluster']

            if self.config['part_covis']:
                desc = extract_values_method1(desc, feature['rdm'].transpose(0, 1))
                kpts = extract_values_method1(kpts, feature['rdm'].transpose(0, 1))
                label = feature['label_cluster_rdm']

            if self.config['use_other']:
                desc = self.conv(desc)
            if not self.training:
                kpts_all = torch.cat((kpts_all, kpts), dim=1)

            labels = torch.cat((labels, label), dim=1)
            desc = desc.transpose(1,2)
            desc = desc + self.kenc(kpts)
            desc = self.gnn(desc)  # TODO 也可以尝试加入与其他无人机相互关联的注意力
            desc = self.final_proj(desc)
            inputfeature = torch.cat((inputfeature, desc), dim=2)

        embeddings = inputfeature.transpose(1, 2)
        # groups = collect_same_value_indices(labels)
        loss = self.criterion(embeddings, labels)

        # if not self.training:
        if self.training:
            groups, all_assignments, kpts_all = None, None, None
        else:
            all_assignments = []
            for embedding in embeddings:
                assignments = self.constrained_kmeans(embedding, UAV_num, self.num_classes)
                all_assignments.append(assignments)
            all_assignments = torch.stack(all_assignments, dim=0)
            groups = collect_same_value_indices(labels)

        return {
            'indices': all_assignments,
            'groups': groups,
            'loss': loss,
            'kpts': kpts_all
        }

        # scores big value or small value means confidence? log can't take neg value

