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
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import adjusted_rand_score as ARI
import numpy as np
from torch.autograd import Variable


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


class DeepClusterModel(nn.Module):
    def __init__(self, input_dim=20, num_classes=13, embed_dim=20):
        super().__init__()
        # 特征嵌入网络
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim))

        # 可学习聚类中心
        self.num_classes = num_classes
        self.cluster_centers = nn.Parameter(
            torch.randn(num_classes, embed_dim))

        # 辅助分类器
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # embeddings = self.embedder(x)
        # logits = self.classifier(x.transpose(1,2)) #NOTEsp  这里不再学了
        return x.transpose(1, 2)#, logits


class ClusterLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings, groups, assignments, centers):

        # 聚类中心约束
        distances = torch.cdist(embeddings, centers)

        batch_centers = centers[labels]
        cluster_loss = torch.mean(
            torch.norm(embeddings - batch_centers, dim=1))#TODO 这里已经取过平均值了

        var = torch.mean(torch.var(centers, dim=0, unbiased=True))#TODO 这里也可以用标准差
        cluster_loss = cluster_loss/var

        # 正则化项
        # reg_term = torch.mean(torch.norm(centers, dim=1)) #TODO 正则化项的意义是什么？
        # return cls_loss + self.alpha * cluster_loss + 0.1 * reg_term
        return cluster_loss


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


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


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


def group_constraint_loss(assignments, groups, temperature=0.1):
    """
    组约束损失函数
    assignments: 分配概率 [batch_size, num_clusters]
    groups: 当前批次中的约束组列表 [[idx1, idx2,...], ...]
    """
    loss = 0.0
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

                # 对称KL散度
                kl_loss = 0.5 * (
                        F.kl_div(torch.log(p + 1e-8), q, reduction='batchmean') +
                        F.kl_div(torch.log(q + 1e-8), p, reduction='batchmean')
                )
                loss += kl_loss
    return loss


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
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.criterion = ClusterLoss(0.5)
        self.DeepCluster = DeepClusterModel(20, 20, 20)# 第二个是目标数量

        self.num_clusters = 20
        self.num_iterations = 3
        self.hard = False

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)
        self.temperature = 1.0
        # bin_score = torch.nn.Parameter(torch.tensor(1.))
        # self.register_parameter('bin_score', bin_score)

    def _init_centers(self, x):
        """可微的聚类中心初始化"""
        # batch_size = x.size(0)

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

    def forward(self, data, temp=None,lambda_constraint=1.0):
        """Run SuperGlue on a pair of keypoints and descriptors"""

        # TODO 后续可以放到数据集部分
        batch, num_object, featuredim = data[0]['descriptors'].shape
        inputfeature_ = torch.empty((batch, featuredim, 0))
        labels_ = torch.empty((batch, 0))
        inputfeature, labels = inputfeature_.cuda(), labels_.cuda()
        for feature in data:
            desc = feature['descriptors'].transpose(1, 2)
            desc = desc + self.kenc(feature['keypoints'])
            desc = self.gnn(desc)  # TODO 之后只用自注意力
            desc = self.final_proj(desc)
            inputfeature = torch.cat((inputfeature, desc), dim=2)
            labels = torch.cat((labels, feature['label_cluster']), dim=1)


        # NOTEsp 删除部分点时要用
        '''
        n0, n1 = data[0]['rdm'][:, 0], data[1]['rdm'][:, 0]

        unique_vec0, _ = torch.unique(n0, sorted=True, return_inverse=True)
        unique_vec1, _ = torch.unique(n1, sorted=True, return_inverse=True)
        # Notesp alt 是否随机遮盖部分点,从这选择
        # desc0, kpts0, desc1, kpts1 = desc01[:, unique_vec0], kpts01[:, unique_vec0], desc11[:, unique_vec1], kpts11[:, unique_vec1]
        desc0, kpts0, desc1, kpts1 = desc01, kpts01, desc11, kpts11
        '''
        embeddings = self.DeepCluster(inputfeature)# feature transform

        # section 这一段时计算聚类中心的，可以改成可学习的参数


        # 迭代优化中心
        centers = torch.empty((0, num_object, featuredim)).cuda()
        for embedding in embeddings:
            center = self._init_centers(embedding)
            for _ in range(self.num_iterations):
                # 计算样本-中心相似度
                distances = torch.cdist(embedding, center)  # [B, K]
                logits = -distances

                # Gumbel-Softmx分配
                tau = temp if temp is not None else self.temperature
                assignment = F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=-1)

                # 更新聚类中心（加权平均）
                sum_weights = assignment.sum(dim=0).unsqueeze(-1) + 1e-8  # [K,1]
                center = torch.mm(assignment.t(), embedding) / sum_weights  # [K,D]

                # 添加稳定性正则化
                center = center + (torch.randn_like(center) * 0.01)  # 噪声注入
            centers = torch.cat((centers, center.unsqueeze(0)), dim=0)


        # final_distances = torch.cdist(embeddings, centers)
        # tau = temp if temp is not None else self.temperature
        # final_assignment = F.gumbel_softmax(-final_distances, tau=tau, hard=self.hard)

        # distances = torch.cdist(embeddings, self.DeepCluster.cluster_centers)
        distances = torch.cdist(embeddings, centers) # Notesp Alt 选择迭代的中心
        logits = -distances  # 使用负距离作为logits
        assignments = F.softmax(logits, dim=-1)

        # section 计算损失
        # labels = torch.cat((data[0]['label_cluster'], data[1]['label_cluster']), dim=1)
        cluster_loss = (assignments * distances).mean()
        groups = collect_same_value_indices(labels)
        const_loss = group_constraint_loss(assignments, groups)
        loss = cluster_loss + lambda_constraint * const_loss

            # loss = self.criterion(embeddings, groups, assignments, self.DeepCluster.cluster_centers)
        # else:
        #     kmeans = KMeans(n_clusters=self.DeepCluster.num_classes)
        #     nmi = []
        #     ari = []
        #
        #     for embedding, label in zip(embeddings, labels):
        #         cluster_ids = kmeans.fit_predict(embedding.detach().cpu().numpy())
        #         nmi.append(NMI(label.squeeze().cpu().numpy(), cluster_ids))
        #         ari.append(ARI(label.squeeze().cpu().numpy(), cluster_ids))
                # print(f"NMI Score: {NMI(labels.squeeze().cpu().numpy(), cluster_ids):.4f}")
                # print(f"ARI Score: {ARI(labels.squeeze().cpu().numpy(), cluster_ids):.4f}")
            # a=1

        # pass

        """
        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, max1.values.exp(), zero)
        # mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & (mscores1 > self.config['match_threshold'])
        # valid0 = mutual0
        # valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices0.new_tensor(-1))
        # indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        num_yes = torch.sum(all_matches[:, :-1, :-1].max(2).indices == indices0)+torch.sum(all_matches[:, :-1, :-1].max(1).indices == indices1)
        num_none = torch.sum(-1 == indices0)+torch.sum(-1 == indices1)
        num_no = (all_matches.shape[1]-1+all_matches.shape[2]-1)*all_matches.shape[0]-num_yes-num_none
        num_sum = torch.sum(all_matches[:, :-1, :-1])*2 # 应该有的正确匹配的数量,a->b,b->a,所以乘以2

        # check if indexed correctly
        loss = []

        # rdm0, rdm1 = data['rdm0'], data['rdm1']

        log_p = torch.log(abs(scores.exp()) + 1e-9)
        indices_p = torch.where(all_matches == 1)

        loss_mean = -log_p[indices_p].mean()
        
        """




        '''
        for cur_log_p in log_p:
            # cur_log_p = log_p[i]
            # loss = -torch.diag(cur_log_p)[:-1].mean()
            loss.append(-torch.diag(cur_log_p)[:-1].mean())
            # loss.append(-torch.log(scores[0][i][i].exp()))

        # for i in range(len(all_matches[0])):
        #     x = all_matches[0][i][0]
        #     y = all_matches[0][i][1]
        #     loss.append(-torch.log(scores[0][x][y].exp() )) # check batch size == 1 ?


        # for p0 in unmatched0:
        #     loss += -torch.log(scores[0][p0][-1])
        # for p1 in unmatched1:
        #     loss += -torch.log(scores[0][-1][p1])
        loss_mean = torch.mean(torch.stack(loss))
        '''
        # loss_mean = torch.reshape(loss_mean, (1, -1))
        return {
            # 'num_yes': num_yes, # use -1 for invalid match
            # 'num_none': num_sum, # use -1 for invalid match TODO
            # 'num_no': num_no,
            # 'indices0': indices0,
            # 'trueorfalse': all_matches[:, :-1, :-1].max(2).indices == indices0,
            'groups': groups,
            'assign': assignments,
            # # 'matching_scores1': mscores1[0],
            'loss': loss,
            # 'skip_train': False
        }

        # scores big value or small value means confidence? log can't take neg value
