import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


def constrained_clustering(features, num_classes=20, class_size=9):
    """
    带约束的聚类算法：每个类固定大小，且考虑无人机约束
    Args:
        features: 形状为 [180, 32] 的特征向量
        num_classes: 类别数 (20)
        class_size: 每类样本数 (9)
    Returns:
        labels: 形状为 [180] 的聚类标签
    """
    # 1. 特征归一化
    features = F.normalize(features, p=2, dim=1)

    # 2. 初始化中心点 - 使用相似度最高的样本作为种子
    sim_matrix = torch.mm(features, features.t())
    topk_indices = torch.topk(sim_matrix.flatten(), num_classes).indices
    center_indices = [idx // 180 for idx in topk_indices.cpu().numpy()]
    centers = features[center_indices]

    # 3. 迭代优化
    max_iters = 20
    for _ in range(max_iters):
        # 计算样本与中心的相似度 [180, 20]
        center_sim = torch.mm(features, centers.t())

        # 构建分配代价矩阵（使用相似度负值）
        cost_matrix = -center_sim.cpu().numpy()

        # 使用匈牙利算法解决分配问题
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 创建初始分配
        assignment = torch.full((180,), -1, dtype=torch.long)
        assignment[row_ind] = torch.tensor(col_ind)

        # 确保每个类恰好9个向量
        class_counts = torch.zeros(num_classes, dtype=torch.long)
        final_assignment = torch.full((180,), -1, dtype=torch.long)

        # 按相似度排序分配
        sorted_indices = torch.argsort(center_sim.flatten(), descending=True)
        for idx in sorted_indices:
            sample_idx = idx // num_classes
            class_idx = idx % num_classes

            if final_assignment[sample_idx] == -1 and class_counts[class_idx] < class_size:
                final_assignment[sample_idx] = class_idx
                class_counts[class_idx] += 1

        # 更新中心点
        new_centers = torch.zeros_like(centers)
        for class_idx in range(num_classes):
            class_mask = (final_assignment == class_idx)
            if class_mask.sum() > 0:
                new_centers[class_idx] = features[class_mask].mean(dim=0)

        # 中心点归一化
        new_centers = F.normalize(new_centers, p=2, dim=1)

        # 检查收敛
        if torch.allclose(centers, new_centers, atol=1e-6):
            break

        centers = new_centers

    return final_assignment


def process_batch(batch_features):
    """
    处理单个batch的特征向量
    Args:
        batch_features: 形状为 [180, 32] 的特征张量
    Returns:
        cluster_labels: 形状为 [180] 的聚类标签
        similarity_matrix: 形状为 [20, 20] 的类内相似度矩阵
    """
    # 执行约束聚类
    labels = constrained_clustering(batch_features)

    # 计算类内相似度
    class_sims = []
    for class_id in range(20):
        class_indices = torch.where(labels == class_id)[0]
        class_vectors = batch_features[class_indices]

        # 计算类内平均相似度
        sim_matrix = torch.mm(class_vectors, class_vectors.t())
        mask = torch.eye(class_vectors.size(0), dtype=torch.bool)
        avg_sim = sim_matrix[~mask].mean().item()
        class_sims.append(avg_sim)

    # 转换为相似度矩阵
    similarity_matrix = torch.zeros(20, 20)
    for i in range(20):
        similarity_matrix[i, i] = class_sims[i]

    return labels, similarity_matrix


# 示例使用
if __name__ == "__main__":
    # 模拟输入数据 [batch_size=10, num_vectors=180, feat_dim=32]
    batch_features = torch.randn(10, 180, 32)

    # 存储结果
    all_labels = []
    all_similarities = []

    # 处理每个batch
    for i in range(batch_features.size(0)):
        print(f"Processing batch {i+1}/10")
        labels, sim_matrix = process_batch(batch_features[i])

        all_labels.append(labels)
        all_similarities.append(sim_matrix)

        # 打印聚类质量
        intra_sim = torch.diag(sim_matrix).mean().item()
        print(f"  Batch {i+1}: Average intra-class similarity = {intra_sim:.4f}")

    # 转换为张量
    cluster_labels = torch.stack(all_labels)  # [10, 180]
    similarity_matrices = torch.stack(all_similarities)  # [10, 20, 20]

    print("\nClustering completed!")
    print(f"Cluster labels shape: {cluster_labels.shape}")
    print(f"Similarity matrices shape: {similarity_matrices.shape}")