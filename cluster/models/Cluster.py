# %BANNER_BEGIN%
# ---------------------------------------------------------------------
#  cluster based on deepseek
#  Originating Authors: Shaopeng Li
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans


class DeepClusterModel(nn.Module):
    def __init__(self, input_dim, num_classes, embed_dim=64):
        super().__init__()
        # 特征嵌入网络
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)

        # 可学习聚类中心
        self.cluster_centers = nn.Parameter(
            torch.randn(num_classes, embed_dim))

        # 辅助分类器
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embeddings = self.embedder(x)
        logits = self.classifier(embeddings)
        return embeddings, logits


class ClusterLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, embeddings, logits, labels, centers):
        # 分类损失
        cls_loss = self.ce_loss(logits, labels)

        # 聚类中心约束
        batch_centers = centers[labels]
        cluster_loss = torch.mean(
            torch.norm(embeddings - batch_centers, dim=1))

        # 正则化项
        reg_term = torch.mean(torch.norm(centers, dim=1))

        return cls_loss + self.alpha * cluster_loss + 0.1 * reg_term


# # 训练参数
# input_dim = 256  # 特征维度
# num_classes = 100  # 目标类别数
# batch_size = 64
# lr = 1e-4
# epochs = 100
#
# # 初始化组件
# model = DeepClusterModel(input_dim, num_classes)
# optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = ClusterLoss()
#
# # 数据准备（示例）
# train_features = [...]  # 输入特征列表
# train_labels = [...]  # 对应标签列表
# train_dataset = TensorDataset(
#     torch.tensor(train_features),
#     torch.tensor(train_labels))
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练过程
# model.train()
# for epoch in range(epochs):
#     total_loss = 0.0
#     for features, labels in train_loader:
#         optimizer.zero_grad()
#
#         # 前向传播
#         embeddings, logits = model(features.float())
#
#         # 损失计算
#         loss = criterion(embeddings, logits, labels, model.cluster_centers)
#
#         # 反向传播
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")


# 聚类推理
def cluster_association(features, n_clusters):
    # model.eval()
    with torch.no_grad():
        embeddings = model.embedder(torch.tensor(features).float())

    # 使用K-means进行最终聚类
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_ids = kmeans.fit_predict(embeddings.numpy())
    return cluster_ids


# 使用示例
new_features = [...]  # 新观测的特征向量
pred_clusters = cluster_association(new_features, num_classes)