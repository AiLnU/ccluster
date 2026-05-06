import torch
import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据（替换为你的实际数据）
torch.manual_seed(42)  # 确保可重现性

# 20个坐标点 [20, 2]
points = torch.randn(20, 2) * 5

# 点的序号 [20]
indices = torch.arange(20)

# 点的类别 [20] (假设有3个类别)
categories = torch.randint(0, 3, (20,))

# 创建绘图
plt.figure(figsize=(12, 8), dpi=100)

# 定义类别颜色和标签
colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
'beige', 'bisque', 'black', 'blanchedalmond', 'blue']  # 最多支持6个类别
class_labels = ['Class A', 'Class B', 'Class C', 'Class D', 'Class E', 'Class F']

# 为每个类别创建图例句柄
legend_handles = []

# 绘制每个点
for cat_id in torch.unique(categories):
    # 获取当前类别的点
    mask = (categories == cat_id)
    cat_points = points[mask]
    cat_indices = indices[mask]

    # 绘制点
    scatter = plt.scatter(
        cat_points[:, 0],
        cat_points[:, 1],
        s=150,  # 点的大小
        c=colors[cat_id],
        alpha=0.7,
        edgecolors='black',
        label=class_labels[cat_id]
    )
    legend_handles.append(scatter)

    # 添加序号标签
    for i, (x, y) in enumerate(cat_points):
        plt.annotate(
            str(cat_indices[i].item()),
            (x, y),
            xytext=(5, 5),  # 标签偏移量
            textcoords='offset points',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )

# 添加标题和标签
plt.title('Coordinate Points with Indices and Categories', fontsize=14)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)

# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 添加图例
plt.legend(handles=legend_handles, title='Categories', loc='best')

# 自动调整坐标轴范围
margin = 0.5  # 边界留白
x_min, x_max = points[:, 0].min().item() - margin, points[:, 0].max().item() + margin
y_min, y_max = points[:, 1].min().item() - margin, points[:, 1].max().item() + margin
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 添加平均点
mean_point = points.mean(dim=0)
plt.scatter(mean_point[0], mean_point[1], s=300, marker='*', c='gold', label='Center')
plt.annotate('Center', (mean_point[0], mean_point[1]),
             xytext=(10, 15), textcoords='offset points', fontsize=12)

# 显示图形
plt.tight_layout()
plt.show()

a=1