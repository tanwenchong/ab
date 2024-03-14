from torch.utils.data import WeightedRandomSampler
import torch
def get_sampler(dataset):
    num_samples = len(dataset)
    num_classes = len(torch.unique(dataset.y))
    class_counts = torch.zeros(num_classes)
    for data in dataset:
        class_counts += torch.bincount(data.y, minlength=num_classes)

# 3. 计算每个类别的权重
    class_weights = 1.0 / class_counts

# 4. 为每个样本分配权重
    weights = class_weights[dataset.y]

# 5. 使用 WeightedRandomSampler 创建 DataLoader
    sampler = WeightedRandomSampler(weights, num_samples=len(weights))
    return sampler