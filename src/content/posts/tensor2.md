---
title: 练习tensor
published: 2024-05-13
description: 练习tensor
tags: [numpy, tensor]
category: 学习记录
draft: false
---
在 tensor 应用过程中，PyTorch 的 tensor 和 NumPy 的数组都是非常核心的数据结构，所以学习中我也时常会想，这两个结构区别在哪里，我的结论如下：

- `numpy.array` 不需要严格对齐，并不是严格意义上的矩阵。比如 `[[2,3],[1,2,3]]` 这种数据可以装进 `array` 但不可以装进 `tensor`。`tensor` 和矩阵更接近，numpy 也同样内置了矩阵的概念。但矩阵仅限于 2 维，`tensor` 相当于 n 维矩阵。`tensor` 和矩阵都是 `array` 的特例，都是严格对齐的 `array`。也正是因为 `tensor` 是严格对齐才可以装进去 GPU 进行训练。

下面是主要区别以及知识点：

### 相似点
- **数据存储**：无论是 NumPy 数组还是 PyTorch 的 tensor，它们都可以看作是一个多维数组，用来存储数值数据。这些数据通常以连续的内存块形式存储，便于快速的数据访问和操作。
- **基本操作**：两者在很多基本操作上都非常相似，比如切片、索引、形状变换等。

### 不同点
- **GPU 支持**：最大的不同是 PyTorch tensors 支持 GPU 加速，这对于深度学习模型训练来说非常重要，因为 GPU 能够提供非常高效的数值计算能力。而 NumPy 数组主要在 CPU 上运行。
- **自动微分**：PyTorch tensors 支持自动微分（使用 Autograd），这使得在构建和训练神经网络时能自动计算导数。这是深度学习框架的一个核心功能，NumPy 并不支持。

### 存储形式
- NumPy 数组和 PyTorch Tensor 都是以行优先顺序存储在内存中的。

### 模型训练
- 在训练深度学习模型，如卷积神经网络（CNN）时，我们通常用 tensor 来存储和计算数据。NumPy 用得较少，因为它不支持 GPU 加速和自动微分。

### CNN 中的角色
- 在 CNN 中，输入数据、权重和偏置等都以 tensor 的形式存在。例如：
  - **输入层**：输入的图像数据被转换为 tensor。
  - **卷积层**：权重（卷积核）和输入数据进行卷积操作，通常输出还会通过激活函数（如 ReLU）进一步处理。
  - **全连接层**：权重和特征图（flatten后的输出）进行矩阵乘法加偏置操作。

### 权重和 Tensor 的关系
- 在神经网络中，权重本身就是一种特殊的 tensor。在训练过程中，通过反向传播算法，这些权重 tensor 会根据损失函数的梯度进行更新。

### 代码
再来练习一下如何创建 tensor，并使用一个卷积层：

```python
import torch
import torch.nn as nn

# 创建一个随机数据 tensor
input_tensor = torch.randn(1, 1, 28, 28)
conv_layer = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3)
output_tensor = conv_layer(input_tensor)

print("Output shape:", output_tensor.shape)
```
以上代码，就是将一个 1x1x28x28 的 tensor 转换为 1x3x26x26 的 tensor。
