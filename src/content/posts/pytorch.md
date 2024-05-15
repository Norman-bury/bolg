---
title: Pytorch框架下的模型的训练与验证
published: 2024-05-15
description: "学习记录pytorch框架比较复杂的部分，也就是训练和验证，先从对比tensorflow框架开始"
image: "src/content/cover1.jpg"
tags: ["PyTorch", "Tensor"]
category: 学习记录
draft: false
---

# PyTorch vs Keras 框架对比
在学习模型的训练和验证之前，我们先来对比一下PyTorch和Keras框架，这方便我们更好理解PyTorch框架，个人感觉高度的集成化也导致了tensorflow在未来发展上会注定不如pytorch。Keras 设计初衷是使神经网络的设计和实验变得简单，它对初学者非常友好。它的高层次抽象允许用户快速搭建和实验标准网络，但这种设计也可能在需要高度定制模型时显得不够灵活。随着我们对深度学习技术的掌握逐渐深入，我们可能倾向于选择提供更多控制和定制选项的框架，就是PyTorch。下面是对比不同之处
## 1. 模型定义

### PyTorch:
在 PyTorch 中，我们通常会继承 `nn.Module` 类来定义自己的模型。在模型类中，需要先定义 `__init__` 方法来初始化模型的层，以及 `forward` 方法来指定前向传播。

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```
### Keras:
在 Keras 中，可以使用 Sequential 模型或者函数式 API 快速定义模型。这些方法通常比 PyTorch 更简洁。

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Dense(20, activation='relu', input_shape=(10,)),
    layers.Dense(2, activation='softmax')
])
```
## 2. 损失函数和优化器

### PyTorch:
在 PyTorch 中，我们则需要手动定义损失函数和优化器。这些通常在训练循环外部定义。

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
### Keras:
在 Keras 中，在编译模型时直接可以定义损失函数和优化器，整个过程更集成化。

```python
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
## 3. 训练循环

### PyTorch:
在 PyTorch 中，我们则需要手动编写训练循环。这包括前向传播、损失计算、梯度清零、反向传播和参数更新。

```python
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
### Keras:
在 Keras 中，训练过程可以通过一个简单的 fit 方法调用实现，大大简化了代码。

```python
model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
```
## 4. 验证过程

### PyTorch:
在 PyTorch 中，验证过程通常在训练循环中进行，需要手动设置模型为验证模式，并关闭梯度计算。

```python
model.eval()
with torch.no_grad():
    for inputs, labels in validation_loader:
        outputs = model(inputs)
        val_loss = criterion(outputs, labels)
```
### Keras:
在 Keras 中，我们可以在 fit 方法中直接设置验证数据集，模型会在每个 epoch 后评估验证数据集的性能。

```python
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=batch_size)
```
下面是PyTorch框架的代码的关键部分，也是我近来学习编写感觉不流畅经常卡顿的地方
# PyTorch 框架下模型训练的关键步骤

## 1. 数据加载
在 PyTorch 中，数据通常通过 `torch.utils.data.DataLoader` 加载，它提供了一个高效的方式来迭代数据，支持自动批处理、采样、打乱数据和多线程数据加载。

```python
from torch.utils.data import DataLoader, TensorDataset
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
```
## 2. 设置模型、损失函数和优化器
我们如上面所示，需要定义模型架构，选择合适的损失函数和优化器。

## 3. 训练循环
在 PyTorch 中，数据通常通过 `torch.utils.data.DataLoader` 加载，它提供了一个高效的方式来迭代数据，支持自动批处理、采样、打乱数据和多线程数据加载。
-前向传播：模型对输入数据进行计算，输出预测结果。
-计算损失：使用损失函数计算模型输出与真实标签之间的误差。
-梯度清零：在每次迭代开始之前，需要清除旧的梯度，因为梯度是累积的。
-反向传播：根据损失函数对模型参数进行梯度计算。
-参数更新：优化器根据计算得到的梯度更新模型的权重。
代码如下：
```python
# 设置模型为训练模式
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # 梯度清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
```
## 4. 验证过程
训练过程中或训练后，通常需要在一个或多个验证数据集上评估模型性能。这需要切换到模型的评估模式，并关闭梯度计算以提高性能。

```python
model.eval()
total_loss = 0
with torch.no_grad():  # 关闭梯度计算
    for inputs, labels in validation_loader:
        outputs = model(inputs)
        val_loss = criterion(outputs, labels)
        total_loss += val_loss.item()
```
# 总结
PyTorch 提供了更高的灵活性和控制度，允许手动处理许多细节。而 TensorFlow 的 Keras API 则提供了更高层次的抽象，使得常见的训练过程更简单、更快捷。两个框架各有千秋，选择哪个框架取决于个人需求、项目要求以及对特定框架的熟悉程度。Keras 因其简单性和易用性在初学者和某些工业应用中依然非常受欢迎，而 PyTorch 在研究和需要高度定制的应用中表现更为出色。
