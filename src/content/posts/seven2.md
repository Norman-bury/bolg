---
title: 从梯度裁剪到模型剪枝：SEVEN方法的复现
published: 2024-05-07
description: 学习记录介绍这篇论文《SEVEN：通过保留哨兵修剪Transformer模型》部分2代码复现
tags: [深度学习, 模型剪枝方法]
category: 学习记录
draft: false
---
上一篇讲述了论文的基本知识，这一篇我们来介绍要实现这个SEVEN方法需要进行哪些操作。在复现之前，我们需要了解清楚论文涉及的关键名词和概念和基本的数学公式推导都有哪些，以方便我们更好的应用。

### 关键名词和概念：
- **梯度裁剪 (Gradient Clipping)**：它是一种防止梯度爆炸的优化技术，它通过限制梯度的大小来保证训练过程的稳定性。梯度裁剪不会改变模型的大小或结构，只是调整梯度值以稳定训练过程。
- **符号下降 (Symbolic Descent, SD)**：符号下降（SD）是一种针对梯度优化的变体，特别关注于减少梯度更新中的噪声和不确定性。SD方法通常涉及到对梯度更新步骤的一些形式的调整或变换，以便更好地控制优化过程中的变量。它包括使用更复杂的数学表达式来表示梯度计算，或者在更新步骤中引入额外的控制参数。
- **梯度噪声处理 (Gradient Noise Handling)**：SEVEN 方法中的一个关键概念是处理梯度噪声，这对于判断权重的真正重要性至关重要。噪声大的梯度可能导致权重的重要性被高估。

说到这里，我们又不得不介绍一下SEVEN 方法中，它的两种剪枝方法和普通剪枝又有哪些区别呢。接下来我将详细解释 SEVEN 方法中的两种剪枝方式（SEVENpre 和 SEVENdyn），并将它们与传统的剪枝方法进行对比。

1. **普通修剪方法**
   普通修剪方法通常在训练后进行，基于某种标准（如权重的大小或梯度的大小）移除网络中的参数。这种方法的目的是减小模型的大小，提高推断效率，同时尽量保持模型性能。普通修剪通常是静态的，意味着一旦完成修剪，被移除的权重在后续的使用中不再参与计算。

2. **SEVENpre（预修剪方法）**
   SEVENpre 是在模型训练开始之前进行的一种修剪技术。它利用了模型在未训练前的初始化状态，通过分析权重的梯度噪声来预测每个权重在训练中可能的重要性。这种方法尝试识别并保留那些即使在模型训练早期就显示出高重要性的权重（稳定性和灵敏度低的梯度噪声）。这样做的好处是可以在训练开始前就减少模型的复杂度，可能帮助加速初期训练过程。

3. **SEVENdyn（动态修剪方法）**
   与 SEVENpre 不同，SEVENdyn 在整个训练过程中动态地执行修剪操作。它不仅考虑了模型权重的初始重要性，还观察了权重随训练进度如何变化。在多个训练周期中，根据权重的变化和梯度噪声动态调整修剪决策。这种方法的目的是在模型训练过程中不断优化网络结构，以响应训练数据的变化和模型需求的演进。

### 它俩与普通修剪的区别
- **时机和动态性**：普通修剪通常在训练完成后基于最终模型状态进行，而 SEVENpre 和 SEVENdyn 分别在训练前和训练中进行，更加动态和适应性强。
- **基于噪声的决策**：SEVEN 方法特别关注梯度噪声，这是区别于常规方法（通常只考虑梯度大小或权重大小）的一个重要特点。通过这种方式，SEVEN 能够更好地评估哪些权重是真正重要的，哪些是冗余的。
- **目标和效果**：SEVEN 旨在通过精细的剪枝策略保持或甚至提升模型性能，而传统剪枝更多关注减少计算量和模型大小，有时可能牺牲一些性能。

### 数学公式和理论：
#### 1. 计算重要性分数 (compute_importance_scores)

这部分代码负责计算每个权重的重要性分数，主要依据是权重的梯度大小。这基于一个基本的数学原理：如果一个权重的梯度很大，说明这个权重对损失函数的影响很大，因此这个权重被认为是“重要”的。

**数学表达：**
$$
\text{Importance}(W) = \left| \frac{\partial L}{\partial W} \right|
$$


其中 \( L \) 是损失函数，\( W \) 是权重，
$$
\(\frac{\partial L}{\partial W}\) 
$$
是权重 \( W \) 的梯度。在代码中，通过调用 `loss.backward()` 来自动计算这些梯度。

#### 2. 动态修剪权重 (prune_model)

根据计算得到的重要性分数，代码将进行动态修剪。通过设置一个阈值，只有那些其重要性分数高于这个阈值的权重会被保留。

**数学表达：**
$$
\[ W = \begin{cases} 
W & \text{if } \text{Importance}(W) > \tau \\
0 & \text{otherwise}
\end{cases} \]
$$

在代码中，这个阈值是通过 `torch.quantile(importance_scores[name], sparsity_level)` 计算得到的，它基于指定的稀疏度级别动态确定。

#### 3. 应用掩码

这个过程涉及到将一个掩码应用于模型的权重。这个掩码基于上述的阈值判断，决定哪些权重应该被保留。

**数学表达：**
$$
\[ W_{\text{new}} = W \cdot \text{mask} \]
$$
其中$$ 
\(\text{mask}\)
$$ 
是一个二元数组，其元素根据权重的重要性分数是否大于阈值 
$$
\(\tau\) 
$$
而确定为1或0。

### 梯度的计算

梯度的计算公式是：
$$
\[ \nabla L(\theta) \]
$$
其中 \( L \) 是损失函数，
$$
\( \theta \) 
$$
表示模型参数。这个公式用于计算参数对损失的敏感度，是参数更新的基础。

### 权重更新规则

权重更新规则为：
$$
\[ \theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla L(\theta) \]
$$
其中
$$
\( \eta \) 
$$
是学习率。这个规则说明了如何利用梯度来迭代更新权重，以改进模型性能。
了解到知识点和基础概念以上之后，我们先来通过PyTorch复现这预修剪这一操作。在实际操作中，预修剪通常是通过设置一个阈值，然后将低于这个阈值的权重直接设置为零来实现。以下是一个简单的Python代码示例，展示如何使用PyTorch进行预修剪操作。这个例子中，我们将根据权重的绝对值进行修剪，只保留那些绝对值大于某个阈值的权重。

这里我将展示如何为一个简单的卷积层进行预修剪：

```python
import torch
import torch.nn as nn

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

# 假设这是预训练后的权重，随机初始化
conv_layer.weight.data = torch.randn_like(conv_layer.weight.data)

# 打印修剪前权重
print("Weight before pruning:")
print(conv_layer.weight.data)

# 设置修剪阈值50%
threshold = torch.quantile(torch.abs(conv_layer.weight.data), 0.5)

# 创建一个权重的掩码
mask = torch.abs(conv_layer.weight.data) > threshold

# 应用掩码，小于阈值的权重被设置为0
conv_layer.weight.data *= mask.float()

# 打印修剪后的权重
print("Weight after pruning:")
print(conv_layer.weight.data)
```
### 代码解释：
- **创建卷积层**：定义一个包含随机权重的卷积层。
- **设置阈值**：使用权重的绝对值的中位数作为阈值。
- **创建掩码**：创建一个布尔掩码，其中权重的绝对值大于阈值的位置为True，其他为False。
- **应用掩码**：通过与掩码相乘，将低于阈值的权重设置为零。

这里我们看到，预修剪主要是应用一个mask面罩数据来实现，它对两种数据进行融合，这个操作思想有点类似于多模态模型的输入，但又不完全相似，多模态是在模块训练之前将输入数据就进行混合统一尺寸和通道。

### 打印效果
#### 权重修剪前的
修剪前的权重如下所示：
```python
tensor([[[[ 0.3449, -0.1833,  1.1621],
          [-0.3372, -1.8285, -0.8584],
          [-1.1111,  0.3171, -1.2085]]]])
```
这是一个1x1x3x3的张量，表示有一个卷积核，核大小为3x3。

#### 权重修剪后的
修剪后的权重如下所示：
```python
tensor([[[[ 0.0000, -0.0000,  1.1621],
          [-0.0000, -1.8285, -0.0000],
          [-1.1111,  0.0000, -1.2085]]]])
```
在这里，阈值是根据权重的绝对值计算的，例如这里选择了0.5作为阈值。在这个例子中，所有小于阈值的权重都被设置为0（修剪掉了）。这意味着这些权重在卷积运算中不会再对输出有任何贡献。
然后接下来我们终于到了重头戏，也就是两个方法的实现代码，这块代码我是通过查阅公布的GitHub代码进行复制，然后做了一些注释方便更好理解。

### SEVENpre：预修剪
SEVENpre 方法是在模型开始训练前进行修剪的。它是对模型的权重进行一次分析，确定它们在未经训练的状态下的重要性，然后根据这些权重的预测重要性进行修剪。代码如下：
```python
import torch
from torchvision import models
import torch.nn as nn

def prune_pretrained(model, threshold=0.01):
    """ 对预训练模型进行预修剪 """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # 获取权重的绝对值
                weights_abs = torch.abs(module.weight.data)
                # 计算阈值
                th = torch.quantile(weights_abs, threshold)
                # 创建掩码并应用
                mask = weights_abs > th
                module.weight.data *= mask.float()
                
# 加载预训练模型
model = models.resnet50(pretrained=True)

# 对模型进行预修剪
prune_pretrained(model)

# 修剪后，模型可以进行正常的训练过程
```
### SEVENdyn：动态修剪
与 SEVENpre 不同，SEVENdyn 在整个训练过程中动态地修剪模型。这需要在训练的每个步骤或特定周期内评估权重的重要性，并根据当前的训练状态决定是否修剪某些权重。这种类型的剪枝需要在训练循环内部实现，代码如下所示：
```python
def train_model_dynamic_pruning(model, data_loader, criterion, optimizer, epochs=10, prune_every_n_steps=100):
    step = 0
    for epoch in range(epochs):
        for inputs, labels in data_loader:
            model.train()
            inputs, labels = inputs.to(device), labels.to(device)

            # 正向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 动态修剪
            if step % prune_every_n_steps == 0:
                dynamic_prune(model, threshold=0.01 * epoch / epochs)  # 假设阈值随时间变化

            step += 1

def dynamic_prune(model, threshold):
    """ 在训练中动态修剪权重 """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weights_abs = torch.abs(module.weight.data)
                th = torch.quantile(weights_abs, threshold)
                mask = weights_abs > th
                module.weight.data *= mask.float()
```
至此我们的讲解结束，我们来总结一下SEVEN方法：
SEVEN 方法实际上不仅仅是梯度裁剪，而是一个更复杂的模型剪枝策略。它从梯度处理技术，也就是符号下降（Symbolic Descent, SD）中汲取了灵感，它的主要目的是通过剪枝（即去除模型中认为不重要的权重）来优化了模型结构和性能。

现在LLM当行其道，大模型的优秀性能以及GPT、Claude的强势出圈也基本宣告了自然语言处理的最终目标就是大模型，但是现在小研究组小实验室算力不够限制了对大模型的开发，适配下游任务等等，导致大模型的前线进展只在谷歌、微软这类大公司的手中。这就导致我们认为，人工智能离我们这些研究者越来越远了，那么我们能做的工作还有哪些，我们已经无能为力了吗？当然不是，就比如transformer框架绝对不是一个终极框架，它的发展还在继续，它的问题还在产生，就比如如何使大模型更加高效？大模型如何适配到下游任务？如何实现大模型的可控生成？如何提高大模型的性能，减重大模型？如何降低大模型的使用门槛？等等。所以从这些角度来看，其实它拉进了我们对人工智能的一个研究。模型剪枝、知识蒸馏就是代表，研究还在发展，我们仍需努力。

