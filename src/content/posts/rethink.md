---
title: 简化版Transformer模型的探索
published: 2024-05-23
description: "学习记录下Rethinking Attention这篇论文"
tags: ["Transformer", "注意力机制"]
category: 学习记录
draft: false
---

# 论文题目
Rethinking Attention: Exploring Shallow Feed-Forward Neural Networks as an Alternative to Attention Layers in Transformers

本文介绍一篇来自苏黎世联邦理工学院（ETH Zurich）的最新Transformer优化工作，目前该文已被人工智能顶级会议AAAI 2024录用。本文的核心出发点是，能否使用更加轻量经济的前馈神经网络（MLP）来替代Transformer中笨重的自注意力层，并通过知识蒸馏的方式使用原始模块进行迁移训练，作者将优化后的模型称为”attentionless Transformers“。作者在IWSLT2017等数据集上的实验验证了attentionless Transformer可以达到与原始架构相当的性能，同时进行了一系列消融实验表明，如果正确的配置参数，浅层MLP完全具有模拟注意力机制的潜力。

在今天这个AI技术快速发展的时代，大型语言模型（LLMs）和AIGC正引领着科技的前沿。在这股潮流中，Transformer模型因其强大的处理能力，成为了研究的焦点。然而，Transformer的核心—自注意力机制，虽然功能强大，却也资源消耗巨大。但是本篇论文，这种新型的“无注意力Transformer”在实验中展现了与传统模型相媲美的效果。今天，我来看一看究竟，看看这项新技术是如何工作的，是真是假。

## 一种新的尝试：用MLP模拟注意力机制

传统的Transformer模型包括多个编码器和解码器，每一层都有自己的注意力机制，帮助模型“关注”输入数据中的关键信息。而新的研究提出，可以用一个简单的MLP层来替代这些复杂的自注意力层。研究者们尝试了四种不同的替代方式：

- 注意力层替换（ALR）：在这种模式下，多头注意力（MHA）被一个MLP层替代，但保留了残差连接和归一化层。
- 残差连接替换的注意力层（ALRR）：这种模式不仅替换了MHA，还替换了残差连接，尝试彻底简化结构。
- 注意力头分离替换（ASLR）：每个注意力头被一个单独的MLP层替代，模拟原有的多头注意力效果。
- 编码器层替换（ELR）：最极端的替代，直接用MLP层替换整个编码器层。

这些新的替代方法减少了模型的复杂性和资源消耗，同时保持了相对较高的效能。

## 实验验证：新思路的效果如何？

研究者们在IWSLT2017数据集上对这些新模型进行了测试。这个数据集包括多种语言的翻译任务，是检验模型性能的理想场所。结果显示，这些新的替换模式都能达到与原始Transformer相当的翻译质量，其中一些甚至表现更佳。

## 关键代码

假设我们要用MLP替换原始Transformer中的多头注意力层（ALR模式）。以下是MLP的关键代码示例：

```python
import torch
import torch.nn as nn

class MLPAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        return self.linear(x)

class TransformerWithMLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        this.mlp_attention = MLPAttention(embed_dim)
        this.norm = nn.LayerNorm(embed_dim)
        this.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.mlp_attention(x) + x
        x = this.norm(x)
        return this.dropout(x)
```
以上定义了一个MLPAttention类来替换原有的注意力机制，并在Transformer结构中使用它。

## 结论：基本向着更简单的AI模型迈进

通过这项研究，确实看到了AI模型简化的可能性。使用MLP来替代传统的注意力机制不仅能节约计算资源，还能保持良好的模型性能。但是已经有许多知识蒸馏的模型可供使用，不知道未来谁的发展会更胜一筹。

