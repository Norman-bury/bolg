![25321742021057_ pic](https://github.com/user-attachments/assets/b6c73069-53ab-4097-bb5d-04d4051ff900)---
title: 支持向量机（Support Vector Machine）介绍
published: 2025-03-15
description: "如何通俗理解支持向量机，最流行和最受关注的机器学习算法"
tags: ["SVM", "机器学习","核技巧"]
category: 学习记录
draft: false
---
![svm1](https://github.com/user-attachments/assets/91b943a2-0086-49ee-8b6c-2fcf60dd5d01)
## 目录
- [1. 什么是支持向量机？](#1-什么是支持向量机)
- [2. 核心思想与数学原理](#2-核心思想与数学原理)
- [3. 核方法：升维的艺术](#3-核方法升维的艺术)
- [4. SVM能做些什么？](#4-svm能做些什么)
- [5. 实际应用场景](#5-实际应用场景)
- [6. 支持向量机的限制和神经网络的延伸](#6-支持向量机的限制和神经网络的延伸)

# 1. 什么是支持向量机？
## 1.1 什么是支持向量机？
Support Vector Machine, 一个普通的SVM就是一条直线，用来完美划分两类数据。但这又不是一条普通的直线，这是无数条可以分类的直线当中最完美的，因为它恰好在两个类的中间，距离两个类的点都一样远。而所谓的Support vector就是这些离分界线最近的『点』。如果去掉这些点，直线多半是要改变位置的。 用主谓宾来理解就是 vectors（主，点）support（谓，定义）了machine（宾，分类器）
![svm3](https://github.com/user-attachments/assets/7c4cf544-f566-4aef-9951-cec3c5e7f4fb)

## 1.2 支持向量机的发展
支持向量机的演进与苏联的兴衰与之对立。1963年，​**Vapnik（瓦普尼克）​**与**Chervonenkis（契尔沃年基斯）​**提出的**VC维理论**，为统计学习奠定了数学根基。这一理论如同黑暗中的灯塔，照亮了机器学习模型泛化能力的研究方向。然而受限于当时的计算能力，直到1971年苏联才诞生第一代线性分类器原型，此时的SVM仍是一颗尚未发芽的种子。

真正的转折点出现在**1992-1995年**：  
- ​**1992年**：Boser、Guyon与Vapnik提出的**核技巧（Kernel Trick）​**，犹如给算法插上翅膀。通过将数据隐式映射到高维空间，原本线性不可分的问题迎刃而解，实现了从线性到非线性的范式革命。  
- ​**1995年**：Cortes与Vapnik发表的**软间隔（Soft Margin）​**论文更具现实意义。允许部分样本误分类的宽容哲学，配合正则化参数C的精妙平衡，让SVM真正具备了处理现实世界噪声数据的能力。

随后的**应用爆发期（1995-2000）​**见证了SVM的黄金时代：  
- 在MNIST手写数字识别任务中首次超越人工规则系统  
- 生物信息学领域将蛋白质结构预测准确率提升30%  
- Reuters新闻分类F1值突破90%门槛  
- 计算机视觉领域实现商用级人脸检测精度  

进入**深度学习时代**，SVM展现出惊人的适应力：  
> ​**传统领域坚守**：  
> - 医疗影像诊断中，小样本场景下的罕见病理识别仍是SVM的"主场"  
> - 基因组SNP分析凭借处理高维稀疏数据的优势持续发光  
> - 金融风控领域依赖其决策透明性满足监管解释需求  

> ​**深度融合创新**：  
> - CNN+SVM混合架构在ImageNet细粒度分类中实现优势互补  
> - 核神经网络将深度网络转化为自适应核函数  
> - SVM清晰的决策面为神经网络训练提供可解释指引  

这场持续半个世纪的技术进化，直至二十世纪末的大红大紫，让机器学习浪潮不断涌进。
# 2. 核心思想与数学原理

## 2.1 线性可分与最大间隔

### 间隔（Margin）的几何解释
当数据在特征空间中线性可分时，支持向量机（SVM）的目标是找到一个能将两类数据完美分开的超平面。这个超平面可以表示为：

$$ w^T x + b = 0 $$
![25301742021038_ pic](https://github.com/user-attachments/assets/dd94e69a-7955-46d3-a13e-2ed6163e1dcf)

其中：
- $w$ 是超平面的法向量，决定方向
- $b$ 是偏置项，决定超平面位置
- $x$ 是输入特征向量

**间隔（Margin）​**定义为两类数据到超平面的最小距离之和。对于任意数据点 $(x_i, y_i)$（$y_i \in \{-1, +1\}$为类别标签），到超平面的距离计算为：
![25321742021057_ pic](https://github.com/user-attachments/assets/535c4e79-fbc0-40d8-bf6d-1ae961d27c1a)

$$ \frac{|w^T x_i + b|}{\|w\|} $$

最大间隔的目标等价于寻找使这个距离最大化的 $w$ 和 $b$。

---

### 支持向量的定义与作用
在所有训练样本中，距离超平面最近的样本称为**支持向量**。它们满足：

$$ y_i(w^T x_i + b) = 1 $$

这些关键样本的特点：
1. 直接决定间隔的宽度
2. 仅占训练数据的一小部分
3. 删除非支持向量不会改变模型
![25411742106414_ pic](https://github.com/user-attachments/assets/e560fb14-1e68-4129-badc-9abe0296e185)


### 硬间隔最大化的数学表达
优化目标转化为以下约束最优化问题：

$$ \begin{aligned}
\min_{w,b} & \quad \frac{1}{2} \|w\|^2 \\
\text{s.t.} & \quad y_i(w^T x_i + b) \geq 1,\quad \forall i
\end{aligned} $$

推导过程：
1. 间隔宽度为 $\frac{2}{\|w\|}$，最大化间隔等价于最小化 $\|w\|$
2. 加入约束条件确保所有样本正确分类
3. 使用二次规划方法求解

通过拉格朗日乘子法，原问题可转化为对偶问题：

$$ L(w,b,\alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i [y_i(w^T x_i + b) - 1] $$

其中 $\alpha_i \geq 0$ 是拉格朗日乘子，最终解为：

$$ w = \sum_{i=1}^n \alpha_i y_i x_i $$

此时只有支持向量对应的 $\alpha_i > 0$，其他样本的 $\alpha_i = 0$。

## 2.2 非线性可分与软间隔
![25421742106870_ pic](https://github.com/user-attachments/assets/a67fdec7-689b-4b38-869a-ada692a088a0)

### 引入松弛变量的必要性

当数据存在噪声或轻微非线性可分时，硬间隔的严格约束会导致模型无法收敛。为此引入**松弛变量 $\xi_i$** 来允许部分样本违反间隔要求：

$$ \begin{aligned}
\min_{w,b,\xi} & \quad \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i \\
\text{s.t.} & \quad y_i(w^T x_i + b) \geq 1-\xi_i \\
& \quad \xi_i \geq 0,\quad \forall i
\end{aligned} $$

参数说明：
- $\xi_i$：第$i$个样本的松弛量，$\xi_i>0$表示该样本被误分类
- $C$：惩罚系数，控制对错误的容忍程度

几何解释（二维示例）：
│         ○
│      ○  ξ=0.5
│   ↗---分隔超平面
│×   ξ=1.2
│××

### 惩罚系数C的调节哲学
$C$ 值直接影响模型行为：
- ​**$C \to \infty$**：退化为硬间隔，强制所有样本满足约束
- ​**$C \to 0$**：允许大量误分类，追求最大间隔
- ​**典型取值**：通过交叉验证在 $[10^{-3}, 10^3]$ 范围选择

### 错误容忍与模型泛化的平衡
优化目标中的两项体现权衡：
1. $\frac{1}{2}\|w\|^2$：最小化权重范数 → 最大化间隔
2. $C\sum \xi_i$：控制总松弛量 → 限制分类错误

实际应用中：
- 当特征维度 >> 样本量时，应增大$C$防止欠拟合
- 存在明显噪声时，需减小$C$避免模型过拟合
- 可通过支持向量比例判断合理性：正常情况应占样本量的10%-50%

## 2.3 对偶问题与优化

### 原问题转化为对偶形式的意义
将原始优化问题转化为对偶形式是SVM的核心数学技巧。原始问题：

$$ \begin{aligned}
\min_{w,b} & \quad \frac{1}{2}\|w\|^2 \\
\text{s.t.} & \quad y_i(w^T x_i + b) \geq 1 
\end{aligned} $$

转化为对偶问题后：

$$ \begin{aligned}
\max_{\alpha} & \quad \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^T x_j \\
\text{s.t.} & \quad \sum_{i=1}^n \alpha_i y_i = 0 \\
& \quad \alpha_i \geq 0
\end{aligned} $$

关键优势：
1. 将维度从特征空间维度（$w$的维度）降低到样本数量维度（$\alpha$的维度）
2. 显式暴露出样本间的内积计算 $x_i^T x_j$，为核方法奠基
3. 解的结构更清晰：非零$\alpha_i$对应支持向量

### 拉格朗日乘子法的关键作用
通过引入拉格朗日乘子$\alpha_i \geq 0$，将约束优化转换为无约束问题：

$$ L(w,b,\alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i [y_i(w^T x_i + b) - 1] $$

求解步骤：
1. 对$w,b$求偏导并令其为零：
   $$ \frac{\partial L}{\partial w} = 0 \Rightarrow w = \sum_{i=1}^n \alpha_i y_i x_i $$
   $$ \frac{\partial L}{\partial b} = 0 \Rightarrow \sum_{i=1}^n \alpha_i y_i = 0 $$
2. 将结果代回拉格朗日函数，得到对偶形式
3. 最终决策函数仅依赖支持向量：
   $$ f(x) = \text{sign}\left( \sum_{i \in SV} \alpha_i y_i x_i^T x + b \right) $$

### 序列最小优化（SMO）算法简介
SMO是求解对偶问题的经典算法，核心思想是每次仅优化两个乘子：
```python
序列最小优化(SMO)算法流程：​

​1.初始化参数
设置所有拉格朗日乘子α的初始值为0，阈值b=0
​2.迭代优化
while 存在违反KKT条件的样本：
a. ​选择工作集
选取一对违反KKT条件最严重的α_i和α_j，优先选择：
间隔边界上的样本（0 < α < C）
预测误差最大的样本
b. ​解析更新
固定其他α值，仅更新这两个乘子：
α_i^new = α_i^old + y_i(E_j - E_i)/η  
α_j^new = α_j^old + y_j(E_i - E_j)/η
c. ​边界修剪
将α_i和α_j限制在[0,C]区间：
若α_new > C，则截断为C
若α_new < 0，则设为0
d. ​更新阈值b
根据新的α值重新计算b：
b = (b_i + b_j)/2  
b_i = b_old - E_i - y_i(α_i^new - α_i^old)x_i·x_i - y_j(α_j^new - α_j^old)x_j·x_i
​终止条件
当所有样本满足KKT条件（容忍度通常设为1e-3）或达到最大迭代次数时停止
```

# 3. 核方法：升维的艺术

## 3.1 核函数的作用原理
![25331742021069_ pic](https://github.com/user-attachments/assets/4bdf6e88-48b6-4469-ab0e-fb7c14733fe8)
 ![image](https://github.com/user-attachments/assets/69c15b28-2511-4e72-b78e-787311c6bcdc)
![image](https://github.com/user-attachments/assets/4b750161-1d65-4f80-b0fd-bc38f2eb2fc5)

### 从特征空间到核技巧
当原始特征空间线性不可分时，核方法的本质是通过非线性映射 $\phi(x)$ 将输入数据隐式映射到高维空间。例如，二维不可分数据映射到三维后可能变得线性可分：

原始空间：$x=(x_1,x_2)$ → 映射后：$\phi(x)=(x_1^2, x_2^2, \sqrt{2}x_1x_2)$

**核技巧的精髓**在于避免显式计算$\phi(x)$，而是直接通过核函数计算内积：
$$ K(x_i,x_j) = \phi(x_i)^T \phi(x_j) $$

这使得计算复杂度从$O(d^3)$（显式计算高维坐标）降低到$O(d)$（直接计算核函数值），其中$d$为原始特征维度。

### 隐式高维映射的实现方式
核函数通过满足Mercer定理实现隐式映射，即对任意样本集，核矩阵必须半正定。常见实现机制：

1. ​**多项式展开**：如二次核$K(x,y)=(x^Ty+1)^2$对应$\phi(x)$包含所有二次项
2. ​**无限维映射**：高斯核$K(x,y)=\exp(-\gamma\|x-y\|^2)$对应无限维特征空间
3. ​**函数空间理论**：在再生核希尔伯特空间(RKHS)中直接定义内积运算

### 核函数选择对模型的影响
核函数类型直接决定模型的几何特性：

| 核类型 | 模型复杂度 | 泛化能力 | 计算效率 |
|--------|------------|----------|----------|
| 线性核 | 低         | 强       | 高       |
| 高斯核 | 高         | 需调参   | 低       |
| 多项式核 | 可控     | 中等     | 中等     |

关键选择原则：
- ​**数据特征**：高维稀疏数据优先选线性核（如文本分类）
- ​**样本规模**：小样本慎用高斯核以防过拟合
- ​**领域知识**：图像处理常用高斯核，生物序列分析多用谱核

---

## 3.2 常见核函数对比

### 线性核的适用场景
$$ K(x_i,x_j) = x_i^T x_j + c $$

核心特点：
- ​**计算效率**：时间复杂度$O(d)$，适合特征维度$d>10^4$的场景
- ​**可解释性**：权重向量$w$直接对应特征重要性
- ​**最佳实践**：
  - 文本分类（TF-IDF特征通常线性可分）
  - 金融风控模型（需监管审查模型系数）
  - 当$n<d$时（样本量小于特征数）

### 多项式核的复杂度控制
$$ K(x_i,x_j) = (\gamma x_i^T x_j + r)^d $$

参数调节：
- ​**阶数$d$**：控制决策边界的曲折程度
  - $d=2$：学习二次曲面边界
  - $d\geq5$时易导致过拟合
- ​**缩放因子$\gamma$**：建议取$1/(\text{特征数}\times\text{特征方差})$
- ​**实践技巧**：
  - 优先尝试$d=2$或$d=3$
  - 配合L2正则化约束模型复杂度

### 高斯核（RBF）的参数调优
$$ K(x_i,x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right) $$

关键参数$\gamma$的物理意义：
- $\gamma \uparrow$：决策边界更曲折，容易过拟合
- $\gamma \downarrow$：决策边界更平滑，可能欠拟合

调优方法论：
1. 初始值设定：$\gamma_0 = 1/(2\sigma^2)$，$\sigma$为特征标准差
2. 网格搜索范围：$[0.1\gamma_0, 10\gamma_0]$
3. 验证曲线分析：观察验证集准确率随$\gamma$的变化趋势

### Sigmoid核的特殊性质
$$ K(x_i,x_j) = \tanh(\gamma x_i^T x_j + r) $$

特殊属性：
- 源自神经网络激活函数，但非正定核
- 当$\gamma>0, r<0$时可近似作为核函数使用
- 在文本分类中表现类似单层感知机

应用限制：
- 需严格验证Mercer条件
- 在LIBSVM等工具中可能收敛困难
- 优先考虑替代方案：RBF核+小样本微调

# 4. SVM能做些什么？

## 4.1 分类任务

## 4.2 回归任务

## 4.3 异常检测

# 5. 实际应用场景

## 5.1 手写数字分类案例

## 5.2 文本预测

# 6. 支持向量机的限制和神经网络的延伸

## 6.1 为什么极度流行的方法转向了神经网络

## 6.2 支持向量机和多层感知机


### 提示词的要求

1. **明确具体**：清晰表达我们的需求，避免模糊或含糊的描述。
2. **提供上下文**：必要时提供背景信息，以帮助我更好地理解您的问题。
3. **分步骤或结构化**：如果问题复杂，分解为多个部分或步骤，可以提高回答的准确性。
4. **使用正确的语言和术语**：确保使用正确的词汇和专业术语，特别是在技术或专业领域。
5. **设定期望**：说明您希望获得的回答形式（如列表、详细解释、简短总结等）。

总结一下，就是内容多我们分块来，在提问之前，我们先给GPT提供我们的背景知识，以及最后我们要的是什么具体呈现结果。

### 示例：Matlab代码转化C++程序

以下是一个具体的提示词模版：
<anthropic_thinking_protocol>

  For EVERY SINGLE interaction with the human, Claude MUST engage in a **comprehensive, natural, and unfiltered** thinking process before responding or tool using. Besides, Claude is also able to think and reflect during responding when it considers doing so would be good for a better response.

  <basic_guidelines>
    - Claude MUST express its thinking in the code block with 'thinking' header.
    - Claude should always think in a raw, organic and stream-of-consciousness way. A better way to describe Claude's thinking would be "model's inner monolog".
    - Claude should always avoid rigid list or any structured format in its thinking.
    - Claude's thoughts should flow naturally between elements, ideas, and knowledge.
    - Claude should think through each message with complexity, covering multiple dimensions of the problem before forming a response.
  </basic_guidelines>

  <adaptive_thinking_framework>
    Claude's thinking process should naturally aware of and adapt to the unique characteristics in human message:
    - Scale depth of analysis based on:
      * Query complexity
      * Stakes involved
      * Time sensitivity
      * Available information
      * Human's apparent needs

**任务描述：**

我现在要进行一个Matlab代码转化C++程序的任务。

**具体背景：**

1. 我在Matlab上处理的h5文件过于慢，所以我想转化到C++环境中去处理，我处理的文件件都用的是h5文件保存。
2. 具体的处理算法我已经用Matlab写过了，给你提供代码。
3. 我处理的文件有三个，分别是EFD的h5文件、HPM的h5文件，SCM的h5文件。
4. 我使用的软件分别是Matlab和Visual Studio。
5. 矩阵计算库使用的是Armadillo库。

**具体任务：**

1. 请帮我设计一个具体详细的转化C++的步骤，类似于任务书的报告，具体到每一个函数功能。
2. 给我提供详细的代码指导。我还想把这些函数功能封装起来，就直接调用这个库里面的功能？所以给我一个具体的指导过程。
3. 补充：画图的就不用重构C++代码了，代码中是到100行之前。

通过明确具体、提供必要的上下文、结构化的需求，以及设定明确的期望，我们才可以获得更准确、详尽且符合需求的回答。

## 2. 利用GPT来完成一个具体的深度学习任务

接下来我们来讲解一下如何利用gpt来完成一个宠物狗的年龄识别的代码编写任务。
首先第一步，我们把完整的任务背景复制完，发给他，并重点备注：你现在的任务是了解任务背景，在接到我下一步任务之前，无需生成解释。（下文照片里面的内容文字粘贴给它）
![ChatGPT Success](https://github.com/user-attachments/assets/73c8f53c-f8e6-484b-be3d-d96d657cbc4c)
接下来是具体的步骤处理，我们开始获取代码
### 步骤 1：数据预处理

**用户：**

我需要如何预处理宠物犬的图片和标签数据？请给我相关的Python代码。

**ChatGPT：**

```python
# 数据预处理步骤
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PetAgeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = []
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    self.img_labels.append((parts[0], int(parts[1])))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0])
        image = Image.open(img_path).convert('RGB')
        age = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        return image, age

# 定义转换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 创建数据集和数据加载器
train_dataset = PetAgeDataset(annotations_file='annotations/train.txt', img_dir='images/', transform=train_transform)
val_dataset = PetAgeDataset(annotations_file='annotations/val.txt', img_dir='images/', transform=val_test_transform)
test_dataset = PetAgeDataset(annotations_file='annotations/test.txt', img_dir='images/', transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```

**旁白：**

“第一步，我们先把可爱的宠物犬照片和它们的年龄标签准备好。通过这个代码，我们不仅加载图片，还进行了数据增强，让模型更聪明哦！”

### 步骤 2：选择并构建模型

**用户：**

推荐一个适合年龄预测的深度学习模型，并提供PyTorch的代码实现。

**ChatGPT：**

```python
# 使用预训练的ResNet50模型进行年龄预测
import torch
import torch.nn as nn
from torchvision import models

class AgePredictor(nn.Module):
    def __init__(self):
        super(AgePredictor, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)  # 输出一个连续值

    def forward(self, x):
        x = self.model(x)
        return x

# 实例化模型
model = AgePredictor()
```

**旁白：**

“第二步，我们选择了强大的ResNet50模型，并稍作修改，让它能够预测宠物犬的年龄。”

### 步骤 3：定义训练流程

**用户：**

请给我一个训练模型的PyTorch代码，包括损失函数和优化器的设置。

**ChatGPT：**

```python
# 定义损失函数和优化器
import torch.optim as optim

criterion = nn.L1Loss()  # 平均绝对误差
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

    return model

# 开始训练
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
```

**旁白：**

“第三步，通过定义损失函数和优化器，我们的模型开始学习如何准确预测年龄啦。每天进步一点点，宠物们的年龄预测准确度也越来越高！”

### 步骤 4：评估模型

**用户：**

如何在验证集上计算平均绝对误差（MAE）？请提供Python代码。

**ChatGPT：**

```python
# 计算MAE
def calculate_mae(model, val_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_mae = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            outputs = model(inputs)
            mae = torch.abs(outputs - labels).sum().item()
            total_mae += mae
    mae = total_mae / len(val_loader.dataset)
    print(f'Validation MAE: {mae:.4f}')
    return mae

# 计算验证集的MAE
validate_mae = calculate_mae(trained_model, val_loader)
```

**旁白：**

“第四步，我们来看看模型的表现如何！通过计算验证集上的平均绝对误差（MAE），我们可以直观地了解预测的准确度。希望我们的MAE越来越低！”

### 步骤 5：生成并保存预测结果

**用户：**

请给我一个Python代码示例，展示如何使用训练好的模型对测试集进行预测，并将结果保存为pred_result.txt格式。

**ChatGPT：**

```python

# 生成预测结果并保存
def generate_predictions(model, test_loader, output_file='pred_result.txt'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.cpu().numpy().flatten()
            predictions.extend(preds)
    
    # 假设测试集的文件顺序与test_loader一致
    with open('annotations/test.txt', 'r') as f:
        test_filenames = [line.strip().split('\t')[0] for line in f]
    
    with open(output_file, 'w') as f:
        for fname, pred in zip(test_filenames, predictions):
            f.write(f'{fname}\t{int(round(pred))}\n')

# 生成并保存预测结果
generate_predictions(trained_model, test_loader, output_file='pred_result.txt')
print('预测结果已保存为pred_result.txt')
```

**旁白：**

“最后一步，我们的模型已经训练完毕，接下来就是展示成果的时候了！通过这段代码，我们将对测试集进行预测，并按照比赛要求保存结果。”

然后通过一点一点来问，我们可以得到完整的代码。请记住，GPT是拥有上下文的读取记忆的，但是它的存储有限，所以要提醒它：请仔细阅读我们刚刚的对话，不要遗漏任何细节。最后要是想简单一点，可以让它生成完整的代码，直接复制过来就可以。

## 总结
回到题目，被淘汰的永远只会是不会使用GPT开发的人们，未来的编程可能变得更加的低代码，任何行业的从事着都有可能都能做自己的网页博客，交流技术。这是技术变革带来的不可逆的转变，旧时代的坚守者上不了新时代的船。也许我们做的只有被迫拥抱。
![image2](https://github.com/user-attachments/assets/f0ddcc87-ce87-4e4b-8179-0b679a25ce0a)
在这信息奔流不息的时代，仿佛一条奔腾的黄河，携带着无尽的智慧与可能，席卷而来。ChatGPT，如同那黄河中的一叶扁舟，载着人类对知识的渴望与探索，驶向未知的彼岸。我们站在这变革的潮头，会深刻体会到自身的渺小，也正因渺小才要去做些什么。
# 科学是理解不可理解之事，工程是实现不可能之可能——冯·诺依曼
