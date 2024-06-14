---
title: 深入理解常用统计误差评估指标
published: 2024-06-06
description: "这次学习了期望均值方差，另外介绍了常用的统计误差评估指标，包括均方误差（MSE）、平均绝对误差（MAE）、均方根误差（RMSE）、平均绝对百分比误差（MAPE）和决定系数（$R^2$）。"
tags: ["期望", "mse"]
category: 学习记录
draft: false
---

# 期望、均值和方差

期望、均值和方差是概率统计中的重要概念，我们先来学习这几个基础概念。

### 2.4.1 方差、矩及随机变量的函数和期望规则

- **期望**：随机变量及其分布的重要特征，用来量化随机变量平均情况下的结果。
### 对于离散随机变量

对于一个离散随机变量 \(X\)，其数学期望 \(E(X)\) 定义为其所有可能值的加权平均，权重即为每个值的概率：

$$
E(X) = \sum_{i} x_i P(X = x_i)
$$

其中，\(x_i\) 表示随机变量 \(X\) 可以取的值，\(P(X = x_i)\) 是随机变量 \(X\) 取这些值的概率。

### 对于连续随机变量

对于连续随机变量，期望是其概率密度函数 \(f(x)\) 与其值的乘积的积分：

$$
E(X) = \int_{-\infty}^{\infty} x f(x) dx
$$

在这里，\(f(x)\) 是随机变量 \(X\) 的概率密度函数，表示 \(X\) 在某一点值的概率密度。

- **方差公式**：
  $$
  \operatorname{Var}(X) = E[(X - E[X])^2]
  $$
- **计算示例**：对于随机变量 $X$，其分布列 $p_X(x)$ 定义如下：
  $$
  p_X(x) = \begin{cases} 
  \frac{1}{9}, & \text{如果 } x \text{ 是 } [-4,4] \text{ 中的整数} \\ 
  0, & \text{其他} 
  \end{cases}
  $$
  均值 $E[X] = 0$（从分布列的对称性或直接计算）。方差计算如下：
  $$
  \operatorname{Var}(X) = \frac{1}{9} \sum_{x=-4}^4 x^2 = \frac{60}{9}
  $$

### 2.4.2 均值和方差的性质

- **常见误区**：除非 $g(X)$ 是线性函数，否则 $E[g(X)] \neq g(E[X])$。
- **例子**：Alice 根据天气步行或骑摩托上学，步行或骑摩托的速度不同，因此计算平均时间时不能简单地使用速度的平均值来反推时间。

### 2.4.3 某些常用的随机变量的均值和方差

- **伯努利随机变量**：
  $$
  p_X(k) = \begin{cases} 
  p, & \text{如果 } k = 1 \\ 
  1-p, & \text{如果 } k = 0 
  \end{cases}
  $$
  均值和方差公式：
  $$
  \begin{aligned}
  E[X] & = p \\
  E[X^2] & = p \\
  \operatorname{Var}(X) & = p(1-p)
  \end{aligned}
  $$

- **离散均匀随机变量（例如抛骰子）**：
  $$
  p_X(k) = \begin{cases} 
  \frac{1}{6}, & \text{如果 } k = 1,2,3,4,5,6 \\ 
  0, & \text{其他} 
  \end{cases}
  $$
  均值 $E[X] = 3.5$，方差：
  $$
  \operatorname{Var}(X) = \frac{1}{6}(1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2) - (3.5)^2 = \frac{35}{12}
  $$


## 1. 均方误差（MSE）

**定义**：MSE 是预测误差的平方和的平均值，用于衡量预测值与实际值之间的差异。

**公式**：
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

**公式讲解**：
- $n$ 是样本数量。
- $y_i$ 是第 $i$ 个实际值。
- $\hat{y}_i$ 是第 $i$ 个预测值。
- $(y_i - \hat{y}_i)^2$ 是单个预测误差的平方，平方项使得大的误差被赋予更大的权重。

**适用条件**：
- 数据集中异常值较少。
- 模型误差的分布较为均匀。

**推荐使用场景**：
- 在大多数机器学习回归任务中，特别是当我们想要强调避免大误差时，MSE 非常有用，因为大的误差在计算中会被放大。

**代码示例**：
```python
import tensorflow as tf
model.compile(optimizer='adam', loss='mse')
```
## 2. 平均绝对误差（MAE）

**定义**：MAE 是预测误差绝对值的平均值，衡量预测值与实际值之间的差异。

**公式**：
$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

**公式讲解**：
- $|y_i - \hat{y}_i|$ 是单个预测误差的绝对值，有助于平等处理所有大小的误差，避免误差方向的影响。

**适用条件**：
- 数据集中包含异常值。
- 当你需要对所有误差等同对待时。

**推荐使用场景**：
- 如果数据中包含离群点或异常值，并且我们不想这些值对总体误差评估产生过大影响，那么 MAE 是一个更好的选择。

**代码示例**：
```python
import tensorflow as tf
model.compile(optimizer='adam', loss='mae')
```
## 3. 均方根误差（RMSE）

**定义**：RMSE 是 MSE 的平方根，衡量预测值与实际值之间的差异。

**公式**：
$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

**公式讲解**：
- RMSE 通过取 MSE 的平方根来调整误差的量级，使其与原始数据的量级相同。

**适用条件**：
- 数据集中异常值较少。
- 误差分布需要和预测值保持相同的量纲。

**推荐使用场景**：
- 当我们需要误差指标与原始数据具有相同的单位时，RMSE 是合适的选择。它对大的误差更敏感，适用于需要重点关注避免大误差的应用。
**代码示例**：
```python
import tensorflow as tf
def rmse(y_true, y_pred):
    return tf.math.sqrt(tf.keras.losses.MSE(y_true, y_pred))

model.compile(optimizer='adam', loss=rmse)
```
## 4. 平均绝对百分比误差（MAPE）

**定义**：MAPE 表示误差大小与实际值之比的百分比的平均值。

**公式**：
$$
\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^n \left|\frac{y_i - \hat{y}_i}{y_i}\right|
$$

**公式讲解**：
- $\left|\frac{y_i - \hat{y}_i}{y_i}\right|$ 是单个预测误差与实际值的比例的绝对值，用于表示误差相对于实际值的大小。

**适用条件**：
- 实际值 $y_i$ 都远离零，避免因实际值接近零导致的误差放大。

**推荐使用场景**：
- 当我们需要衡量误差相对于真实值的重要性时，MAPE 特别有用。它在财务和预算预测中尤其受欢迎，因为它们通常关注相对误差百分比。
**代码示例**：
```python
import tensorflow as tf
model.compile(optimizer='adam', loss='mape')
```

## 5. $R^2$ （决定系数）

**定义**：$R^2$ 表示模型预测值的变异程度占实际值变异程度的比例，用于衡量模型的解释力。

**公式**：
$$
R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \overline{y})^2}
$$

**公式讲解**：
- 分子 $\sum_{i=1}^n (y_i - \hat{y}_i)^2$ 是模型误差的总平方和，称为残差平方和（RSS）。
- 分母 $\sum_{i=1}^n (y_i - \overline{y})^2$ 是总变异的总平方和，称为总平方和（TSS）。
- $R^2$ 的值越接近 1，说明模型的解释力越强，预测的准确性越高。

**推荐使用场景**：
- 在统计建模和机器学习的回归分析中广泛使用，特别是在评估模型的预测效果和解释变量重要性时。

以上就是本次的内容，从期望方差出发，然后记录了时间序列数据中这几个评估指标，我们应该清楚什么时候使用。大部分情况下，mse使用的范围最广。