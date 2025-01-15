---
title: ChatGPT Is All You Need
published: 2024-11-22
description: "我们不会被AI淘汰，被淘汰的一直都是不会使用工具的人，"
tags: ["ChatGPT", "AI", "GPT使用","prompt"]
category: 经验总结
draft: false
---

# ChatGPT Is All You Need
![image1](https://github.com/user-attachments/assets/eaf93bfa-1695-4059-8cab-04f43effb356)

## 目录

- [前言](#前言)
- [1. Prompt Engineering](#1-prompt-engineering)
- [2. 利用GPT来完成一个具体的深度学习任务](#2-利用gpt来完成一个具体的深度学习任务)
- [总结](#总结)

## 前言

### 今天这篇文章我想来谈谈如何正确使用GPT，即如何利用GPT来更快完成我们手头的工作呢？

在开始之前，先回顾一下GPT的国内外的发展历程。在2022年11月30日GPT3.5版本上线，迅速引起了广泛关注和热议。实际上，从上线当天起，ChatGPT 就在社交媒体、新闻报道以及各类讨论平台上迅速传播开来，ChatGPT也创造了当天用户增量最多的记录。用户量在12月内短短几天内激增至数百万，在 ChatGPT 推出仅两个月后，它在 2023 年 1 月末的月活用户已经突破了 1 亿，TikTok 达到 1 亿用户用了 9 个月，Instagram 则花了 2 年半的时间。如今，马上就要两年。

![User Growth Comparison](https://github.com/user-attachments/assets/5fb22442-8687-493b-ba3c-8dea83209f55)

**全球用户达到 1 亿所需时间：**

- 电话：75年
- 手机：16年
- 万维网：7年
- iTunes：6.5 年
- 推特：5年
- 脸书：4.5年
- WhatsApp：3.5 年
- Instagram：2.5年
- 苹果应用商店：2年
- ChatGPT：2个月

恐怕当时的OpenAI公司可能也不想到会做得如此成功。他们更多的是苦恼于GPT2.0失败的前车之鉴。不断堆叠transformer模型，是否会有希望？而且更多的科技公司是跟随Bert的路径研发自己的大模型，比如谷歌，谷歌都还没有成功，自己的GPT架构是否能有优秀表现？恐怕在训练好之前都还是一个未知数。但是，在当今GPT架构已经成为大模型的主流选择。在黄仁勋递给OpenAI公司自己的NVIDIA显卡的时候，OpenAI公司的命运已然迎来转折。

![OpenAI and NVIDIA](https://github.com/user-attachments/assets/31ad9b7f-67a5-45f6-bdb5-a3308437cbbd)

从2023年起，大语言模型（LLM）走进我们的视野，这次AI之风比以往（区块链，元宇宙等等）吹得更加强劲，它背后的真实性能让这次大语言模型彻底出圈。当时让深度学习分支之一，自然语言处理（NLP）的研究领域，遭受了灭顶之灾。从此NLP再也不是过家家的玩具，小型公司、实验室，无法负担起往后的大模型训练成本。NLP方向不会再是普通研究者的水论文的方式，能做的只有中下游任务的适配。随着ChatGPT的火爆，国内的各种套壳GPT网站也随之兴起，大家都想尝试这个新型的人工智能。与之而来的还有各种人工智能的卖课，最典型的是清华博士李一舟199元的AI课卖了5000万，又是一次风口，许多人乘风而起。

之后就是国内外大型公司推出的大模型竞品，都要来这市场分一杯羹。但在GPT4开源之前，国内的大语言模型永远只能追赶，ChatGPT4.0在各种领域的优秀表现可以用恐怖来形容，现在又推出的o1模型，更是让AI更加接近人类。o1的模型设计类似GitHub去年爆火的AutoGPT，现有用户的提示词水平不足以激发出模型的最大潜力，所以它在你第一次提问之后，会自己总结，自己给自己设计下一步的任务，来弥补我们提问时的缺陷。所以它会在你现有的基础上提供一个更加完美的答案。

![Advanced AI Models](https://github.com/user-attachments/assets/99e44070-0101-43a6-9905-8ed803fef815)

那么先进的代价是什么？代价则是大量的计算成本。从GPT4.0面世，到现在已经一年多，这一年多我们可以觉察到它的性能明显下降。一是OpenAI公司的安全政策原因，给现有的模型加了诸多的约束限制；二是要把算力分给更流行的模型，比如4o和o1模型。而且o1模型一周有着50条提问限制，比GPT4.0模型三小时50条更加严格。

ChatGPT的成功一定是人类走向生成式强人工智能的主要关键节点。在4o和o1的模型发布会上，OpenAI对模型的强劲性能没有过多赘述，反倒是一直在强调对于模型、公司的安全政策和措施，让模型坚持遵循OpenAI的政策，提供帮助和准确性，决不会违背内容政策等等。上次一直强调安全政策的发明还是核弹。

![OpenAI Safety Policies](https://github.com/user-attachments/assets/4a35cca8-8b11-4549-a8be-d0253bfbf1a6)

## 1. Prompt Engineering

闲言少叙，我们来到今天的重点——Prompt Engineering，提示词工程。提示词是继GPT火起来之后，随之而来的一项领域，可以说是小深度学习。但是往后提示词功能作用一定会越来越小，因为优秀模型与提示词的帮助是成反比的，越优秀的模型越不在意提示词。但是现在，如何写好我们GPT的提问呢？

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
