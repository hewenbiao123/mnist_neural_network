# MNIST 两层神经网络（PyTorch）实现指南

> 本文档面向**完全没接触过 PyTorch** 的学习者。目标是让你自己动手，从零写出 `mnist_neural_network_v4` 项目中的 5 个 Python 文件。
>
> 建议顺序：先完成第 0 节的 **conda 环境准备**，再读完第 1、2、3 节，然后按第 4 节的顺序**逐个文件**实现，每写完一个就做一次「自测」，通过后再写下一个。卡住时再看第 6 节的参考答案。

---

## 目录

0. [前置准备：conda 环境](#0-前置准备conda-环境)
1. [先理解 6 个核心概念](#1-先理解-6-个核心概念)
2. [文件夹架构设计](#2-文件夹架构设计)
3. [整体流程设计（pipeline）](#3-整体流程设计pipeline)
4. [每一部分的具体实现](#4-每一部分的具体实现)
5. [常见错误排查](#5-常见错误排查)
6. [附录：完整参考实现](#6-附录完整参考实现)

---

## 0. 前置准备：conda 环境

本文假设你使用 **conda** 管理 Python 环境。下面所有命令都在 PowerShell（Windows）中执行。

### 0.1 确认 conda 可用

```bash
conda --version
```

能打印出版本号（如 `conda 24.x.x`）即可。

### 0.2 创建并激活一个独立环境

```bash
conda create -n mnist-torch python=3.10 -y
conda activate mnist-torch
```

- `-n mnist-torch`：环境名，可改成你喜欢的名字
- `python=3.10`：指定 Python 版本
- `-y`：自动确认，避免中途卡在交互提问

> **为什么用 3.10 而不是最新的 3.13/3.14**：太新的 Python 版本，PyTorch 的 GPU 版安装包（wheel）往往还没跟上，容易报「找不到对应版本」。3.10 是目前 PyTorch 支持最稳、配套最全的版本。

### 0.3 安装 PyTorch 与 torchvision（用 pip，不用 conda）

> **为什么用 pip 而不是 `conda install`**：PyTorch 2.6 起官方已不再发布 conda 包；而且 conda 下载 GPU 版大包（约 2GB）容易网络中断、无法续传。pip 更稳，还支持国内镜像加速。

先确认你的 CUDA 版本（决定装哪个 GPU 版）：

```bash
nvidia-smi
```

看输出右上角的 `CUDA Version`。如果没有 NVIDIA 显卡或没装驱动（命令报错），直接装 CPU 版即可。

**CPU 版**（没有 NVIDIA 显卡，或只想快速跑通）：

```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**GPU 版**（有 NVIDIA 显卡，把 `cu132` 换成你 `nvidia-smi` 对应的版本）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

CUDA 版本与命令的对应关系：

| nvidia-smi 显示的 CUDA Version | 命令里用 |
|-------------------------------|---------|
| 13.4 | `cu132`（注意：PyTorch 没有 `cu134` 的 Windows 版） |
| 13.2 | `cu132` |
| 13.0 | `cu130` |
| 12.6 | `cu126` |

> 官方源慢的话，可用阿里云镜像（注意用 `-f` 而不是 `-i`）：
>
> ```bash
> pip install torch torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cu132/ -i https://mirrors.aliyun.com/pypi/simple/
> ```

### 0.4 验证安装

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- 能打印出版本号即表示安装成功。
- GPU 版应显示类似 `2.13.0+cu132 True`（`cuda.is_available()` 为 `True`）；CPU 版显示 `2.13.0+cpu False`。

> 之后的每一条 `python ...` 命令，都要在**已激活的 `mnist-torch` 环境**里执行。如果关闭了终端，重新打开后记得先执行 `conda activate mnist-torch`。

---

## 1. 先理解 6 个核心概念

在写代码前，先搞懂 PyTorch 最常用的几个概念。每个概念我都会用一个"最小例子"让你亲手感受，不用背，理解即可。

### 1.1 张量 Tensor

张量就是**能上 GPU、能自动求导的多维数组**，可以粗暴地理解成 `numpy` 数组的升级版。

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # 从列表创建
print(x.shape)   # torch.Size([2, 2])   形状
print(x.dtype)   # torch.float32        类型
print(x.device)  # cpu                 所在设备

y = x @ x.T      # 矩阵乘法
z = x.view(4, -1)  # 改变形状，-1 表示自动推断
```

一句话：**所有参与训练的数据（图片、标签、模型参数）最终都是 Tensor**。

### 1.2 自动求导 autograd

这是 PyTorch 最核心的能力。你只需要用 Tensor 搭建计算过程，调用 `.backward()` 就能自动算出梯度。

```python
x = torch.tensor(2.0, requires_grad=True)  # 告诉框架：我要对这个变量求导
y = x ** 2 + 3 * x                          # y = x^2 + 3x
y.backward()                                # 反向传播
print(x.grad)                               # 7，因为 dy/dx = 2x + 3
```

要点：**梯度默认会累加**，所以训练每步开始前都要清空梯度（后面会用到 `optimizer.zero_grad()`）。

### 1.3 神经网络模块 nn.Module

在 PyTorch 里，一个神经网络就是一个继承 `nn.Module` 的类。你需要做两件事：

1. 在 `__init__` 里定义网络用到的层；
2. 在 `forward` 里定义数据如何流过这些层。

**你不需要手写反向传播**，autograd 会自动帮你算。

```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)   # 一个全连接层：输入 10 维，输出 5 维

    def forward(self, x):
        return self.fc(x)

net = Net()
out = net(torch.randn(3, 10))   # 直接像函数一样调用
print(out.shape)                # torch.Size([3, 5])
```

### 1.4 损失函数 Loss

损失函数用来衡量"模型预测"和"真实答案"差多远。训练的目标就是**让损失越来越小**。

多分类问题最常用 `nn.CrossEntropyLoss`，它内部已经包含了 softmax，所以**模型只需输出原始分数（logits），不要手动加 softmax**。

```python
criterion = nn.CrossEntropyLoss()
logits = torch.randn(3, 10)            # 3 个样本，每个 10 个分数
labels = torch.tensor([2, 5, 0])       # 真实类别（用索引，不是 one-hot）
loss = criterion(logits, labels)
print(loss.item())
```

### 1.5 优化器 optimizer

优化器负责"用梯度更新参数"，也就是完成 $\theta \leftarrow \theta - \eta \nabla L$ 这件事。

```python
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.1)  # SGD，学习率 0.1
```

训练时固定四步：`zero_grad` → `backward` → `step`，中间夹着前向和算 loss（详见 1.6 后面的完整套路）。

### 1.6 数据集与数据加载器 Dataset / DataLoader

- `Dataset`：负责"给我第 i 个样本"。
- `DataLoader`：负责"把样本打包成一个个 batch，并打乱顺序"。

对于 MNIST，PyTorch 自带了现成的 `torchvision.datasets.MNIST`，我们不用自己写 Dataset：

```python
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])  # 把图像转成 Tensor 并归一化到 [0,1]
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

img, label = train_set[0]
print(img.shape)   # torch.Size([1, 28, 28])
print(label)       # 5（0~9 的类别索引）
```

```python
from torch.utils.data import DataLoader
train_loader = DataLoader(train_set, batch_size=100, shuffle=True)

images, labels = next(iter(train_loader))
print(images.shape, labels.shape)   # (100,1,28,28)  (100,)
```

> ⚠️ 这里用 `transforms.ToTensor()` 只是为了演示 Dataset/DataLoader 的概念。**实际训练时不要用它处理 MNIST**，否则会因逐样本转换而变得很慢——正确做法见第 4.2 节（一次性向量化转成 Tensor）。

---

## 2. 文件夹架构设计

项目的目录结构如下（`data/`、`models/`、`utils/` 三个包各自放不同职责的代码）：

```text
mnist_neural_network_v4/
├── data/
│   └── mnist_loader.py      # 数据加载：下载数据集 + 构造 DataLoader
├── models/
│   └── two_layer_net.py     # 网络定义：两层神经网络
├── utils/
│   └── trainer.py           # 训练与推理：训练循环、评估、推理
├── config.py                # 超参数：学习率、epoch 数等集中管理
├── main.py                  # 入口：把上面所有模块串起来
└── README.md                # 项目说明
```

各文件职责与依赖关系：

| 文件 | 职责 | 依赖谁 |
|------|------|--------|
| `config.py` | 集中存放超参数，只定义一个 `MNISTConfig` 类 | 无 |
| `data/mnist_loader.py` | 下载 MNIST 并返回两个 `DataLoader` | `torchvision`、`torch.utils.data` |
| `models/two_layer_net.py` | 定义 `TwoLayerNet` 网络 | `torch.nn` |
| `utils/trainer.py` | 提供 `train`、`evaluate`、`inference` 三个函数 | `config`、`torch` |
| `main.py` | 编排整个流程 | 以上全部 |

> 关于导入：`data/`、`models/`、`utils/` 目录下可以不放 `__init__.py`（Python 3 支持"命名空间包"）。只要在项目根目录 `mnist_neural_network_v4/` 下运行 `python main.py`，`from data.mnist_loader import ...` 就能正常工作。

---

## 3. 整体流程设计（pipeline）

整个程序是一条流水线，按顺序分为 6 个阶段：

| 阶段 | 做什么 | 在哪个文件 |
|------|--------|-----------|
| 1. 读配置 | 拿到所有超参数 | `config.py` |
| 2. 加载数据 | 得到训练集、测试集的 `DataLoader` | `data/mnist_loader.py` |
| 3. 定义模型 | 构造两层网络并搬到设备（CPU/GPU） | `models/two_layer_net.py` |
| 4. 准备训练组件 | 定义损失函数和优化器 | `utils/trainer.py` |
| 5. 训练循环 | 反复执行"前向 → 损失 → 反向 → 更新" | `utils/trainer.py` |
| 6. 评估与推理 | 算准确率、展示预测结果 | `utils/trainer.py` |

`main.py` 负责按顺序调用这些阶段。

训练循环里**每一个 batch** 的固定套路（务必背下来）：

```python
optimizer.zero_grad()          # 1. 清空上一步的梯度
outputs = model(images)        # 2. 前向传播，得到预测分数
loss = criterion(outputs, labels)  # 3. 计算损失
loss.backward()                # 4. 反向传播，自动求梯度
optimizer.step()               # 5. 更新参数
```

---

## 4. 每一部分的具体实现

下面按"从简单到复杂、从底层到顶层"的顺序逐个实现。每节包含：**目标 / 新知识 / 步骤 / 关键代码 / 自测**。

### 4.1 `config.py`

**目标**：定义一个 `MNISTConfig` 类，把所有超参数集中起来，方便以后修改。

**新知识**：无 PyTorch 内容，纯 Python 类。

**实现**：在 `__init__` 里给 `self` 挂上这些属性。

| 属性 | 值 | 含义 |
|------|----|----|
| `input_size` | 784 | 输入维度（28×28 展平） |
| `hidden_size` | 50 | 隐藏层神经元数 |
| `output_size` | 10 | 输出类别数（0~9） |
| `learning_rate` | 0.1 | 学习率 |
| `weight_init_std` | 0.01 | 权重初始化标准差 |
| `batch_size` | 100 | 每个 mini-batch 的大小 |
| `epochs` | 17 | 训练轮数 |
| `data_dir` | `"./data"` | 数据集缓存目录 |
| `random_seed` | 42 | 随机种子，保证可复现 |

**自测**：在根目录运行

```bash
python -c "from config import MNISTConfig; c = MNISTConfig(); print(c.hidden_size)"
```

应输出 `50`。

---

### 4.2 `data/mnist_loader.py`

**目标**：写一个函数 `get_mnist_dataloaders(data_dir, batch_size)`，返回 `(train_loader, test_loader)`。

**新知识**：

- `datasets.MNIST(root=..., train=..., download=...)`：MNIST 数据集；`train=True` 是训练集，`train=False` 是测试集。**不传 `transform` 时**，`dataset.data` 是原始像素（`torch.Tensor`，形状 `(60000, 28, 28)`，值 0~255），`dataset.targets` 是标签。
- `tensor.float() / 255.0`：向量化地把像素归一化到 `[0, 1]`。
- `tensor.unsqueeze(1)`：在位置 1 增加一维，把 `(N, 28, 28)` 变成 `(N, 1, 28, 28)`。
- `TensorDataset(images, labels)`：把两个 Tensor 打包成一个 Dataset，取样本时只做切片（非常快）。
- `DataLoader(dataset, batch_size=..., shuffle=...)`：批量 + 打乱。

> ⚠️ **性能陷阱（务必注意）**：不要用 `transforms.ToTensor()` 来处理 MNIST！它会让 DataLoader 每取一个样本都走一遍 Python 层的 `__getitem__` + 转换，60000 个样本 × 17 个 epoch = 100 多万次重复调用，能把训练拖慢到几分钟。正确做法是**一次性向量化转成 Tensor**，之后每次取 batch 只是做切片。

**步骤**：

1. 用 `datasets.MNIST(..., download=True)` 加载原始数据（**不传 transform**）。
2. 一次性把像素转成 `float` 并除以 255，再 `unsqueeze(1)` 增加通道维。
3. 用 `TensorDataset` 把图像和标签打包。
4. 用 `DataLoader` 包起来：训练集 `shuffle=True`，测试集 `shuffle=False`。
5. 返回两个 loader。

**关键代码**：

```python
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


def get_mnist_dataloaders(data_dir="./data", batch_size=100):
    # 1. 加载原始数据（不加 transform）
    train_set = datasets.MNIST(root=data_dir, train=True, download=True)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True)

    # 2. 一次性向量化：归一化到 [0,1] + 增加通道维
    train_images = train_set.data.float() / 255.0      # (60000, 28, 28)
    train_images = train_images.unsqueeze(1)           # (60000, 1, 28, 28)
    test_images = test_set.data.float() / 255.0
    test_images = test_images.unsqueeze(1)             # (10000, 1, 28, 28)

    # 3. 打包成 Dataset（纯 Tensor 切片，很快）
    train_dataset = TensorDataset(train_images, train_set.targets)
    test_dataset = TensorDataset(test_images, test_set.targets)

    # 4. 构造 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
```

> 提示：之前已经下载过数据，`download=True` 会自动跳过下载，不会重复下载。

**自测**：在根目录运行

```bash
python -c "from data.mnist_loader import get_mnist_dataloaders; tr, te = get_mnist_dataloaders(); imgs, labels = next(iter(tr)); print(imgs.shape, labels.shape)"
```

应输出 `torch.Size([100, 1, 28, 28]) torch.Size([100])`。

---

### 4.3 `models/two_layer_net.py`

**目标**：定义 `TwoLayerNet`，结构为 784 → 50 → 10，中间用 ReLU 激活。

**新知识**：

- `nn.Linear(in_features, out_features)`：全连接层，内部自带权重和偏置。
- `nn.ReLU()`：ReLU 激活函数。
- `nn.init.normal_(tensor, mean, std)` / `nn.init.zeros_(tensor)`：初始化权重和偏置（带下划线的函数是"原地修改"）。

**步骤**：

1. 继承 `nn.Module`，`super().__init__()`。
2. 定义三个成员：`self.fc1`、`self.relu`、`self.fc2`。
3. 初始化：`fc1` 和 `fc2` 的权重用正态分布 `N(0, weight_init_std)`，偏置全 0。
4. 写 `forward(x)`：先把输入展平成 `(batch, 784)`，再依次 `fc1 → relu → fc2`，**返回 logits（不要加 softmax）**。

**关键代码**：

```python
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        # 权重初始化
        nn.init.normal_(self.fc1.weight, mean=0.0, std=weight_init_std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=weight_init_std)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)   # (batch,1,28,28) -> (batch,784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)             # logits
        return x
```

**自测**：

```bash
python -c "import torch; from models.two_layer_net import TwoLayerNet; m=TwoLayerNet(); print(m(torch.randn(8,1,28,28)).shape)"
```

应输出 `torch.Size([8, 10])`。

---

### 4.4 `utils/trainer.py`

这是最核心也最长的一个文件，拆成 4 个函数来写：`evaluate`、`train`、`inference`。

**目标**：提供评估、训练、推理三个函数。

**新知识**：

- `nn.CrossEntropyLoss()`：损失函数。
- `optim.SGD(model.parameters(), lr=...)`：优化器。
- `model.train()` / `model.eval()`：切换训练/评估模式。
- `torch.no_grad()`：推理/评估时关闭梯度，省内存、加速。
- `torch.max(outputs, dim=1)`：返回每行的最大值和索引（索引即预测类别）。
- `torch.softmax(logits, dim=1)`：把分数转成概率。
- `tensor.to(device)`：把张量搬到 CPU 或 GPU。

#### 4.4.1 `evaluate(model, loader, device)` —— 算准确率

1. `model.eval()`。
2. 在 `torch.no_grad()` 里遍历 loader。
3. 对每个 batch：搬到 device、前向得到 logits、`torch.max` 取预测类别、统计正确数。
4. 返回 `正确数 / 总数`。

**关键代码**：

```python
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total
```

#### 4.4.2 `train(model, train_loader, test_loader, config, device)` —— 训练主循环

1. 创建 `criterion` 和 `optimizer`。
2. 外层循环 `epoch`（共 `config.epochs` 次）。
3. 内层循环遍历 `train_loader` 的每个 batch，执行 5 步套路。
4. 每个 epoch 结束后，调用 `evaluate` 得到训练/测试准确率并打印。

**关键代码**：

```python
def train(model, train_loader, test_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()                 # 1
            outputs = model(images)               # 2
            loss = criterion(outputs, labels)     # 3
            loss.backward()                       # 4
            optimizer.step()                      # 5

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{config.epochs} | loss {avg_loss:.4f} | "
              f"train acc {train_acc:.4f} | test acc {test_acc:.4f}")
```

#### 4.4.3 `inference(model, loader, device, num_samples)` —— 推理演示

1. `model.eval()`，在 `torch.no_grad()` 里取一个 batch。
2. `softmax` 得概率，`argmax` 得预测类别。
3. 打印前 `num_samples` 个样本的真实标签 / 预测标签 / 置信度。

**自测**：等 `main.py` 写完一起测。

---

### 4.5 `main.py`

**目标**：把前面的模块串起来，完成完整流程。

**新知识**：

- `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`：选择设备。
- `torch.manual_seed(seed)`：固定随机种子。
- `model.to(device)`：把模型搬到设备。

**步骤**：

1. 读配置 `config = MNISTConfig()`。
2. 固定种子。
3. 选设备。
4. 调 `get_mnist_dataloaders` 得到两个 loader。
5. 构造 `TwoLayerNet(...)` 并 `.to(device)`。
6. 调 `train(...)`。
7. 调 `inference(...)`。

**自测**：在根目录运行

```bash
python main.py
```

正常会依次看到数据加载、17 个 epoch 的训练日志、以及推理演示输出。最终测试准确率约 **97%**。

---

## 5. 常见错误排查

1. **`Expected all tensors to be on the same device`**
   模型、输入、标签不在同一设备。检查是否都 `.to(device)`。

2. **忘了 `optimizer.zero_grad()`**
   梯度会累加，训练会发散或 NaN。这是最高频的 bug。

3. **`CrossEntropyLoss` 标签报错**
   标签必须是类别索引（如 `tensor([2,5,0])`），不能是 one-hot。

4. **`DataLoader` 在 Windows 上报多进程错误**
   保持 `num_workers` 为默认值 0 即可（本项目的 loader 没传 `num_workers`，就是 0）。

5. **`ModuleNotFoundError: No module named 'config'`**
   要在项目根目录 `mnist_neural_network_v4/` 下运行，而不是在 `data/` 或 `utils/` 子目录里运行。

6. **训练很慢（几分钟才能跑完）**
   最常见的元凶是**逐样本预处理**：用 `transforms.ToTensor()` 处理 MNIST，会让 DataLoader 每取一个样本都走一遍 Python 层的 `__getitem__` + 转换，60000 样本 × 17 epoch = 100 多万次调用，把训练拖慢到几分钟。改成"一次性向量化转成 Tensor + `TensorDataset`"后，应该十几秒就能跑完。详见 4.2 节。

---

## 6. 附录：完整参考实现

> 强烈建议：先按第 4 节自己写，遇到困难再对照下面代码，不要一开始就照抄。

### `config.py`

```python
class MNISTConfig:
    def __init__(self):
        self.input_size = 784
        self.hidden_size = 50
        self.output_size = 10
        self.learning_rate = 0.1
        self.weight_init_std = 0.01
        self.batch_size = 100
        self.epochs = 17
        self.data_dir = "./data"
        self.random_seed = 42
```

### `data/mnist_loader.py`

```python
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


def get_mnist_dataloaders(data_dir="./data", batch_size=100):
    train_set = datasets.MNIST(root=data_dir, train=True, download=True)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True)

    train_images = train_set.data.float() / 255.0
    train_images = train_images.unsqueeze(1)
    test_images = test_set.data.float() / 255.0
    test_images = test_images.unsqueeze(1)

    train_dataset = TensorDataset(train_images, train_set.targets)
    test_dataset = TensorDataset(test_images, test_set.targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
```

### `models/two_layer_net.py`

```python
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

        nn.init.normal_(self.fc1.weight, mean=0.0, std=weight_init_std)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=weight_init_std)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### `utils/trainer.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim

from config import MNISTConfig


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


def train(model, train_loader, test_loader, config, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{config.epochs} | loss {avg_loss:.4f} | "
              f"train acc {train_acc:.4f} | test acc {test_acc:.4f}")


@torch.no_grad()
def inference(model, loader, device, num_samples=10):
    model.eval()
    shown = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1)

        for true_label, pred_label, prob in zip(labels, predicted, probs):
            print(f"真实: {true_label.item()} | 预测: {pred_label.item()} | "
                  f"置信度: {prob[pred_label].item():.4f}")
            shown += 1
            if shown >= num_samples:
                return
```

### `main.py`

```python
import torch

from config import MNISTConfig
from data.mnist_loader import get_mnist_dataloaders
from models.two_layer_net import TwoLayerNet
from utils.trainer import train, inference


def main():
    config = MNISTConfig()
    torch.manual_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_mnist_dataloaders(config.data_dir, config.batch_size)

    model = TwoLayerNet(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        weight_init_std=config.weight_init_std,
    ).to(device)

    train(model, train_loader, test_loader, config, device)
    inference(model, test_loader, device, num_samples=10)


if __name__ == "__main__":
    main()
```

---

祝你顺利写出自己的第一版 PyTorch 网络！有任何一步卡住，都可以回来问我。
