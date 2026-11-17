# MNIST 手写数字识别 · 从零实现到框架

用同一个任务（MNIST 手写数字识别）串联四种实现：纯 NumPy 推理、纯 NumPy 数值微分训练、纯 NumPy 反向传播、PyTorch 框架实现。既可作为神经网络的入门学习路径，也便于对照"从零手写"与"框架封装"之间的差异。

## 版本总览

| 版本 | 实现方式 | 梯度计算 | 依赖框架 | 测试集准确率 | 入口 |
|------|----------|----------|----------|--------------|------|
| [v1](v1/README.md) | 纯 NumPy 推理（784 → 100 → 50 → 10，Sigmoid） | 无（加载预训练权重 `sample_weight.pkl`） | 无 | ~93.5% | `v1/neural_network.py` |
| [v2](v2/README.md) | 纯 NumPy 两层网络（784 → 50 → 10，Sigmoid） | 数值微分（中心差分） | 无 | 训练极慢，教学用途 | `v2/main.py` |
| [v3](v3/README.md) | 纯 NumPy 模块化层（784 → 50 → 10，ReLU） | 手动反向传播（Backpropagation） | 无 | ~97.0% | `v3/main.py` |
| [v4](v4/README.md) | PyTorch 复刻 v3 的网络与超参数 | autograd 自动微分 | PyTorch 2.x + torchvision | ~97.1% | `v4/main.py` |

## 目录结构

```
mnist_neural_network/
├── v1/                  # 纯 NumPy 推理（预训练权重）
│   ├── neural_network.py
│   ├── sample_weight.pkl
│   ├── common/functions.py
│   ├── dataset/mnist.py
│   └── README.md
├── v2/                  # 纯 NumPy 训练（数值微分）
│   ├── data/            # 数据集下载与解析
│   ├── models/          # TwoLayerNet：前向、损失、数值梯度
│   ├── utils/           # 数学函数与训练器
│   ├── config.py
│   ├── main.py
│   └── README.md
├── v3/                  # 纯 NumPy + 反向传播
│   ├── data/
│   ├── models/          # layers.py（Affine/ReLU/SoftmaxWithLoss）+ two_layer_net.py
│   ├── utils/
│   ├── config.py
│   ├── main.py
│   └── README.md
├── v4/                  # PyTorch 实现
│   ├── data/
│   ├── models/
│   ├── train.py
│   ├── config.py
│   ├── main.py
│   ├── GUIDE.md         # 面向初学者的逐步实现教程
│   └── README.md
└── README.md
```

## 快速开始

### 环境依赖

- Python 3.8+（v4 需要 3.10+）
- v1 / v2 / v3：仅 NumPy
- v4：PyTorch 2.x + torchvision（CPU 即可运行）

```bash
pip install numpy              # v1 / v2 / v3
pip install torch torchvision  # v4
```

### 运行

```bash
git clone https://github.com/hewenbiao123/mnist_neural_network.git
cd mnist_neural_network

# v1：纯 NumPy 推理（使用预训练权重）
cd v1 && python neural_network.py

# v2：纯 NumPy 训练（数值微分，速度极慢，仅用于教学）
cd v2 && python main.py

# v3：纯 NumPy + 反向传播训练
cd v3 && python main.py

# v4：PyTorch 训练
cd v4 && python main.py
```

每个版本都在**自己的目录内**运行入口脚本。首次运行会自动下载 MNIST 数据集（约 11MB）并缓存到本地，之后的运行直接读取缓存。

## 学习路径

- **v1 — 跑通推理**：加载预训练权重，完成一次前向传播与批量推理，先建立"网络输入输出"的直观认识。
- **v2 — 理解梯度**：用纯 NumPy 补齐完整训练流程（前向传播、交叉熵损失、数值梯度、SGD 更新）。数值微分要对每个参数做两次前向传播，速度极慢，但能把"梯度"讲清楚。
- **v3 — 掌握反向传播**：把数值微分换成反向传播，训练速度提升约 4 万倍；同时引入 `Affine` / `Relu` / `SoftmaxWithLoss` 等层抽象，每层封装 `forward()` / `backward()`，隐藏层激活由 Sigmoid 改为 ReLU。
- **v4 — 熟悉框架**：网络结构与超参数不变，改用 PyTorch 实现——`nn.Linear` 对应 `Affine`，`nn.ReLU` 对应 `Relu`，`nn.CrossEntropyLoss` 对应 `SoftmaxWithLoss`，`loss.backward()` 对应 v3 中逐层调用的 `backward()`。[GUIDE.md](v4/GUIDE.md) 提供了从零实现 v4 各文件的逐步教程。

## 数据集

使用 [MNIST](http://yann.lecun.com/exdb/mnist/)：60,000 张训练图 + 10,000 张测试图，均为 28×28 灰度图，10 个类别（数字 0–9）。

首次运行自动下载并缓存在各版本的数据目录（`v1/dataset/`、`v2/data/`、`v3/data/`、`v4/data/`）。缓存与原始数据文件均已通过 `.gitignore` 排除，不会进入版本控制；若网络受限，可手动下载 4 个 IDX 文件放入对应目录，各版本 README 中有具体说明。

## 参考资源

- [Deep Learning from Scratch（斋藤康毅）](https://github.com/oreilly-japan/deep-learning-from-scratch) — 本项目主要参考书籍
- [MNIST 数据集](http://yann.lecun.com/exdb/mnist/) — Yann LeCun 提供的手写数字数据集
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html) — v4 实现所用框架
