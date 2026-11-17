# MNIST 手写数字识别 · 从零实现到框架

本仓库把原先分散的四个独立仓库（`mnist_neural_network_v1` ~ `mnist_neural_network_v4`）合并为一个仓库，按版本收录在 `v1/` ~ `v4/` 四个子目录中。

四个版本是同一主题（MNIST 手写数字识别）上递进式的实现：从"加载预训练权重做纯 NumPy 推理"，到"纯 NumPy 手写训练"，再到"手动反向传播"，最后"用 PyTorch 复刻同一网络"。每个子目录都保留了原有代码、文档，以及原仓库的完整提交历史（通过 `git subtree` 迁移）。

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

```bash
# 克隆合并后的仓库
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

依赖安装：

```bash
pip install numpy            # v1 / v2 / v3
pip install torch torchvision  # v4
```

> 每个版本都在**自己的目录内**运行入口脚本。数据集首次运行时会自动下载（约 11MB），之后的缓存文件已被 `.gitignore` 排除，不会进入版本控制。

## 版本演进

- **v1 → v2**：从只有推理，到用纯 NumPy 实现完整训练流程（前向传播、交叉熵损失、数值梯度、SGD 更新）。数值微分要对每个参数做两次前向传播，速度极慢，目的是讲清"梯度"是什么。
- **v2 → v3**：把数值微分换成反向传播，训练速度提升约 4 万倍；同时引入 `Affine` / `Relu` / `SoftmaxWithLoss` 等层抽象，每层封装 `forward()` / `backward()`，隐藏层激活由 Sigmoid 改为 ReLU。
- **v3 → v4**：网络结构与超参数不变，改用 PyTorch 实现——`nn.Linear` 对应 `Affine`，`nn.ReLU` 对应 `Relu`，`nn.CrossEntropyLoss` 对应 `SoftmaxWithLoss`，`loss.backward()` 对应 v3 中逐层调用的 `backward()`，便于对照理解框架到底做了什么。

## 关于合并

- 四个版本原为四个独立仓库，现已通过 `git subtree` 迁入本仓库的 `v1/` ~ `v4/` 子目录，各自的提交历史完整保留。
- 各版本 README 中的目录名、克隆命令和跨版本链接已更新为合并后的路径；原独立仓库不再作为主要入口。
- 提交历史中早于合并的部分仍是原仓库的提交，因此部分早期提交的路径不带 `vN/` 前缀。
