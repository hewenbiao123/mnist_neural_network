# MNIST 手写数字识别 — 两层神经网络（反向传播版）

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-1.21+-013243?logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Framework-None-red?logo=numpy" alt="No DL Framework">
</p>

<p align="center">
  <b>从零手写神经网络 · 纯 NumPy 实现 · 零深度学习框架依赖</b>
</p>

---

> **English:** A two-layer neural network for MNIST handwritten digit recognition, built from scratch using only NumPy. Compared to v2, the key upgrade is **replacing numerical differentiation with Backpropagation** — achieving ~40,000× speedup. No PyTorch / TensorFlow dependencies.

> **仓库说明：** 本目录是 [mnist_neural_network](https://github.com/hewenbiao123/mnist_neural_network) 合并仓库中的 **v3** 版本（原独立仓库 `mnist_neural_network_v3`），保留了原有提交历史。

## 关于本项目

本目录是同一仓库中 [v2](https://github.com/hewenbiao123/mnist_neural_network/tree/main/v2) 的升级版本，属于"从零实现深度学习"系列项目。目标是通过纯 NumPy 手动构建神经网络的每一部分（数据加载、层抽象、前向传播、反向传播、梯度下降），深入理解深度学习底层原理，而非仅仅调用框架 API。

**适合人群：** 正在学习深度学习基础、希望理解反向传播数学原理、准备面试或教学演示的开发者。

### 版本演进

| 版本 | 核心特点 | 定位 |
|------|---------|------|
| v2 | 数值微分求梯度，过程式编码 | 理解梯度概念，验证梯度计算正确性 |
| **v3** | **反向传播求梯度，模块化层抽象** | 掌握反向传播原理，构建可扩展网络架构 |

## 特性

- **纯 NumPy 实现** — 不依赖 PyTorch、TensorFlow、Keras 等任何深度学习框架
- **反向传播（Backpropagation）** — 通过链式法则高效计算梯度，训练速度比数值微分快约 40,000 倍
- **模块化的层抽象** — `Affine`、`ReLU`、`Sigmoid`、`SoftmaxWithLoss` 各自封装 `forward()` / `backward()`
- **完整的数学文档** — README 中包含所有前向/反向传播的数学公式推导
- **自动数据集管理** — 首次运行自动下载 MNIST 数据集并缓存为 pickle 文件
- **类型注解** — 完整的 `numpy.typing.NDArray` 类型标注，提升代码可读性
- **集中参数配置** — 所有超参数在 `config.py` 中统一管理

## 目录

- [网络架构](#网络架构)
- [数学公式](#数学公式)
- [反向传播详解](#反向传播详解)
- [项目结构](#项目结构)
- [模块详解](#模块详解)
- [v2 vs v3 核心变化](#v2-vs-v3-核心变化)
- [快速开始](#快速开始)
- [训练流程](#训练流程)
- [自定义配置](#自定义配置)
- [常见问题](#常见问题)
- [参考资源](#参考资源)
- [License](#license)

## 网络架构

```
输入层 (784)  →  Affine (W1,B1)  →  ReLU  →  隐藏层 (50)  →  Affine (W2,B2)  →  SoftmaxWithLoss  →  输出层 (10)
```

| 层 | 形状变换 | 参数量 |
|----|---------|--------|
| 输入 | `(batch, 784)` | — |
| Affine1（W1 + B1） | `(784, 50)` + `(50,)` | 39,250 |
| ReLU | `(batch, 50)` | — |
| Affine2（W2 + B2） | `(50, 10)` + `(10,)` | 510 |
| SoftmaxWithLoss | `(batch, 10)` | — |
| **总计** | | **39,760** |

## 数学公式

### 前向传播

$$
Z_1 = X W_1 + B_1
$$

$$
A_1 = \mathrm{ReLU}(Z_1) = \max(0, Z_1)
$$

$$
Z_2 = A_1 W_2 + B_2
$$

$$
\hat{Y} = \mathrm{softmax}(Z_2) = \frac{e^{Z_2 - \max(Z_2)}}{\sum e^{Z_2 - \max(Z_2)}}
$$

### 损失函数 — 交叉熵误差

$$
L = -\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{K} t_{nk} \ln(\hat{y}_{nk} + \delta)
$$

- N: 批量大小，K: 类别数（10），δ = 1e⁻⁷

## 反向传播详解

v3 的核心：通过链式法则逐层反向传递梯度，每个参数的梯度在对应层的 `backward()` 中一次计算完成。

### 反向传播链路

反向传播按从输出层到输入层的顺序逐层计算梯度：

1. **SoftmaxWithLoss.backward()** → 计算 `dZ₂ = (Ŷ - Y) / N`
2. **Affine2.backward(dZ₂)** → 计算 `dW₂ = A₁ᵀ · dZ₂`、`dB₂ = Σ dZ₂`，返回 `dA₁ = dZ₂ · W₂ᵀ`
3. **ReLU.backward(dA₁)** → 计算 `dZ₁ = dA₁ ⊙ (Z₁ > 0)`
4. **Affine1.backward(dZ₁)** → 计算 `dW₁ = Xᵀ · dZ₁`、`dB₁ = Σ dZ₁`，返回 `dX = dZ₁ · W₁ᵀ`

### 各层局部梯度公式

**SoftmaxWithLoss：**

$$
\frac{\partial L}{\partial Z_2} = \frac{\hat{Y} - Y}{N}
$$

**Affine（全连接层）：**

$$
\frac{\partial L}{\partial W} = X^\mathsf{T} \cdot \frac{\partial L}{\partial Z}, \quad
\frac{\partial L}{\partial B} = \sum \frac{\partial L}{\partial Z}, \quad
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} \cdot W^\mathsf{T}
$$

**ReLU：**

$$
\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \mathbb{1}(Z_1 > 0)
$$

**Sigmoid（备选，当前未启用）：**

$$
\frac{\partial L}{\partial Z_1} = \frac{\partial L}{\partial A_1} \odot \sigma(Z_1) \odot (1 - \sigma(Z_1))
$$

### 性能对比

| | v2（数值微分） | v3（反向传播） |
|----|----|----|
| 每个 mini-batch 的前向传播次数 | ~79,520 次 | **2 次**（1 forward + 1 backward） |
| 1 epoch（600 batches）的前向传播 | ~4,770 万次 | **~1,200 次** |
| 速度比 | 1× | **~40,000×** |

## 项目结构

```
v3/
├── data/
│   └── mnist_loader.py      # 数据集：下载、解析二进制 IDX 文件、pickle 持久化
├── models/
│   ├── layers.py            # 🆕 层级模块：Affine、Sigmoid、ReLU、SoftmaxWithLoss
│   └── two_layer_net.py     # 模型：使用 layers 组装网络，反向传播计算梯度
├── utils/
│   ├── function.py          # 数学工具：Sigmoid、ReLU、Softmax、交叉熵误差
│   └── trainer.py           # 训练器：Mini-batch SGD、loss/accuracy 记录、日志输出
├── config.py                # 超参数集中管理
├── main.py                  # 程序入口
└── README.md
```

## 模块详解

### `models/layers.py` — 🆕 层级模块

将神经网络各层抽象为独立类，每层封装前向计算和反向梯度传播：

| 类 | `forward()` | `backward(dout)` | 说明 |
|----|-------------|-------------------|------|
| `Affine` | `X·W + B` | 计算 `dW, dB`，返回 `dX` | 全连接层，保存 `X` 用于反向传播 |
| `Sigmoid` | `σ(Z)` | `dout ⊙ σ(Z) ⊙ (1-σ(Z))` | 备选隐藏层激活（当前注释） |
| `Relu` | `max(0, Z)` | 屏蔽 `Z ≤ 0` 位置的梯度 | **默认隐藏层激活**，保存零值掩码 |
| `SoftmaxWithLoss` | `softmax(Z)` + 交叉熵 | `(Ŷ - Y) / N` | 输出层，同时处理 one-hot 和非 one-hot 标签 |

**设计要点：**

- 每个层在 `forward()` 中保存必要的中间变量（如 `Affine` 保存 `X`，`Relu` 保存零值掩码），供 `backward()` 使用
- `backward(dout)` 接收上游传来的梯度，计算本层参数的梯度并返回传递给下一层的梯度
- `SoftmaxWithLoss.backward()` 无需参数，返回 `∂L/∂Z₂`，作为反向传播的起点

### `models/two_layer_net.py` — 网络模型（重构）

**与 v2 的核心区别：**

```python
# v2：过程式前向传播
def predict(self, x):
    Z1 = Function.sigmoid(np.dot(x, self.params['W1']) + self.params['b1'])
    return Function.softmax(np.dot(Z1, self.params['W2']) + self.params['b2'])

# v3：层级式组装
self.layers = [Affine(W1, B1), Relu(), Affine(W2, B2)]
self.lastLayer = SoftmaxWithLoss()

def predict(self, X):
    for layer in self.layers:
        X = layer.forward(X)
    return X
```

**方法说明：**

| 方法 | 说明 |
|------|------|
| `predict(X)` | 遍历 `self.layers` 逐层 `forward()`，返回 softmax 概率 |
| `forward(X, Y)` | `predict(X)` + `lastLayer.forward()`，返回 loss |
| `backward()` | 从 `lastLayer` 反向逐层 `backward()`，收集各 Affine 层的 `dW/dB` |
| `accuracy(X, Y)` | 计算分类准确率，支持 one-hot 和原始标签 |
| `update(X, Y, lr)` | `forward → backward → θ -= lr * ∇L`，一次完成训练步 |

**权重维护结构：**

```python
self.parameters = {
    'W1': NDArray (784, 50),
    'B1': NDArray (50,),
    'W2': NDArray (50, 10),
    'B2': NDArray (10,),
}
self.layers = [Affine(W1, B1), Relu(), Affine(W2, B2)]
self.lastLayer = SoftmaxWithLoss()
```

### `data/mnist_loader.py` — 数据加载器

| 类/方法 | 说明 |
|---------|------|
| `MNISTMeta` | 数据集元信息常量（URL、文件名、样本数、偏移量等） |
| `MNISTLoader.load()` | 主接口，返回 `(训练集, 测试集)`；支持归一化、展平、one-hot 编码 |
| `MNISTLoader.init()` | 首次调用时触发：下载 → 转换 → pickle 保存 |

**`load()` 方法参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `normalize` | `True` | 像素值除以 255，归一化到 [0, 1] |
| `flatten` | `True` | 图像展平为 `(N, 784)`；False 时保持 `(N, 1, 28, 28)` |
| `one_hot_label` | `False` | 标签转为 one-hot 编码 `(N, 10)` |

### `utils/function.py` — 数学函数

| 方法 | 公式 | 说明 |
|------|------|------|
| `sigmoid(x)` | 1/(1+e⁻ˣ) | 激活函数 |
| `relu(x)` | max(0, x) | 激活函数（v3 隐藏层默认使用） |
| `softmax(x)` | exp(x-max)/Σexp(x-max) | 输出概率分布 |
| `cross_entropy_error(y, t)` | -Σt·log(y+δ)/N | 损失函数 |

### `utils/trainer.py` — 训练循环

- 使用 **Mini-batch SGD**
- 每个 epoch 开始前随机打乱训练集
- 每个 mini-batch 调用 `network.update()`（内部自动完成 forward + backward + 参数更新）
- 每个 epoch 结束后在全量训练集和测试集上评估 accuracy
- batch loss 的打印已注释，减少控制台输出

### `config.py` — 超参数配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_size` | int | 784 | 输入维度（28×28 像素展平） |
| `hidden_size` | int | 50 | 隐藏层神经元数 |
| `output_size` | int | 10 | 输出类别（数字 0-9） |
| `learning_rate` | float | 0.1 | SGD 学习率 |
| `weight_init_std` | float | 0.01 | 权重高斯初始化标准差 |
| `batch_size` | int | 100 | Mini-batch 大小 |
| `epochs` | int | 17 | 训练轮数 |

## v2 vs v3 核心变化

| 维度 | v2 | v3 |
|------|----|----|
| **梯度计算** | 数值微分（中心差分），每个参数需 2 次前向传播 | **反向传播（链式法则）**，一次前向 + 一次反向 |
| **训练速度** | 每个 epoch 约需 4,770 万次前向传播，极慢 | 每个 epoch 仅需约 600 次前向传播，秒级完成 |
| **模块化程度** | 网络前向/反向逻辑耦合在 `TwoLayerNet` 中 | 新增 `layers.py`，每层独立封装 `forward()` / `backward()` |
| **隐藏层激活** | Sigmoid | **ReLU**（缓解梯度消失，收敛更快） |
| **权重字典 key** | `W1, W2, b1, b2` | `W1, W2, B1, B2` |
| **架构模式** | 过程式调用静态方法 | **层的链表式组合**，`self.layers` 列表 + 统一 `forward/backward` 接口 |

## 快速开始

### 环境要求

- Python 3.8+
- NumPy 1.21+

### 安装

```bash
# 克隆仓库
git clone https://github.com/hewenbiao123/mnist_neural_network.git
cd mnist_neural_network/v3

# 安装依赖
pip install numpy
```

### 运行

```bash
python main.py
```

首次运行会自动下载 MNIST 数据集（约 11MB）并保存为 pickle 缓存，后续运行直接加载。

运行输出示例：

```
Starting  Model Training ...
Epoch  1/17 | train acc: 0.9009, test acc: 0.9064
Epoch  2/17 | train acc: 0.9229, test acc: 0.9263
Epoch  3/17 | train acc: 0.9354, test acc: 0.9363
Epoch  4/17 | train acc: 0.9449, test acc: 0.9420
Epoch  5/17 | train acc: 0.9529, test acc: 0.9500
Epoch  6/17 | train acc: 0.9584, test acc: 0.9553
Epoch  7/17 | train acc: 0.9631, test acc: 0.9584
Epoch  8/17 | train acc: 0.9663, test acc: 0.9596
Epoch  9/17 | train acc: 0.9697, test acc: 0.9632
Epoch 10/17 | train acc: 0.9721, test acc: 0.9644
Epoch 11/17 | train acc: 0.9735, test acc: 0.9670
Epoch 12/17 | train acc: 0.9748, test acc: 0.9661
Epoch 13/17 | train acc: 0.9775, test acc: 0.9687
Epoch 14/17 | train acc: 0.9789, test acc: 0.9686
Epoch 15/17 | train acc: 0.9792, test acc: 0.9683
Epoch 16/17 | train acc: 0.9810, test acc: 0.9705
Epoch 17/17 | train acc: 0.9816, test acc: 0.9698
Finished Model Training!
```

**预期结果：** 使用默认超参数，训练 17 个 epoch 后，训练集准确率约 **98.2%**，测试集准确率约 **97.0%**。

## 训练流程

1. **加载 MNIST 数据集** — 归一化到 [0,1]，标签转为 one-hot 编码
2. **初始化 TwoLayerNet** — 权重 N(0, 0.01)，偏置 0，构建层链 `Affine → ReLU → Affine → SoftmaxWithLoss`
3. **循环训练 17 个 epoch**，每个 epoch 中：
   - 随机打乱训练集
   - 按 batch_size=100 遍历所有 mini-batch：
     - `forward()` — 逐层前向计算，得到 loss
     - `backward()` — 逐层反向传播，计算梯度 ∇L(θ)
     - SGD 更新 — `θ -= lr × ∇L`
   - 在全量训练集和测试集上评估 accuracy

## 自定义配置

修改 `config.py` 即可调整训练参数：

```python
config = MNISTConfig()
config.hidden_size = 100      # 增加隐藏层神经元
config.learning_rate = 0.05   # 降低学习率
config.epochs = 30            # 增加训练轮数
config.batch_size = 50        # 减小批量
```

### 切换隐藏层激活函数

在 `models/two_layer_net.py` 中，将 `Relu()` 替换为 `Sigmoid()`：

```python
# self.layers.append(Relu())
self.layers.append(Sigmoid())
```

## 常见问题

### 数据集下载失败或速度慢

若所在网络无法访问 MNIST 官方源（如国内环境），可手动下载以下 4 个文件放入 `data/` 目录，程序会自动跳过下载步骤：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

下载地址：[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### 训练精度不理想

可以尝试以下调整：

1. **增加 `epochs`** —— 当前默认 17，可尝试 30~50
2. **调整学习率** —— 当前默认 0.1，可尝试 0.5、0.01 等
3. **增大隐藏层** —— 当前 50 个神经元，可尝试 100、200
4. **减小 `weight_init_std`** —— 当前 0.01，过大可能导致梯度爆炸

### 如何对比 v2 的数值微分模式？

v2 代码位于同一仓库的 [v2](https://github.com/hewenbiao123/mnist_neural_network/tree/main/v2) 目录，可直接切换过去运行：

```bash
git clone https://github.com/hewenbiao123/mnist_neural_network.git
cd mnist_neural_network/v2
python main.py
```

> **注意：** v2 使用数值微分，训练极其缓慢（每 epoch 可能需数小时），仅用于教学演示。

## 参考资源

- [Deep Learning from Scratch (斋藤康毅)](https://github.com/oreilly-japan/deep-learning-from-scratch) — 本项目的主要参考书籍
- [MNIST 数据集](http://yann.lecun.com/exdb/mnist/) — Yann LeCun 提供的经典手写数字数据集

---

<p align="center">
  <sub>If you find this project helpful, please consider giving it a ⭐ Star!</sub>
</p>
