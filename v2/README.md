# MNIST 手写数字识别 — 两层神经网络

基于 NumPy 从零实现的两层全连接神经网络，用于 MNIST 手写数字识别。所有组件（数据加载、激活函数、损失函数、梯度计算、参数更新、训练循环）均不依赖 PyTorch/TensorFlow 等深度学习框架。

## 目录

- [项目结构](#项目结构)
- [网络架构](#网络架构)
- [数学公式](#数学公式)
- [模块详解](#模块详解)
- [快速开始](#快速开始)
- [训练流程](#训练流程)
- [自定义配置](#自定义配置)
- [注意事项](#注意事项)

## 项目结构

```
mnist_neural_network_v2/
├── data/
│   └── mnist_loader.py   # 数据集：下载、解析二进制 IDX 文件、pickle 持久化
├── models/
│   └── two_layer_net.py  # 模型：前向传播、损失、精度、数值梯度、参数更新
├── utils/
│   ├── function.py       # 数学工具：Sigmoid、ReLU、Softmax、交叉熵误差
│   └── trainer.py        # 训练器：Mini-batch SGD、loss/accuracy 记录、日志输出
├── config.py             # 超参数集中管理
├── main.py               # 程序入口
└── README.md
```

## 网络架构

```
输入层 (784)  →  全连接 (W1,b1)  →  Sigmoid  →  隐藏层 (50)  →  全连接 (W2,b2)  →  Softmax  →  输出层 (10)
```

| 层 | 形状变换 | 参数量 |
|----|---------|--------|
| 输入 | `(batch, 784)` | — |
| 权重 W1 + 偏置 b1 | `(784, 50)` + `(50,)` | 39,250 |
| 隐藏层（Sigmoid 后） | `(batch, 50)` | — |
| 权重 W2 + 偏置 b2 | `(50, 10)` + `(10,)` | 510 |
| 输出（Softmax 后） | `(batch, 10)` | — |
| **总计** | | **39,760** |

## 数学公式

### 前向传播

$$
Z_1 = X W_1 + b_1
$$

$$
A_1 = \sigma(Z_1) \qquad \text{(Sigmoid)}
$$

$$
Z_2 = A_1 W_2 + b_2
$$

$$
\hat{Y} = \mathrm{softmax}(Z_2) \qquad \text{(输出概率分布)}
$$

其中 Sigmoid 函数：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

Softmax 函数（含数值稳定性处理）：

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}
$$

### 损失函数 — 交叉熵误差

$$
L = -\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{K} t_{nk} \ln(y_{nk} + \delta)
$$

- N: 批量大小
- K: 类别数（10）
- δ = 1e-7，防止 log(0)

### 梯度计算 — 数值微分（中心差分）

$$
\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + h) - L(\theta_i - h)}{2h}
$$

其中 h = 1e-4。对网络中全部 39,760 个参数逐个进行独立的前向传播计算。

### 参数更新 — SGD

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta L
$$

- η: 学习率

## 模块详解

### `data/mnist_loader.py` — 数据加载器

| 类/方法 | 说明 |
|---------|------|
| `MNISTMeta` | 数据集元信息常量（URL、文件名、样本数、偏移量等） |
| `MNISTLoader.load()` | 主接口，返回 `(训练集, 测试集)`；支持归一化、展平、one-hot 编码 |
| `MNISTLoader.init()` | 首次调用时触发：下载 → 转换 → pickle 保存 |
| `MNISTLoader.download()` | 从网络下载 4 个 `.gz` 压缩文件到 `data/` 目录 |
| `MNISTLoader.convert_numpy()` | 读取 `.gz` 文件并解析为 NumPy 数组 |
| `MNISTLoader._load_image()` | 解析图像 IDX 文件（offset=16，reshape 为 `(-1, 784)`） |
| `MNISTLoader._load_label()` | 解析标签 IDX 文件（offset=8） |
| `MNISTLoader.save_pickle()` | 将 NumPy 数组序列化为 `.pkl` 文件 |
| `MNISTLoader.load_pickle()` | 从 `.pkl` 文件反序列化，并转换为 `float32` |

**`load()` 方法参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `normalize` | `True` | 像素值除以 255，归一化到 [0, 1] |
| `flatten` | `True` | 图像展平为 `(N, 784)`；False 时保持 `(N, 1, 28, 28)` |
| `one_hot_label` | `False` | 标签转为 one-hot 编码 `(N, 10)` |

**数据流：**

```
网络下载 .gz → gzip 解压 + np.frombuffer 解析 → pickle 保存 → 加载时转 float32 → 归一化/编码
```

### `utils/function.py` — 数学函数

所有方法均为静态方法，以 `NDArray[np.float32]` 为输入输出。

| 方法 | 公式 | 说明 |
|------|------|------|
| `sigmoid(x)` | 1/(1+e⁻ˣ) | 隐藏层激活函数，输出范围 (0, 1) |
| `relu(x)` | max(0, x) | 备选激活函数（当前未使用） |
| `softmax(x)` | exp(x-max)/Σexp(x-max) | 输出层激活函数，输出概率分布，`keepdims=True` 保证广播正确 |
| `cross_entropy_error(y, t)` | -Σt·log(y+δ)/N | 损失函数，y 为预测概率，t 为 one-hot 标签 |

### `models/two_layer_net.py` — 网络模型

**`TwoLayerNet` 类：**

| 方法 | 说明 |
|------|------|
| `__init__(input, hidden, output, weight_init_std)` | 初始化 W₁、W₂（高斯分布 × std）和 b₁、b₂（全零） |
| `predict(x)` | 前向传播，返回 softmax 概率输出 |
| `loss(x, t)` | 计算交叉熵误差 |
| `accuracy(x, t)` | 计算分类准确率（argmax 比较） |
| `_numerical_gradient(x, arg, t)` | 对单个参数张量求数值梯度（中心差分），遍历每个元素 |
| `numerical_gradient(x, t)` | 遍历所有参数（W1, b1, W2, b2）分别求梯度 |
| `update(x, t, lr)` | 梯度下降更新：`θ -= lr * ∇L` |

**权重维护结构：**

```python
self.params = {
    'W1': NDArray (784, 50),   # 输入 → 隐藏
    'b1': NDArray (50,),       # 隐藏层偏置
    'W2': NDArray (50, 10),    # 隐藏 → 输出
    'b2': NDArray (10,),       # 输出层偏置
}
```

### `utils/trainer.py` — 训练循环

- 使用 **Mini-batch SGD**
- 每个 epoch 开始前随机打乱训练集（`np.random.permutation`）
- 每个 mini-batch 后记录 loss
- 每个 epoch 结束后在全量训练集和测试集上评估 accuracy
- 实时打印训练进度

### `config.py` — 超参数配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `input_size` | int | 784 | 输入维度（28×28 像素展平） |
| `hidden_size` | int | 50 | 隐藏层神经元数 |
| `output_size` | int | 10 | 输出类别（数字 0-9） |
| `learning_rate` | float | 0.1 | SGD 学习率 |
| `weight_init_std` | float | 0.01 | 权重初始化时高斯分布的标准差 |
| `batch_size` | int | 100 | Mini-batch 大小 |
| `epochs` | int | 17 | 训练轮数 |

## 快速开始

### 环境要求

- Python 3.8+
- NumPy

### 安装

```bash
pip install numpy
```

### 运行

```bash
python main.py
```

首次运行会自动下载 MNIST 数据集（约 11MB）并保存为 pickle 缓存，后续运行直接加载。

运行输出示例：

```
Downloading train-images-idx3-ubyte.gz ...
Done!
...
Creating pickle file ...\data\mnist.pkl...
Done!
Starting  Model Training ...
Batch     0 | train loss: 2.3895
Batch   100 | train loss: 2.3013
...
Epoch  1/17 | train acc: 0.1123, test acc: 0.1135
Epoch  2/17 | train acc: 0.3125, test acc: 0.3190
...
Finished Model Training!
```

## 训练流程

```
                    ┌─────────────────────────────────┐
                    │       加载 MNIST 数据集           │
                    │  normalize=True, one_hot=True    │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │      初始化 TwoLayerNet          │
                    │  权重 N(0, 0.01)，偏置 0         │
                    └──────────────┬──────────────────┘
                                   │
              ┌────────────────────▼───────────────────┐
              │          For epoch in 1..17             │
              │  ┌──────────────────────────────────┐   │
              │  │  随机打乱训练集                     │   │
              │  └──────────────┬───────────────────┘   │
              │                 │                        │
              │  ┌──────────────▼───────────────────┐   │
              │  │  For each mini-batch (size=100)   │   │
              │  │  ① 前向传播 → softmax 输出        │   │
              │  │  ② 交叉熵误差 → loss              │   │
              │  │  ③ 数值梯度 → ∇L(θ)               │   │
              │  │  ④ SGD 更新 → θ -= lr * ∇L        │   │
              │  │  ⑤ 记录 loss                      │   │
              │  └──────────────┬───────────────────┘   │
              │                 │                        │
              │  ┌──────────────▼───────────────────┐   │
              │  │  全量评估 train/test accuracy      │   │
              │  └──────────────────────────────────┘   │
              └─────────────────────────────────────────┘
```

## 自定义配置

修改 `config.py` 即可调整训练参数：

```python
config = MNISTConfig()
config.hidden_size = 100      # 增加隐藏层神经元
config.learning_rate = 0.05   # 降低学习率
config.epochs = 30            # 增加训练轮数
config.batch_size = 50        # 减小批量
```

## 注意事项

### 数值微分的性能

当前使用**数值微分**（中心差分法）计算梯度，每个参数需要 2 次前向传播：

- 网络总参数：≈ 39,760 个
- 每个 mini-batch 前向传播次数：≈ 79,520 次
- 1 个 epoch 前向传播次数：≈ 4,770 万次

这导致训练速度极慢（每个 epoch 可能需数小时），仅用于**教学和梯度验证**。如需实际训练，应将 `numerical_gradient` 替换为**反向传播（Backpropagation）**，可将每个 epoch 缩短至秒级。

### 网络下载

若所在网络无法访问 MNIST 官方源（如国内环境），可手动下载以下 4 个文件放入 `data/` 目录，程序会自动跳过下载步骤：

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

下载地址：http://yann.lecun.com/exdb/mnist/

