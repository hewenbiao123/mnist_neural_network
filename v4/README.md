# MNIST Handwritten Digit Recognition — Two-Layer Neural Network (PyTorch)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/torchvision-0.x-EE4C2C?logo=pytorch&logoColor=white" alt="torchvision">
</p>

<p align="center">
  <b>The same architecture, reimplemented with PyTorch instead of pure NumPy</b>
</p>

---

## About This Project

This directory is the **v4** version inside the merged [mnist_neural_network](https://github.com/hewenbiao123/mnist_neural_network) repository (originally the standalone `mnist_neural_network_v4` repo, with its full commit history preserved). It is the PyTorch port of [v3](https://github.com/hewenbiao123/mnist_neural_network/tree/main/v3). The **network architecture and hyperparameters are identical**; the only change is the underlying implementation, which switches from "pure NumPy from scratch" to "the PyTorch framework".

| Version | Implementation | Gradient computation | Purpose |
|---------|----------------|----------------------|---------|
| [v2](https://github.com/hewenbiao123/mnist_neural_network/tree/main/v2) | Pure NumPy, procedural | Numerical differentiation | Understand the concept of gradients |
| [v3](https://github.com/hewenbiao123/mnist_neural_network/tree/main/v3) | Pure NumPy, modular layers | Manual backpropagation | Master the principle of backpropagation |
| **v4** | **PyTorch framework** | **autograd automatic differentiation** | **Master the framework API while understanding what happens underneath** |

**Why v4?** After manually implementing `forward()` / `backward()` for `Affine`, `ReLU`, and `SoftmaxWithLoss` in v3, reimplementing the same network in PyTorch makes the gap between "from scratch" and "framework" concrete: `nn.Linear` maps to `Affine`, `nn.ReLU` to `Relu`, `nn.CrossEntropyLoss` to `SoftmaxWithLoss`, and `loss.backward()` to calling `backward()` layer by layer in v3.

## Network Architecture

Identical to v3: the input $X \in \mathbb{R}^{N \times 784}$ flows through a fully connected layer, a ReLU activation, and another fully connected layer, producing 10-dimensional logits $Z_2$. The shape and parameter count of each layer are listed below.

| Layer | Shape | Parameters | v3 | v4 |
|-------|-------|------------|----|----|
| Input | `(batch, 784)` | — | Manual flatten | `x.view(-1, 784)` |
| Affine1 ($W_1$ + $B_1$) | `(784, 50)` + `(50,)` | 39,250 | Custom `Affine` | `nn.Linear(784, 50)` |
| ReLU | `(batch, 50)` | — | Custom `Relu` | `nn.ReLU()` |
| Affine2 ($W_2$ + $B_2$) | `(50, 10)` + `(10,)` | 510 | Custom `Affine` | `nn.Linear(50, 10)` |
| Output | `(batch, 10)` | — | Custom `SoftmaxWithLoss` | `nn.CrossEntropyLoss` |
| **Total** | | **39,760** | | |

## Mathematical Formulas

### Forward Propagation

$$
Z_1 = X W_1^\top + B_1
$$

$$
A_1 = \mathrm{ReLU}(Z_1) = \max(0, Z_1)
$$

$$
Z_2 = A_1 W_2^\top + B_2
$$

$$
\hat{Y} = \mathrm{softmax}(Z_2)
$$

Here $W_1 \in \mathbb{R}^{50 \times 784}$ and $W_2 \in \mathbb{R}^{10 \times 50}$, which are exactly the weight shapes of `nn.Linear` (`(out_features, in_features)`). The model's `forward()` returns $Z_2$ (logits); the softmax is applied inside the loss function.

### Loss Function: Cross Entropy

$$
\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

$$
L = -\frac{1}{N}\sum_{n=1}^{N}\sum_{k=1}^{K} t_{nk} \ln \hat{y}_{nk}
$$

where $N$ is the batch size, $K = 10$ is the number of classes, and $t_{nk}$ is the one-hot ground-truth label of the $n$-th sample for class $k$. v4 uses `nn.CrossEntropyLoss`, which performs the softmax and the cross-entropy above internally.

### Parameter Update: SGD

$$
\theta \leftarrow \theta - \eta \nabla_\theta L
$$

where $\eta$ is the learning rate. v4 performs this update via `optim.SGD` and `optimizer.step()`.

## v3 vs v4 Core Comparison

### Overview

| Dimension | v3 (NumPy from scratch) | v4 (PyTorch framework) |
|-----------|--------------------------|------------------------|
| Data loading | Manual IDX download, gzip decompression, NumPy parsing, pickle caching | `torchvision.datasets.MNIST` + `DataLoader` |
| Layer abstraction | Custom `Affine` / `Relu` / `SoftmaxWithLoss` classes | `nn.Linear` / `nn.ReLU` / `nn.CrossEntropyLoss` |
| Forward pass | `predict()` iterates a layer list calling `forward()` | `model(x)`, defined by `forward()` |
| Backward pass | Manual chain rule, calling `backward()` layer by layer | `loss.backward()`, autograd |
| Parameter update | Manual $\theta \leftarrow \theta - \eta \nabla L$ | `optimizer.step()` (`optim.SGD`) |
| Gradient clearing | Gradients overwritten on each recomputation | Explicit `optimizer.zero_grad()` |
| Label format | Manual one-hot encoding | Class indices (accepted directly by `CrossEntropyLoss`) |
| Acceleration | CPU only (NumPy) | CPU (device fixed in `main.py`) |

### Code Comparison

**Network definition:**

```python
# v3: custom layer list
self.layers = [Affine(W1, B1), Relu(), Affine(W2, B2)]
self.lastLayer = SoftmaxWithLoss()

# v4: nn.Module
self.hidden_affine = nn.Linear(784, 50)
self.hidden_relu = nn.ReLU()
self.output_affine = nn.Linear(50, 10)
```

**One training step:**

```python
# v3
network.update(train_images_batch, train_labels_batch, config.learning_rate)
# internally: forward + backward + parameter update

# v4
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

**Accuracy computation:**

```python
# v3
def accuracy(self, X, Y):
    hat_Y = self.predict(X)                       # softmax probabilities
    return np.mean(np.argmax(hat_Y, 1) == Y)

# v4
def evaluate(model, loader, device):
    outputs = model(images)                       # logits
    _, predicted = torch.max(outputs, 1)
    return correct / total
```

### Key Differences

1. **Where softmax lives**
   - In v3, `predict()` returns softmax probabilities and the loss is computed by `SoftmaxWithLoss.forward()`.
   - In v4, `forward()` returns logits (raw scores); `nn.CrossEntropyLoss` applies softmax internally, which is numerically more stable. Call `torch.softmax` when you need probabilities during inference.

2. **Backpropagation is handled by autograd**
   - v3 requires manually deriving and implementing `backward()` for every layer.
   - v4 only needs a single `loss.backward()`; PyTorch computes gradients for all parameters via the computation graph.

3. **No one-hot labels needed**
   - v3 converts labels to `(N, 10)` one-hot by default (`one_hot_label=True`).
   - v4 uses class indices `(N,)`, handled internally by `nn.CrossEntropyLoss`.

## Project Structure

All paths below are relative to `v4/`.

- `data/mnist_loader.py` — Dataset: torchvision download + DataLoader
- `models/two_layer_net.py` — Model: the network defined with `nn.Module`
- `train.py` — Trainer: training loop and evaluation
- `config.py` — Centralized hyperparameters
- `main.py` — Program entry
- GUIDE.md — Step-by-step implementation guide for beginners
- README.md — This document

> Compared with v3, the custom `models/layers.py` and `utils/function.py` are no longer needed; these capabilities are provided by PyTorch built-ins.

## Module Details

### `data/mnist_loader.py` — Data Loader

| Function | Description |
|----------|-------------|
| `get_mnist_dataloaders(data_dir, batch_size)` | Returns `(train_loader, test_loader)` |

- Uses `torchvision.datasets.MNIST`, which downloads and caches the dataset to `./data` on the first run.
- Pixel values are normalized from `[0, 255]` to `[0, 1]`, equivalent to `normalize=True` in v3.
- `DataLoader(shuffle=True)` shuffles the training set every epoch, equivalent to `np.random.permutation` in v3.

### `models/two_layer_net.py` — Network Definition

| Member | Description |
|--------|-------------|
| `hidden_affine` | `nn.Linear(784, 50)`, corresponding to `Affine(W1, B1)` in v3 |
| `hidden_relu` | `nn.ReLU()`, corresponding to `Relu` in v3 |
| `output_affine` | `nn.Linear(50, 10)`, corresponding to `Affine(W2, B2)` in v3 |
| `forward(x)` | Flattens the input, then applies fully connected, ReLU, and fully connected layers, returning logits |

### `train.py` — Training and Evaluation

| Function | Description |
|----------|-------------|
| `train(...)` | Mini-batch SGD training loop, printing loss, train accuracy and test accuracy per epoch |
| `evaluate(...)` | Computes accuracy, corresponding to `network.accuracy()` in v3 |

### `config.py` — Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_size` | int | 784 | Input dimension (flattened 28×28) |
| `hidden_size` | int | 50 | Number of hidden neurons |
| `output_size` | int | 10 | Number of classes (digits 0–9) |
| `learning_rate` | float | 0.1 | SGD learning rate |
| `batch_size` | int | 100 | Mini-batch size |
| `epochs` | int | 17 | Number of training epochs |
| `data_dir` | str | `./data` | Dataset cache directory |
| `random_seed` | int | 42 | Random seed for reproducibility |

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.x (CPU)
- torchvision 0.x

### Installation

```bash
# Clone the repository
git clone https://github.com/hewenbiao123/mnist_neural_network.git
cd mnist_neural_network/v4

# Install dependencies (CPU example)
pip install torch torchvision
```

> This project runs on CPU only: the device is fixed to `cpu` in `main.py`, so no GPU is required.

### Run

```bash
python main.py
```

The MNIST dataset (about 11 MB) is downloaded to `./data` on the first run and loaded directly afterwards.

Example output:

```
Using device: cpu
Epoch  1/17 | loss 0.7325 | train acc 0.9043 | test acc 0.9071
Epoch  2/17 | loss 0.3008 | train acc 0.9221 | test acc 0.9241
...
Epoch 17/17 | loss 0.0752 | train acc 0.9800 | test acc 0.9713
```

**Expected result:** with the default hyperparameters, the training accuracy reaches about **98%** and the test accuracy about **97%** after 17 epochs, matching v3.

## Training Flow

1. **Load data** — `torchvision` downloads MNIST automatically; pixels are normalized to [0,1]; `DataLoader` chunks it into batches of 100.
2. **Initialize the model** — `TwoLayerNet` (784 → 50 → 10 with ReLU).
3. **Train for 17 epochs**, each epoch:
   - Shuffles the training set (`shuffle=True`).
   - Iterates over mini-batches, executing these steps:
     1. `optimizer.zero_grad()` clears gradients.
     2. `model(images)` forward pass.
     3. `criterion(outputs, labels)` computes the loss.
     4. `loss.backward()` backward pass.
     5. `optimizer.step()` updates parameters.
   - Evaluates the training-set and test-set accuracy.
4. **Done** — the final test accuracy is printed.

## Custom Configuration

Modify `config.py`, exactly as in v3:

```python
config = MNISTConfig()
config.hidden_size = 100      # more hidden neurons
config.learning_rate = 0.05   # lower learning rate
config.epochs = 30            # more epochs
config.batch_size = 50        # smaller batches
```

## FAQ

### Dataset download fails or is slow

`torchvision.datasets.MNIST` downloads from the official source by default, which may be slow in some regions. You can manually download the following four raw files into `./data/MNIST/raw/` (keep the exact filenames):

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`
- `t10k-images-idx3-ubyte.gz`
- `t10k-labels-idx1-ubyte.gz`

Download: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

The program skips the download step once these files exist.

### Accuracy is not ideal

As in v3, try:

1. Increase `epochs` (30–50).
2. Adjust `learning_rate` (0.5, 0.01, etc.).
3. Enlarge `hidden_size` (100, 200).

## References

- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [torchvision.datasets.MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [Deep Learning from Scratch](https://github.com/oreilly-japan/deep-learning-from-scratch) — the main reference of v3
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

---

<p align="center">
  <sub>If you find this project helpful, please consider giving it a ⭐ Star!</sub>
</p>
