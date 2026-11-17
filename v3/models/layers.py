from utils.function import Function
import numpy as np
from numpy.typing import NDArray


"""
以层为模块定义神经网络层级
"""

class Affine:
    def __init__(self, W: NDArray[np.float32], B: NDArray[np.float32]) -> None:
        self.W: NDArray[np.float32] = W
        self.B: NDArray[np.float32] = B
        self.X: NDArray[np.float32]
        self.dW: NDArray[np.float32]
        self.dB: NDArray[np.float32]

    def forward(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        self.X = X
        return np.dot(self.X, self.W) + self.B

    def backward(self, dZ: NDArray[np.float32]) -> NDArray[np.float32]:
        self.dW = np.dot(self.X.T, dZ)
        self.dB = np.sum(dZ, axis=0)
        return np.dot(dZ, self.W.T)

class Sigmoid:
    def __init__(self) -> None:
        # 不保存 Z，而是保存 sigmoid(Z)，因为 sigmoid(Z) = sigmoid(Z) * (1 - sigmoid(Z))
        self.sigmoid_Z = NDArray[np.float32]

    def forward(self, Z: NDArray[np.float32]) -> NDArray[np.float32]:
        self.sigmoid_Z = Function.sigmoid(Z)
        return self.sigmoid_Z

    def backward(self, dH: NDArray[np.float32]) -> NDArray[np.float32]:
        return dH * self.sigmoid_Z * (np.float32(1) - self.sigmoid_Z) # type: ignore

class Relu:
    def __init__(self) -> None:
        # relu(x) = x > 0，考虑保存0位置
        self.zero_idxs: NDArray[np.bool]

    def forward(self, Z: NDArray[np.float32]) -> NDArray[np.float32]:
        self.zero_idxs = (Z <= 0)
        out: NDArray[np.float32] = Z.copy() # 不影响原矩阵
        out[self.zero_idxs] = 0
        return out

    def backward(self, dH: NDArray[np.float32]) -> NDArray[np.float32]:
        # if x > 0, relu'(x) = 1 else relu'(x) = 0
        dZ: NDArray[np.float32] = dH.copy()
        dZ[self.zero_idxs] = 0
        return dZ

class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.hat_Y: NDArray[np.float32]
        self.Y: NDArray[np.float32]

    # 前向传播：loss(softmax(Z), Y)
    def forward(self, Z: NDArray[np.float32], Y: NDArray[np.float32]) -> float:
        self.Y = Y
        self.hat_Y = Function.softmax(Z)
        return Function.cross_entropy_error(self.hat_Y, self.Y)

    def backward(self) -> NDArray[np.float32]:
        batch_size: int = self.Y.shape[0]
        dZ: NDArray[np.float32]
        # 如果是one-hot编码
        if self.hat_Y.size == self.Y.size:
            dZ = (self.hat_Y - self.Y) / np.float32(batch_size)
        else :
            dZ = self.hat_Y.copy()
            dZ[np.arange(batch_size), self.Y.astype(int)] -= np.float32(1) # dZ[i][Y[i]] -= 1
            dZ = dZ / np.float32(batch_size)
        return dZ



