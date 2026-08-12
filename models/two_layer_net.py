from models.layers import *
import numpy as np
from numpy.typing import NDArray
from typing import Any

"""
定义一个二层神经网络
"""
class TwoLayerNet:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01) -> None:
        # 初始化权重与偏置
        self.parameters: dict[str, NDArray[np.float32]] = {}
        self.parameters['W1'] = (weight_init_std * np.random.randn(input_size, hidden_size)).astype(np.float32)
        self.parameters['W2'] = (weight_init_std * np.random.randn(hidden_size, output_size)).astype(np.float32)
        self.parameters['B1'] = np.zeros(hidden_size, dtype=np.float32)
        self.parameters['B2'] = np.zeros(output_size, dtype=np.float32)
        # 初始化层级
        self.layers: list[Any] = []
        self.layers.append(Affine(self.parameters['W1'], self.parameters['B1']))
        self.layers.append(Relu())
        # self.layers.append(Sigmoid())
        self.layers.append(Affine(self.parameters['W2'], self.parameters['B2']))
        self.lastLayer = SoftmaxWithLoss()

    # 预测
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        for idx, layer in enumerate(self.layers):
            X = layer.forward(X)
        return X

    # 准确值计算
    def accuracy(self, X: NDArray[np.float32], Y: NDArray[np.float32]) -> float:
        hat_Y: NDArray[np.float32] = self.predict(X)
        # 如果 Y 是 one-hot 编码，转化为原始编码
        if Y.ndim == 2:
            Y = np.argmax(Y, axis=1)
        return np.sum(np.argmax(hat_Y, axis = 1) == Y) / float(X.shape[0])

    # 前向传播（计算一次损失值）
    def forward(self, X: NDArray[np.float32], Y: NDArray[np.float32]) -> float:
        return self.lastLayer.forward(self.predict(X), Y)

    # 反向传播（计算并且返回梯度）
    def backward(self) -> dict[str, NDArray[np.float32]]:
        dout: NDArray[np.float32] = self.lastLayer.backward() # 每一层返回提供给上一层的值
        for idx, layer in enumerate(reversed(self.layers)):
            dout = layer.backward(dout)
        grads: dict[str, NDArray[np.float32]] = {}
        grads['W1'] = self.layers[0].dW
        grads['B1'] = self.layers[0].dB
        grads['W2'] = self.layers[2].dW
        grads['B2'] = self.layers[2].dB
        return grads

    # 梯度更新
    def update(self, X: NDArray[np.float32], Y: NDArray[np.float32], learning_rate: float) -> None:
        self.forward(X, Y)
        grads: dict[str, NDArray[np.float32]] = self.backward()
        for key, value in grads.items():
            self.parameters[key] -= learning_rate * value


        
        
