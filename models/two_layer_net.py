import numpy as np 
from numpy.typing import NDArray
from utils.function import Function



class TwoLayerNet:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01) -> None:
        # 初始化权重
        self.params: dict[str, NDArray[np.float32]] = {}
        self.params['W1'] = (weight_init_std * np.random.randn(input_size, hidden_size)).astype(np.float32) 
        self.params['W2'] = (weight_init_std * np.random.randn(hidden_size, output_size)).astype(np.float32)
        self.params['b1'] = np.zeros(hidden_size, dtype=np.float32)
        self.params['b2'] = np.zeros(output_size, dtype=np.float32)

    # 前向传播
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        # z1 = sigmoid(X * W1 + b1)
        Z1: NDArray[np.float32] = Function.sigmoid(np.dot(X, self.params['W1']) + self.params['b1'])
        # y = softmax(Z1 * W2 + b2)
        y : NDArray[np.float32] = Function.softmax(np.dot(Z1, self.params['W2']) + self.params['b2'])
        return y

    # 准确率计算
    def accuracy(self, x: NDArray[np.float32], t: NDArray[np.float32]) -> float:
        y: NDArray[np.float32] = self.predict(x)
        return np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)) / float(x.shape[0])

    # 损失计算
    def loss(self, x: NDArray[np.float32], t: NDArray[np.float32]) -> float:
        return Function.cross_entropy_error(self.predict(x), t)

    # arg是权重字典的某个键值对的值，是可变参数，修改可以直接作用到原字典
    def _numerical_gradient(self, x: NDArray[np.float32], arg: NDArray[np.float32], t: NDArray[np.float32]) -> NDArray[np.float32]:
        h: float = 1e-4
        grad: NDArray[np.float32] = np.zeros_like(arg)
        # 遍历 arg 每一个元素，利用 np.nditer(arg, flag=['multi_index']) 返回其迭代器
        # 该方法不产生任何副本，对比 np.flatten() 摊开为一维数组更加高效
        # 也可以使用 for idx, val in np.ndenumerate(arg) 进行迭代，idx 是元组下标
        # 或者使用 for idx in np.ndindex(arg.shape) 只迭代索引，避免迭代时修改值的影响
        for idx in np.ndindex(arg.shape):
            # 利用中心差分公式近似求解偏导数：[f(arg + h) - f(arg - h)] / 2h
            pre_value: np.float32 = np.float32(arg[idx]) # arg[idx] 返回的是零维数组
            arg[idx] = pre_value + h
            f1: float = self.loss(x, t)
            arg[idx] = pre_value - h
            f2: float = self.loss(x, t)
            grad[idx] = (f1 - f2) / (2 * h)
            arg[idx] = pre_value # 还原
        return grad

    # 低效数值微分求偏导数
    def numerical_gradient(self, x: NDArray[np.float32], t: NDArray[np.float32]) -> dict[str, NDArray[np.float32]]:
        grads: dict[str, NDArray[np.float32]] = {}
        for key, value in self.params.items():
            grads[key] = self._numerical_gradient(x, value, t)
        return grads

    # 参数更新
    def update(self, x: NDArray[np.float32], t: NDArray[np.float32], learning_rate: float) -> None:
        grads: dict[str, NDArray[np.float32]] = self.numerical_gradient(x, t)
        for key, value in grads.items():
            self.params[key] -= learning_rate * value
        