import numpy as np 
from numpy.typing import NDArray

class Function:
    @staticmethod
    def sigmoid(x: NDArray[np.float32]) -> NDArray[np.float32]:
        one: np.float32 = np.float32(1)
        return one / (one + np.exp(-x))

    @staticmethod
    def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
        return np.maximum(0, x)

    @staticmethod
    def softmax(x: NDArray[np.float32]) -> NDArray[np.float32]:
        # axis=-1，使得总是对
        exp_x: NDArray[np.float32] = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def cross_entropy_error(y: NDArray[np.float32], t: NDArray[np.float32]) -> float:
        delta: np.float32 = np.float32(1e-7)
        batch_size: int = y.shape[0]
        loss: np.float32
        # 监督数据为 one-hot
        if t.size == y.size:
            loss = -np.sum(t * np.log(y + delta)) / batch_size # type: ignore
        # 监督数据不为 one-hot，则监督数组为一维数组
        else:
            loss = -np.sum(np.log(y[np.arange(batch_size), t.astype(int)] + delta)) / batch_size
        return float(loss)
    


        

