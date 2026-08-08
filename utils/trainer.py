from data.mnist_loader import *
from config import MNISTConfig
from models.two_layer_net import TwoLayerNet
import numpy as np
from numpy.typing import NDArray

def train() -> None:
    (train_images, train_labels), (test_images, test_labels) = MNISTLoader.load(normalize=True, flatten=True, one_hot_label=True)
    config: MNISTConfig = MNISTConfig() # 初始化超参数列表
    network: TwoLayerNet = TwoLayerNet(config.input_size, config.hidden_size, config.output_size, config.weight_init_std) # 初始化神经网络

    # 数值记录表
    train_loss_list: list[float] = []   # 每个小批量的损失
    train_accuracy_list: list[float] = [] # 每个epoch后的训练精度
    test_accuracy_list: list[float] = []  # 每个 epoch 后的测试精度

    print("Starting  Model Training ...")

    for epoch in range(config.epochs):
        # 每个epoch开始前，随机打乱训练集
        permutation_idx: NDArray[np.long] = np.random.permutation(MNISTMeta.TRAIN_SAMPLES)
        train_images_shuffled: NDArray[np.float32] = train_images[permutation_idx]
        train_labels_shuffled: NDArray[np.float32] = train_labels[permutation_idx]
        for idx in range(0, MNISTMeta.TRAIN_SAMPLES, config.batch_size):
            train_images_batch: NDArray[np.float32] = train_images_shuffled[idx:idx+config.batch_size]
            train_labels_batch: NDArray[np.float32] = train_labels_shuffled[idx:idx+config.batch_size]
            network.update(train_images_batch, train_labels_batch, config.learning_rate)
            # 记录每个 Mini-batch 的损失函数
            train_loss: float = network.loss(train_images_batch, train_labels_batch)
            train_loss_list.append(train_loss)
            print(f"Batch {idx:5d} | train loss: {train_loss:.4f}")
        # 每个epoch结束后，在全部训练集和测试集上评估精度
        train_accuracy: float = network.accuracy(train_images, train_labels)
        test_accuracy: float = network.accuracy(test_images, test_labels)
        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)
        print(f"Epoch {epoch+1:2d}/{config.epochs} | train acc: {train_accuracy:.4f}, test acc: {test_accuracy:.4f}")

    print("Finished Model Training!")
    
