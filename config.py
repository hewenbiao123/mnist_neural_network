class MNISTConfig:
    """MNIST 训练超参数设计"""
    def __init__(self) -> None:
        self.input_size: int = 784
        self.hidden_size: int = 50
        self.output_size: int = 10
        self.learning_rate: float = 0.1
        self.weight_init_std: float = 0.01
        self.batch_size: int = 100
        self.epochs = 17 # 所有训练集学习一次记为一个epoch
        
