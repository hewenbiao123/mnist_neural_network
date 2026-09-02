""" Define parameters related to neural networks """

class MNISTConfig:
    def __init__(self) -> None:
        # Define the number of neurons in each layer
        self.input_size: int = 784
        self.hidden_size: int = 50
        self.output_size: int = 10

        # Define hyperparameters
        self.learning_rate: float = 0.1
        self.batch_size: int = 100
        self.epochs:int = 17

        # Define other parameters
        self.data_dir: str = "./data"
        self.random_seed = 42
