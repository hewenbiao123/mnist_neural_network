"""
导入 mnist 数据集
1. 下载 MNIST 数据集
2. 读取二进制数据, 将数据转化为numpy数组
3. 数据持久化保存在本地
"""
import urllib.request   # 网络请求库 
import gzip             # 用于读取.gz压缩文件
import pickle           # 将数据持久化保存到磁盘
import os.path          # 用于处理文件路径
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias        # 用于类型别名

class MNISTMeta:
    """MNIST 数据集元信息"""
    NAME: str = "MNIST"
    URL: str = "https://ossci-datasets.s3.amazonaws.com/mnist/"# 下载地址
    TRAIN_IMAGES_NAME: str = "train-images-idx3-ubyte.gz" # 训练图片
    TRAIN_LABELS_NAME: str = "train-labels-idx1-ubyte.gz" # 训练标签
    TEST_IMAGES_NAME: str = "t10k-images-idx3-ubyte.gz"   # 测试图片
    TEST_LABELS_NAME: str = "t10k-labels-idx1-ubyte.gz"   # 测试标签
    TOTAL_FILE_NAME: list[str] = [TRAIN_IMAGES_NAME, TRAIN_LABELS_NAME, TEST_IMAGES_NAME, TEST_LABELS_NAME]
    TRAIN_SAMPLES: int = 60000 # 训练样本数
    TEST_SAMPLES: int = 10000  # 测试样本数
    IMAGE_SIZE: int = 28 * 28  # 一张图片像素点个数
    IMAGE_DIM: tuple[int, int, int] = (1, 28, 28) # 图像维度信息（灰度，行数，列数）
    IMAGE_OFFSET: int = 16     # 图片IDX文件偏移量
    LABEL_OFFSET: int = 8      # 标签IDX文件偏移量
    MAX_PIXEL_VALUE: int = 255 # 像素点最大数字
    MAX_LABEL_VALUE: int = 9   # 标签最大数字



class MNISTLoader:

    SAVE_DIR: str = os.path.dirname(__file__)
    PICKLE_SAVE_PATH = os.path.join(SAVE_DIR, "mnist.pkl")

    @staticmethod
    def load(normalize: bool=True, flatten: bool=True, one_hot_label=False) -> tuple[
            tuple[NDArray[np.float32], NDArray[np.float32]], 
            tuple[NDArray[np.float32], NDArray[np.float32]]
    ]:
        if not os.path.exists(MNISTLoader.PICKLE_SAVE_PATH):
            MNISTLoader.init()
        dataset: dict[str, NDArray[np.float32]] = MNISTLoader.load_pickle()
        if normalize:
            for data_name in ('train_images', 'test_images'):
                dataset[data_name] /= np.float32(MNISTMeta.MAX_PIXEL_VALUE)
        if one_hot_label:
            for data_name in ('train_labels', 'test_labels'):
                T: NDArray[np.float32] = np.zeros((dataset[data_name].size, MNISTMeta.MAX_LABEL_VALUE + 1), dtype=np.float32)
                for idx, label_number in enumerate(dataset[data_name]):
                    T[idx][int(label_number)] = 1.0
                dataset[data_name] = T
        if not flatten:
            for data_name in ('train_images', 'test_images'):
                dataset[data_name] = dataset[data_name].reshape(-1, *MNISTMeta.IMAGE_DIM)
        
        train_pair = (dataset['train_images'], dataset['train_labels'])
        test_pair = (dataset['test_images'], dataset['test_labels'])
        return (train_pair, test_pair)
        

    # 私有方法
    @staticmethod
    def init() -> None:
        MNISTLoader.download()
        MNISTLoader.save_pickle(MNISTLoader.convert_numpy())

    @staticmethod
    def download() -> None:
        for FILE_NAME in MNISTMeta.TOTAL_FILE_NAME:
            save_path: str = os.path.join(MNISTLoader.SAVE_DIR, FILE_NAME)
            if os.path.exists(save_path):
                continue
            print(f"Downloading {FILE_NAME} ...")
            # 向网页发起检索请求
            urllib.request.urlretrieve(MNISTMeta.URL + FILE_NAME, save_path)
            print("Done!")

    @staticmethod
    def convert_numpy() -> dict[str, NDArray[np.uint8]]:
        dataset: dict[str, NDArray[np.uint8]] = {
            'train_images' : MNISTLoader._load_image(MNISTMeta.TRAIN_IMAGES_NAME),
            'train_labels' : MNISTLoader._load_label(MNISTMeta.TRAIN_LABELS_NAME),
            'test_images' : MNISTLoader._load_image(MNISTMeta.TEST_IMAGES_NAME),
            'test_labels' : MNISTLoader._load_label(MNISTMeta.TEST_LABELS_NAME)
        }
        return dataset 

    @staticmethod
    def save_pickle(dataset: dict[str, NDArray[np.uint8]]) ->None:
        print(f"Creating pickle file {MNISTLoader.PICKLE_SAVE_PATH}...")
        with open(MNISTLoader.PICKLE_SAVE_PATH, 'wb') as writer:
            pickle.dump(dataset, writer, -1)
        print("Done!")

    @staticmethod
    def load_pickle() -> dict[str, NDArray[np.float32]]:
        with open(MNISTLoader.PICKLE_SAVE_PATH, 'rb') as reader:
            dataset: dict[str, NDArray[np.uint8]] = pickle.load(reader)
        return {key: value.astype(np.float32) for key, value in dataset.items()}

    @staticmethod
    def _load_image(file_name: str) -> NDArray[np.uint8]:
        file_path: str = os.path.join(MNISTLoader.SAVE_DIR, file_name)
        print(f"Converting {file_name} to Numpy Array ...")
        with gzip.open(file_path, 'rb') as reader:
            # 一个像素点的值为 0-255
            pixels: NDArray[np.uint8] = np.frombuffer(reader.read(), np.uint8, offset=MNISTMeta.IMAGE_OFFSET)
        images: NDArray[np.uint8] = pixels.reshape(-1, MNISTMeta.IMAGE_SIZE)
        print("Done!")
        return images

    @staticmethod
    def _load_label(file_name: str) -> NDArray[np.uint8]:
        file_path: str = os.path.join(MNISTLoader.SAVE_DIR, file_name)
        print(f"Converting {file_name} to Numpy Array ...")
        with gzip.open(file_path, 'rb') as reader:
            # 一个标签的值为 0-9
            labels: NDArray[np.uint8] = np.frombuffer(reader.read(), np.uint8, offset=MNISTMeta.LABEL_OFFSET)
        print("Done!")
        return labels
    
    

        

    

    
    