import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import ssl
from options import args_parser
import random
import numpy as np

# 用于改变默认的HTTPS上下文以忽略对SSL证书的验证
ssl._create_default_https_context = ssl._create_unverified_context

"""
IID数据加载类：加载MNIST、Fashion-MNIST和CIFAR10数据集，并对数据进行预处理
"""

def generate_random_list(n):
    """生成一个长度为n的列表，且列表个元素之和为1"""
    np.random.seed(42)
    # 生成 n-1 个在 [0, 1) 范围内的随机数
    rand_nums = np.random.rand(n-1)
    # 对随机数进行排序
    rand_nums.sort()
    # 计算列表中相邻元素的差值，得到 n 个元素
    result = np.diff(rand_nums, prepend=0, append=1)
    return result

class DatasetLoad_iid(object):
    def __init__(self, args):
        self.num_clients = args.num_clients
        self.batch_size = args.batch_size
        # 根据名称选择指定的数据集
        if args.dataset_name == "MNIST":
            self.mnist_load_dataset()
        elif args.dataset_name == "CIFAR10":
            self.cifar10_load_dataset()
        elif args.dataset_name == "Fashion-MNIST":
            self.fashion_load_dataset()
    
    def mnist_load_dataset(self):
        """下载和转换 MNIST 数据集（训练集和测试集）"""
        # 设置随机种子
        torch.manual_seed(42)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = MNIST(root="./datasets/iid_datasets/minist/trainset", train=True,
                         download=True, transform=transform)
        testset = MNIST(root="./datasets/iid_datasets/minist/testset", train=False,
                        download=True, transform=transform)
        # 计算每个客户端应分配的样本数量
        total_samples = len(trainset)
        # 将mnist数据集中的5/6分给每个客户端
        samples_per_client = total_samples * 5 // (6 * self.num_clients)  # 将总样本数的5/6分给客户端
        new_data_number = total_samples - samples_per_client * self.num_clients
        # 创建 SubsetRandomSampler，用于划分训练集
        datasets = []
        # 根据客户端数量生成随机比例
        clients_data_scale = generate_random_list(self.num_clients)
        sample_client_per_data = new_data_number * clients_data_scale
        for i in range(self.num_clients):
            # 前一半数据均等分给每个客户端
            indices_half_first = list(range(i * samples_per_client, (i + 1) * samples_per_client))
            # 后一半数据随机分给每个客户端
            indices_half_end = list(set(range(total_samples)) - set(indices_half_first))
            indices_half_end = random.sample(indices_half_end,int(sample_client_per_data[i]))
            indices = indices_half_first + indices_half_end
            # 创建 Subset 对象
            ds = Subset(trainset, indices)
            datasets.append(ds)
        # 初始化列表以存储每个客户端的训练和验证数据加载器
        trainloaders = []
        valloaders = []
        # 遍历每个客户端的数据集
        for ds in datasets:
            # 将验证集的数据一并加入训练集
            trainloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                     generator=torch.Generator().manual_seed(42))
            trainloaders.append(trainloader)
        # 对每个客户端分配测试集
        for _ in range(self.num_clients):
            valloaders.append(DataLoader(testset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(42)))
        return trainloaders, valloaders
    
    def cifar10_load_dataset(self):
        """下载和转换 Cifar-10 数据集（训练集和测试集）"""
        # 设置随机种子
        torch.manual_seed(42)
        # 数据转换
        # transform = transforms.Compose(
        #    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        # )
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = CIFAR10(root="./datasets/iid_datasets/cifar10/trainset", train=True,
                           download=True, transform=transform_train)
        testset = CIFAR10(root="./datasets/iid_datasets/cifar10/testset", train=False,
                          download=True, transform=transform_test)
        # 计算每个客户端应分配的样本数量
        total_samples = len(trainset)
        # 将mnist数据集中的5/6分给每个客户端
        samples_per_client = total_samples * 4 // (5 * self.num_clients)  # 将总样本数的5/6分给客户端
        new_data_number = total_samples - samples_per_client * self.num_clients
        # 创建 SubsetRandomSampler，用于划分训练集
        datasets = []
        # 根据客户端数量生成随机比例
        clients_data_scale = generate_random_list(self.num_clients)
        sample_client_per_data = new_data_number * clients_data_scale
        for i in range(self.num_clients):
            # 前一半数据均等分给每个客户端
            indices_half_first = list(range(i * samples_per_client, (i + 1) * samples_per_client))
            # 后一半数据随机分给每个客户端
            indices_half_end = list(set(range(total_samples)) - set(indices_half_first))
            indices_half_end = random.sample(indices_half_end, int(sample_client_per_data[i]))
            indices = indices_half_first + indices_half_end
            # 创建 Subset 对象
            ds = Subset(trainset, indices)
            datasets.append(ds)
        # 初始化列表以存储每个客户端的训练和验证数据加载器
        trainloaders = []
        valloaders = []
        # 遍历每个客户端的数据集
        for ds in datasets:
            # 将验证集的数据一并加入训练集
            trainloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                     generator=torch.Generator().manual_seed(42))
            trainloaders.append(trainloader)
        # 对每个客户端分配测试集
        for _ in range(self.num_clients):
            valloaders.append(DataLoader(testset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(42)))
        return trainloaders, valloaders
        
    def fashion_load_dataset(self):
        """下载和转换 Fashion-MNIST 数据集（训练集和测试集）"""
        # 设置随机种子
        torch.manual_seed(42)
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        trainset = FashionMNIST(root="./datasets/iid_datasets/minist/trainset", train=True,
                                download=True, transform=transform_train)
        testset = FashionMNIST(root="./datasets/iid_datasets/minist/testset", train=False,
                               download=True, transform=transform_test)
        # 计算每个客户端应分配的样本数量
        total_samples = len(trainset)
        # 将mnist数据集中的5/6分给每个客户端
        samples_per_client = total_samples * 5 // (6 * self.num_clients)  # 将总样本数的5/6分给客户端
        new_data_number = total_samples - samples_per_client * self.num_clients
        # 创建 SubsetRandomSampler，用于划分训练集
        datasets = []
        # 根据客户端数量生成随机比例
        clients_data_scale = generate_random_list(self.num_clients)
        sample_client_per_data = new_data_number * clients_data_scale
        for i in range(self.num_clients):
            # 前一半数据均等分给每个客户端
            indices_half_first = list(range(i * samples_per_client, (i + 1) * samples_per_client))
            # 后一半数据随机分给每个客户端
            indices_half_end = list(set(range(total_samples)) - set(indices_half_first))
            indices_half_end = random.sample(indices_half_end, int(sample_client_per_data[i]))
            indices = indices_half_first + indices_half_end
            # 创建 Subset 对象
            ds = Subset(trainset, indices)
            datasets.append(ds)
        # 初始化列表以存储每个客户端的训练和验证数据加载器
        trainloaders = []
        valloaders = []
        # 遍历每个客户端的数据集
        for ds in datasets:
            # 将验证集的数据一并加入训练集
            trainloader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                     generator=torch.Generator().manual_seed(42))
            trainloaders.append(trainloader)
        # 对每个客户端分配测试集
        for _ in range(self.num_clients):
            valloaders.append(DataLoader(testset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(42)))
        return trainloaders, valloaders


def plot_data_distribution(trainloaders):
    for i, trainloader in enumerate(trainloaders):
        # 统计每个类别的样本数量
        class_counts = [0] * 10  # 数据集有10个类别
        for _, labels in trainloader:
            for label in labels:
                class_counts[label] += 1

        # 绘制直方图
        plt.bar(range(10), class_counts)
        plt.title(f"Client {i+1} Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.show()

if __name__ == "__main__":
   # 设置随机种子
   torch.manual_seed(42)
   args = args_parser()
   # 在cifar10_load_dataset函数的末尾调用这个函数
   dataset_loader = DatasetLoad_iid(args)
   trainloaders,_ = dataset_loader.mnist_load_dataset()
   for i, trainloader in enumerate(trainloaders):
       client_samples = len(trainloader.dataset)
       print(client_samples)
   plot_data_distribution(trainloaders)