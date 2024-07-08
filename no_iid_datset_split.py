from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10,MNIST,FashionMNIST
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import ssl
from torch.distributions.dirichlet import Dirichlet
from options import args_parser

# 用于改变默认的HTTPS上下文以忽略对SSL证书的验证
ssl._create_default_https_context = ssl._create_unverified_context


def dirichlet_split_noniid(train_labels, n_clients, alpha=0.5):
    """非独立同分布"""
    torch.manual_seed(42)
    n_classes = train_labels.max() + 1
    label_distribution = Dirichlet(torch.full((n_clients,), alpha)).sample((n_classes,))
    class_idcs = [torch.nonzero(train_labels == y).flatten()
                  for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        total_size = len(c)
        splits = (fracs * total_size).int()
        splits[-1] = total_size - splits[:-1].sum()
        idcs = torch.split(c, splits.tolist())
        for i, idx in enumerate(idcs):
            client_idcs[i] += [idcs[i]]
    client_idcs = [torch.cat(idcs) for idcs in client_idcs]
    return client_idcs


class DatasetLoad_noiid(object):
    def __init__(self, args):
        self.num_client = args.num_clients
        self.batch_size = args.batch_size
        # 根据名称选择指定的数据集
        if args.dataset_name == "MNIST":
            self.noidd_mnist_load_dataset()
        elif args.dataset_name == "CIFAR10":
            self.noidd_cifar10_load_dataset()
        elif args.dataset_name == "Fashion-MNIST":
            self.noiid_fashion_load_dataset()
    
    def noidd_cifar10_load_dataset(self):
        """对cifar10数据集进行处理"""
        # 设置随机种子
        torch.manual_seed(seed=42)
        # 下载和转换 CIFAR-10 数据集（训练集和测试集）
        # 数据转换
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
        # 创建 CIFAR-10 训练集，应用指定的数据转换
        trainset = CIFAR10("./datasets/noiid_datasets/cifar10/trainset", train=True,
                           download=True, transform=transform_train)
        # 创建 CIFAR-10 测试集，应用指定的数据转换
        testset = CIFAR10("./datasets/noiid_datasets/cifar10/testset", train=False,
                          download=True, transform=transform_test)
        # 将训练集分割为用于联邦学习的分区
        #train_labels = torch.tensor(trainset.targets)  # 从 CIFAR-10训练集中提取标签
        #train_labels = trainset.targets.clone().detach()
        train_labels = torch.tensor(trainset.targets).clone().detach()
        # 使用 dirichlet_split_noniid 函数获取客户端索引
        client_idcs = dirichlet_split_noniid(train_labels, self.num_client)
        # 为每个客户端创建 DataLoader
        trainloaders = []
        valloaders = []
        for idcs in client_idcs:
            train_subset = Subset(trainset, indices=idcs)
            trainloader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,drop_last=True,
                                     generator=torch.Generator().manual_seed(42))
            trainloaders.append(trainloader)
        for _ in range(self.num_client):
            valloaders.append(DataLoader(testset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                         generator=torch.Generator().manual_seed(42)))
        return trainloaders, valloaders
    
    def noidd_mnist_load_dataset(self):
        """对mnist数据集进行处理"""
        # 设置随机种子
        torch.manual_seed(seed=42)
        # 下载和转换 CIFAR-10 数据集（训练集和测试集）
        # 数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        # 创建MNIST训练集，应用指定的数据转换
        trainset = MNIST("./datasets/noiid_datasets/minist/trainset", train=True,
                         download=True, transform=transform)
        # 创建MNIST测试集，应用指定的数据转换
        testset = MNIST("./datasets/noiid_datasets/minist/testset", train=False,
                        download=True, transform=transform)
        # 将训练集分割为用于联邦学习的分区
        #train_labels = torch.tensor(trainset.targets)  # 从Mnist训练集中提取标签
        train_labels = trainset.targets.clone().detach()
        #test_labels = testset.targets.clone().detach()
        # 使用 dirichlet_split_noniid 函数获取客户端索引
        client_idcs_train = dirichlet_split_noniid(train_labels, self.num_client)
        #client_idcs_test = dirichlet_split_noniid(test_labels, self.num_client)
        # 为每个客户端创建 DataLoader
        trainloaders = []
        valloaders = []
        for idcs in client_idcs_train:
            train_subset = Subset(trainset, indices=idcs)
            trainloader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                     generator=torch.Generator().manual_seed(42))
            trainloaders.append(trainloader)
        #for _ in range(self.num_client):
        #for idcs in client_idcs_test:
        #    test_subset = Subset(testset, indices=idcs)
        #    valloaders.append(DataLoader(test_subset, batch_size=self.batch_size, shuffle=True))
        for _ in range(self.num_client):
            valloaders.append(DataLoader(testset, batch_size=self.batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(42)))
        return trainloaders, valloaders
    
    def noiid_fashion_load_dataset(self):
        """对Fashion-mnist数据集进行处理"""
        # 设置随机种子
        torch.manual_seed(seed=42)
        # 下载和转换 Fashion-MNIST 数据集（训练集和测试集）
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        #torch.manual_seed(42)
        #transform_train = transforms.Compose([
        #    transforms.RandomHorizontalFlip(),
        #    transforms.RandomGrayscale(),
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.5,), (0.5,))])
        #transform_test = transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.5,), (0.5,))])
        # 创建Fashion-MNIST训练集，应用指定的数据转换
        trainset = FashionMNIST("./datasets/noiid_datasets/fashion_mnist/trainset",
                                train=True, download=True,
                                transform=transform)
        # 创建Fashion-MNIST测试集，应用指定的数据转换
        testset = FashionMNIST("./datasets/noiid_datasets/fashion_mnist/testset",
                               train=False, download=True,
                               transform=transform)
        # 将训练集分割为用于联邦学习的分区
        #train_labels = torch.tensor(trainset.targets)  # 从Mnist训练集中提取标签
        train_labels = trainset.targets.clone().detach()
        #test_labels = testset.targets.clone().detach()
        # 使用 dirichlet_split_noniid 函数获取客户端索引
        client_idcs = dirichlet_split_noniid(train_labels, self.num_client)
        #client_idcs_test = dirichlet_split_noniid(test_labels, self.num_client)
        # 为每个客户端创建 DataLoader
        trainloaders = []
        valloaders = []
        for idcs in client_idcs:
            train_subset = Subset(trainset, indices=idcs)
            trainloader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True,
                                     generator=torch.Generator().manual_seed(42))
            trainloaders.append(trainloader)
        #for idcs in client_idcs_test:
        #    test_subset = Subset(testset, indices=idcs)
        #    valloaders.append(DataLoader(test_subset, batch_size=self.batch_size, shuffle=True))
        for _ in range(self.num_client):
            valloaders.append(DataLoader(testset, batch_size=self.batch_size, shuffle=True,
                                         generator=torch.Generator().manual_seed(42)))
        return trainloaders, valloaders
    
    
def plot_data_distribution(trainloaders):
    for i, trainloader in enumerate(trainloaders):
        # 统计每个类别的样本数量
        class_counts = [0] * 10  # CIFAR-10 有 10 个类别
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
   # 在cifar10_load_dataset函数的末尾调用这个函数
   torch.manual_seed(42)
   args = args_parser()
   dataset_loader = DatasetLoad_noiid(args)
   trainloaders,_ = dataset_loader.noidd_cifar10_load_dataset()
   for i, trainloader in enumerate(trainloaders):
       client_samples = len(trainloader.dataset)
       print(client_samples)
   plot_data_distribution(trainloaders)