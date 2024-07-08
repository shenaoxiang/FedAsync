import torch
import time
import queue
import torch.optim as optim
import torch.nn as nn
import copy
from models.cifar_cnn import CifarCNN
from models.mnist_cnn import Mnist_CNN
from models.f_mnist_cnn import FashionMNIST_CNN

"""联邦学习中的客户端类，即数据拥有者"""

class Client:
    def __init__(self, server, client_id, train_loader, test_loader, args):
        self.server = server # 该客户端所对应的服务器
        self.client_id = client_id # 客户端ID
        self.model = None # 本地模型
        self.weights = None # 本地模型参数
        self.response_queue = queue.Queue()  # 接收服务器信息的队列
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader # 客户端的本地数据集(训练数据集)
        self.test_loader = test_loader # 客户端的本地数据集(测试数据集)
        self.edgemodel = None # 客户端所属的边缘服务器的局部聚合模型(即全局模型)
        self.args = args  # 联邦学习的相关参数设置
        self.criterion = nn.CrossEntropyLoss() # 定义损失函数
        self.loacl_test_acc = None # 测试时的准确率
        self.local_loss = None # 训练时的损失值
        self.local_test_loss = None # 测试时的损失值
        self.client_epoch = None # 全局模型时间戳
    
    def run(self, args):
        """启动联邦学习系统中的客户端"""
        self.__pre_treat()  # 初始化
        print("Client {}: Starting\n".format(self.client_id))
        while True:
            while self.response_queue.empty():
                time.sleep(1)
            # 从服务器接收权重
            response = self.response_queue.get()
            if response == "STOP":
                break
            else:
                self.weights = response[0]
                self.client_epoch = response[1]
            self.__set_local_weights(self.weights)
            train_loss, self.weights = self.local_update(args.num_local_update)
            loss, correct, total = self.test_local_model()
            self.local_loss = train_loss
            self.loacl_test_acc = correct / total
            self.local_test_loss = loss
            #print("Client ID：{}，trainingLoss：{}，Acc：{}".format(
            #    self.client_id, train_loss, correct / total))
            # 向服务器发送权重
            self.server.receive_from_client(self, self.weights, self.client_epoch)
            #while self.response_queue.empty():
            #     time.sleep(1)
            # 从服务器接收权重
            #response = self.response_queue.get()
            #if response == "STOP":
            #   #print("Client ID: {} Stopping.\n".format(self.client_id))
            #   break
            # else:
            #    self.weights = response
            #    #self.edgemodel.update_model(self.weights)
            # print("Client {}: got new weights from Server".format(self.client_id))
            time.sleep(2)
        print("Client {}: Stopping\n".format(self.client_id))
    
    def local_update(self, num_iter):
        """本地模型训练及测试"""
        self.model.train()
        self.edgemodel = copy.deepcopy(self.server.model)
        #print(self.edgemodel.parameters())
        optimizer = optim.SGD(params=self.model.parameters(), lr=self.args.lr)
        #optimizer = optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        # 初始化 ReduceLROnPlateau
        # ReduceLROnPlateau的参数中，mode表示指标的方向（最小化或最大化），factor表示学习率衰减的因子，
        # patience表示在多少个epoch内验证集指标没有改善时才进行学习率调整
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        self.criterion.to(self.device)
        total_loss = 0  # 初始化总损失为0
        for epoch in range(num_iter):
            for features, labels in self.train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(features)
                # 计算正则化项(本地模型和全局模型之间的差异)
                reg_loss = 0
                for w, w_t in zip(self.model.parameters(), self.edgemodel.parameters()):
                    reg_loss += torch.norm(w - w_t, 2) ** 2
                # 计算带有正则化项的损失函数
                reg_loss = (self.args.lambda_reg / 2) * reg_loss
                loss = self.criterion(outputs, labels) + reg_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()  # 累加损失
            # 计算平均损失值
            epoch_loss = total_loss / len(self.train_loader)
            # 在每个epoch结束时更新学习率
            scheduler.step()
            total_loss = 0  # 重置总损失为下一个epoch
        return epoch_loss, self.model.state_dict()
    
    def test_local_model(self):
        """测试模型的性能"""
        # 将模型设置为测试模型
        self.model.eval()
        self.criterion.to(self.device)
        correct, total, total_loss = 0, 0, 0.0
        # 使用torch.no_grad()上下文管理器。在测试阶段中，PyTorch将不用计算梯度，用来节省内存
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                # 使用与训练相同的损失函数
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)  # 返回给定维度上的最大值(每行)
                total += labels.size(0)
                correct += torch.sum(torch.eq(predicted, labels)).item()
        return total_loss / len(self.test_loader), correct, total
    
    # 私有函数
    def __pre_treat(self):
        """本地模型初始化及上传"""
        if self.args.model == "cifar_cnn":
            self.model = CifarCNN().to(self.device)
        elif self.args.model == "f_mnist_cnn":
            self.model = FashionMNIST_CNN().to(self.device)
        elif self.args.model == "mnist_cnn":
            self.model = Mnist_CNN().to(self.device)
        #self.model = FashionMNIST_CNN().to(self.device)
        #self.edgemodel = CifarCNN().to(self.device)
        #print("1")
        #while self.response_queue.empty():
        #    time.sleep(1)
        #response = self.response_queue.get()
        #self.weights = response
        #print("1")
    
    # 私有函数
    def __set_local_weights(self, weights):
        """加载本地模型权重"""
        self.model.load_state_dict(weights)