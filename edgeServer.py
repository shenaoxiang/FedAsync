import time
import torch
import copy
from collections import OrderedDict
import queue
import numpy as np
from models.cifar_cnn import CifarCNN
from models.mnist_cnn import Mnist_CNN
from models.f_mnist_cnn import FashionMNIST_CNN
import torch.nn as nn


"""边缘服务器，每个联邦通信轮次的标准迭代，包含联邦学习中的三个必要过程：客户端选择、通信和模型聚合"""


class Edge:
    def __init__(self, id, args):
        self.id = id # 边缘服务器ID
        #self.server = server
        self.running = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if args.model == "cifar_cnn":
            self.model = CifarCNN().to(self.device)
        elif args.model == "f_mnist_cnn":
            self.model = FashionMNIST_CNN().to(self.device)
        elif args.model == "mnist_cnn":
            self.model = Mnist_CNN().to(self.device)
        self.weights = self.model.state_dict()  # 获取模型初始权重
        self.asynchronous_state = True  # 是否为异步
        self.id_registration = []  # 注册的客户端ID列表
        self.sample_registration = {}  # 每个客户端的样本数量
        self.clients = [] # 客户端列表
        self.data_queue = queue.Queue()  # 用户发给服务器的数据队列
        self.model_test_acc = None # 每一次全局聚合完后边缘模型的准确率
        self.model_test_loss = None
        self.model_loss = None
        self.epoch = 0 # 全局迭代次数
        self.client_epoch = 0 # 客户端的模型时间戳
        self.criterion = nn.CrossEntropyLoss() # 定义损失函数
        
    
    def run(self, args, clients):
        """运行FL系统，其中对全局模型进行迭代训练和评估"""
        updates = 0
        first_update = 0 # 用来记录第一次到达阈值的更新
        start_time = time.time()
        for client in clients:
            self.client_register(client)
        #print(sum(self.sample_registration.values()))
        # 设置模型 获取模型初始权重 发送权重到每个用户
        self.__pre_treat()
        if self.asynchronous_state:
            print('Server: Starting in asynchronous mode\n')
            # 开始进行权重更新
            while updates < args.num_edge_aggregation:
                if not self.data_queue.empty():
                    # 判断是否有新的客户端加入联邦学习训练
                    #responce = self.data_queue.get()
                    #client = responce[0] # 客户端设备
                    #client_require = responce[1] # 客户端加入请求
                    #if client_require == "True":
                    #    self.add_client(client)
                    client, client_weights, epoch = self.data_queue.get()  # 从用户接收权重
                    self.model_test_acc = client.loacl_test_acc
                    #print(self.model_test_loss)
                    if args.model == "cifar_cnn":
                        if self.model_test_acc >= args.cifar_traget_value and first_update == 0:
                            first_update = 1
                            end_time = time.time()
                            print("所花费时间: {}".format(end_time - start_time))
                    elif args.model == "f_mnist_cnn":
                        if self.model_test_acc >= args.f_mnist_traget_value and first_update == 0:
                            first_update = 1
                            end_time = time.time()
                            print("所花费时间: {}".format(end_time - start_time))
                    self.epoch = updates # 全局迭代次数
                    self.model_loss = client.local_loss
                    self.client_epoch = epoch
                    self.weights = self.aggregate_asynchronously(client_weights, args)  # 更新权重
                    self.__set_weights(self.weights) # 加载模型权重
                    print("Server {} : Data updated from Client ID: {}, epoch: {}, "
                          "trainingLoss: {}, testLoss: {}, Acc: {}".format(self.id,
                                                                           client.client_id, updates,
                                                                           client.local_loss,
                                                                           client.local_test_loss,
                                                                           client.loacl_test_acc))
                    self.send_to_client(client,self.weights)  # 将更新好的权重发给客户端
                    updates += 1
                else:
                    time.sleep(1)
            # 停止并且给用户发停止信号(当达到最大迭代次数后)
            for client in clients:
                client.response_queue.put("STOP")
                self.running = False
                #print("Server: Stopping")
        else:
            print('Server: Starting in synchronous mode\n')
            # 按照一定的比例挑选合适的客户端
            select_clients = self.sample(args.frac)
            # 开始进行权重更新
            while updates < args.num_edge_aggregation:
                if self.data_queue.qsize() != len(select_clients):  # 判断队列元素是否为空
                    time.sleep(1)
                else:
                    # 获取采样客户端的模型权重(client_weight)
                    weights_list = [self.data_queue.get()[1] for _ in range(len(select_clients))]
                    # 获取采样客户端的数据集合
                    sample_list = [len(client.train_loader.dataset) for client in select_clients]
                    self.weights = self.aggregate_synchronously(weights_list,sample_list)
                    print("Server: Data updated, epoch: {}".format(updates))
                    # 给每个用户发更新后的权重
                    for client in select_clients:
                        client.response_queue.put(self.weights)
                    updates += 1
            # 停止并且给用户发停止信号(当达到最大迭代次数后)
            for client in select_clients:
                client.response_queue.put("STOP")
                self.running = False
                print("Server: Stopping")
    
    def add_client(self, client):
        """在联邦学习训练过程中有新的客户端加入训练"""
        if client.server.id == self.id:
            if client.client_id not in self.id_registration:
                self.client_register(client)
                self.send_to_client(client, self.weights)
        
    def sample(self, selected_cnum):
        total_sample_num = sum(self.sample_registration.values())
        p_clients = []
        for client_sample_num in self.sample_registration.values():
            p_clients.append(client_sample_num / total_sample_num)
        selected_cids = np.random.choice(self.id_registration, selected_cnum, replace=True, p=p_clients)
        selected_clients = [self.clients[i] for i in selected_cids]
        return selected_clients
    
    def client_register(self, client):
        """客户端注册，以单个客户端形式进行注册"""
        #self.clients = clients
        #self.cids = [int(client.client_id) for client in self.clients]  # 客户端id集合
        #for client in self.clients:
        #    self.id_registration.append(client.client_id)
        #    self.sample_registration[client.client_id] = len(client.train_loader.dataset)
        if client.client_id not in self.id_registration:
            self.clients.append(client)
            self.id_registration.append(client.client_id)
            self.sample_registration[client.client_id] = len(client.train_loader.dataset)
            
    def aggregate_asynchronously(self, client_weights, args):
        """异步聚合算法"""
        #alpha = 1 / (1 + np.exp(self.epoch - self.client_epoch - args.edition))
        alpha = np.power(self.epoch - self.client_epoch + 1, -args.edition)
        #alpha = 0.2
        new_weights = OrderedDict()
        for key in self.weights:
            new_weights[key] = (1 - alpha) * self.weights[key] + alpha * client_weights[key]
        return new_weights
    
    def aggregate_synchronously(self, weights_list, sample_num):
        """同步聚合算法"""
        total_sample_num = sum(sample_num)
        temp_sample_num = sample_num[0]
        w_avg = copy.deepcopy(weights_list[0])
        for k in w_avg.keys():
            for i in range(1, len(weights_list)):
                w_avg[k] += torch.mul(weights_list[i][k], sample_num[i] / temp_sample_num)
            w_avg[k] = torch.mul(w_avg[k], temp_sample_num / total_sample_num)
        return w_avg
    
    def send_to_client(self, client, weights):
        """将聚合完毕的全局参数发送给客户端"""
        client.response_queue.put((copy.deepcopy(weights),self.epoch))
        #print("1")
        #print(client.response_queue.get())
    
    def receive_from_client(self, client, client_weight, epoch):
        """接收客户端的更新参数"""
        self.data_queue.put((client, client_weight, epoch))
    
    def __pre_treat(self):
        """模型初始化"""
        # 加载权重
        self.__set_weights(self.weights)
        for client in self.clients:
            self.send_to_client(client, self.weights)
            #print("1")
        #print(self.weights)
        #print("1")
    
    def __set_weights(self, weights):
        """加载模型权重参数"""
        self.model.load_state_dict(weights)
        #print("1")
        
    def send_to_cloudserver(self, cloud):
        """发送模型聚合后的参数给云服务器"""
        cloud.receive_from_edge(edge_id=self.id,
                                weights=copy.deepcopy(
                                    self.weights))
    
    def receive_from_cloudserver(self, weights):
        """接收云服务器的模型参数"""
        self.weights = weights
        #print(self.weights)