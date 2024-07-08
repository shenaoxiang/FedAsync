import argparse #用于命令行参数解析
import torch


def args_parser():
    # 创建ArgumentParser对象，用于处理命令行参数
    parser = argparse.ArgumentParser()
    
    # 添加参数 --dataset，用于指定数据集名称，如 'mnist' 或 'cifar10'，默认值为 'cifar10'
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='MNIST',
        help='name of the dataset: mnist, cifar10'
    )
    
    # 用于指定模型类型
    parser.add_argument(
        '--model',
        type=str,
        default='mnist_cnn',
        help='name of model.'
    )
    
    # 用于设置训练时的批量大小，默认为 16
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='batch size when trained on client'
    )
    
    # 用于设置训练所使用的数据集类型
    parser.add_argument(
        '--is_iid',
        type=int,
        default=1,
        help="1 is noniid，0 is iid"
    )
    
    # 用于设置与云服务器通信的轮数，默认为 2
    parser.add_argument(
        '--num_communication',
        type=int,
        default=2,
        help='number of communication rounds with the cloud server'
    )
    
    # 用于设置局部更新的次数
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=2,
        help='number of local update (tau_1)'
    )
    
    # 用于设置边缘聚合的次数
    parser.add_argument(
        '--num_edge_aggregation',
        type=int,
        default=300,
        help='number of edge aggregation (tau_2)'
    )
    
    # 用于设置模型相似度阈值
    parser.add_argument(
        '--similarity',
        type=float,
        default=0.99,  # cifar10设置为:0.55, Mnist数据集设置为0.99，Fashion-Mnist为0.52
        help="similarity threshold value"
    )
    
    # 用于设置客户端训练的 SGD 学习率，默认为 0.01
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001, # cifar10和Mnist数据集学习率设为0.01，Fashion-Mnist(BN层)学习率设置为0.001，无BN层为0.01
        help='learning rate of the SGD when trained on client'
    )
    
    # 用于设置学习率衰减周期，默认为 1
    parser.add_argument(
        '--lr_decay_epoch',
        type=int,
        default=1,
        help='lr decay epoch'
    )
    
    # 用于设置SGD的动量，默认为 0
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.8,
        help='SGD momentum'
    )
    
    # 用于设置是否显示进度条，默认为 0（不显示）
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='verbose for print progress bar'
    )
    
    # 用于设置参与客户端的比例，默认为 1
    parser.add_argument(
        '--frac',
        type=float,
        default=1,
        help='fraction of participated clients'
    )
    
    # 用于设置可用客户端的数量
    parser.add_argument(
        '--num_clients',
        type=int,
        default=20,
        help='number of all available clients'
    )
    
    # 用于设置边缘节点的数量
    parser.add_argument(
        '--num_edges',
        type=int,
        default=2,
        help='number of edges'
    )
    
    # 用于设置随机种子，默认为 1
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (defaul: 1)'
    )
    
    # 用于设置是否显示数据分布，默认为 0
    parser.add_argument(
        '--show_dis',
        type=int,
        default=0,
        help='whether to show distribution'
    )
    
    # 用于选择 GPU，默认为 0
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU to be selected, 0, 1, 2, 3'
    )
    
    #  a_initialize_weight: 权重
    parser.add_argument(
        '--a_initialize_weight',
        default=1,
        type=float,
        help="Initial weight"
    )
    
    # 超参数设置(分层异步联邦学习聚合)，衰减函数设置
    parser.add_argument(
        '--edition',
        default=0.5,
        type=float,
        help="hyperparameter",
    )
    
    # 正则化参数
    parser.add_argument(
        '--lambda_reg',
        type=float,
        default=0.0001,
        help='proximal term constant'
    )
    
    # cifar10数据集的目标准确率
    parser.add_argument(
        '--cifar_traget_value',
        type=float,
        default=0.65,
        help='This is target value'
    )
    
    # F-Mnist数据集的目标准确率
    parser.add_argument(
        '--f_mnist_traget_value',
        type=float,
        default=0.85,
        help='This is target value'
    )

    # 解析命令行提供的参数，并将解析后的参数存储在变量args中
    args = parser.parse_args()
    # 检测当前环境是否支持 CUDA,args.cuda将被设置为True或False，取决于当前环境是否支持CUDA
    args.cuda = torch.cuda.is_available()  #将是否支持cuda的属性添加到args变量中
    return args