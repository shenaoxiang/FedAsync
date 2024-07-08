import threading
from edgeServer import Edge
from client import Client
import ssl
from options import args_parser
#from dataset import prepare_dataseta
from iid_dataset_split import DatasetLoad_iid
from no_iid_datset_split import DatasetLoad_noiid

# 用于改变默认的HTTPS上下文以忽略对SSL证书的验证
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    args = args_parser()
    # trainloaders, valloaders, testloader = prepare_dataset(5,10)
    datasets_iid = DatasetLoad_iid(args)
    datasets_noiid = DatasetLoad_noiid(args)
    if args.is_iid == 1:
        if args.dataset_name == 'Fashion-MNIST':
            train_loaders, val_loaders = datasets_noiid.noiid_fashion_load_dataset()
        elif args.dataset_name == 'CIFAR10':
            train_loaders, val_loaders = datasets_noiid.noidd_cifar10_load_dataset()
        elif args.dataset_name == 'MNIST':
            train_loaders, val_loaders = datasets_noiid.noidd_mnist_load_dataset()
    elif args.is_iid == 0:
        if args.dataset_name == 'Fashion-MNIST':
            train_loaders, val_loaders = datasets_iid.fashion_load_dataset()
        elif args.dataset_name == 'CIFAR10':
            train_loaders, val_loaders = datasets_iid.cifar10_load_dataset()
        elif args.dataset_name == 'MNIST':
            train_loaders, val_loaders = datasets_iid.mnist_load_dataset()
    
    clients = []  # 用户列表 每个元素为Client类的实例
    client_threads = []  # 用户线程列表 用于启动线程
    server = Edge(1, args)
    for i in range(int(args.num_clients)):
        client = Client(server=server, client_id=i, train_loader=train_loaders[i], test_loader=val_loaders[i],
                        args=args)
        clients.append(client)
    for client in clients:
        client_threads.append(threading.Thread(target=client.run, args=(args,)))
    server_thread = threading.Thread(target=server.run, args=(args, clients,))
    server_thread.start()
    for i in client_threads:
        i.start()
    server_thread.join()
    for thread in client_threads:
        thread.join()
    


if __name__ == "__main__":
    main()