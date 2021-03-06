from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.PATH import getAbsPath


def getDataloader(batchsz=32):
    batch_size = batchsz
    data_train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.RandomRotation(),
        # 在imageNet上测试的适合于RGB通道的均值和标准差。
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    data_test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    # datasets这个类只能建立一个一次加载一张的数据类型的loader
    cifar_train = datasets.CIFAR10(getAbsPath("data/cifar10"), train=True, transform=data_train_transform,download=True)
    # 这里要建立一次加载多个的使用DataLoader这个类
    cifar_train = DataLoader(cifar_train,batch_size=batch_size,shuffle=True,)


    cifar_test = datasets.CIFAR10(getAbsPath("data/cifar10"), train=False, transform=data_transform,download=True)
    # 这里要建立一次加载多个的使用DataLoader这个类
    cifar_test = DataLoader(cifar_test,batch_size=batch_size,shuffle=True,)

    x, label = iter(cifar_train).next()
    print('x:',x.shape,'label:',label.shape)
    return cifar_train,cifar_test

# batch_size = 32
# data_transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     transforms.ToTensor()
# ])
#
# # datasets这个类只能建立一个一次加载一张的数据类型的loader
# cifar_train = datasets.CIFAR10("cifar10", train=True, transform=data_transform, )
# # 这里要建立一次加载多个的使用DataLoader这个类
# cifar_train = DataLoader(cifar_train,batch_size=batch_size,shuffle=True,)
#
#
# cifar_test = datasets.CIFAR10("cifar10", train=False, transform=data_transform, )
# # 这里要建立一次加载多个的使用DataLoader这个类
# cifar_test = DataLoader(cifar_test,batch_size=batch_size,shuffle=True,)
#
# x, label = iter(cifar_train).next()
# print('x:',x.shape,'label:',label.shape)