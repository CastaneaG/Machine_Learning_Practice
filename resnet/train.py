import torch
from data.cifar10 import getDataloader
from resnet.model import ResNet18
from torch import nn, optim


if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
batch_size=32
model = ResNet18().to(device)
print(model)
cifar_train,cifar_test = getDataloader(batch_size)
## nn.CrossEntropyLoss() 之前包含Softmax 所以不需要自己在进行Softmax
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(1000):
    model.train()
    for batchidx , (x, label) in enumerate(cifar_train):
        x, label = x.to(device), label.to(device)
        logits = model(x)
        # logits: [b, 10]
        # label : [b]
        # loss  : tensor scalar
        loss = criterion(logits,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(epoch , loss.item())
    model.eval()
    # with torch.no_grad() 节省显存空间在验证时不进行存储和计算，如果显存充足可以不使用，仅使用model.eval()也足够
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in cifar_test:
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # pred : [b]
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred,label).float().sum().item()
            total_num += x.size(0)
        acc = total_correct/total_num
        print('epoch:',epoch, 'acc: ',acc)