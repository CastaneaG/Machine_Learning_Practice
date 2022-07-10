import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from rnn_test.model import MyModel

num_time_steps = 50  # 元素个数(一句话的单词数)
input_size = 1  # 一条曲线所以就只有1
hidden_size = 16
output_size = 1
lr = 0.01

model = MyModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr)
hidden_prev = torch.zeros(1, 1, hidden_size)
print(model)


for iter in range(6000):
    # 模拟数据
    start = np.random.randint(3, size=1)[0]
    time_steps = np.linspace(start,start+10,num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps,1)
    # x 为 从第一个到 倒数第二个 y 为 从第二个到最后一个。 即往后预测一个根据x1预测x2，根据x2预测x3
    x = torch.tensor(data[:-1]).float().view(1,num_time_steps-1,1)
    y = torch.tensor(data[1:]).float().view(1,num_time_steps-1,1)

    output, hidden_prev = model(x, hidden_prev)
    # tensor.detach()返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false.
    # 得到的这个tensor永远不需要计算其梯度，不具有grad。
    # 即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
    hidden_prev = hidden_prev.detach()

    loss = criterion(output, y)
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print("iteration :{} loss {}".format(iter, loss.item()))


# 测试 测试时输入一个数，跟据预测进行预测
start = np.random.randint(3, size=1)[0]
time_steps = np.linspace(start, start+10, num_time_steps)
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x = torch.tensor(data[:-1]).float().view(1, num_time_steps-1, 1)
y = torch.tensor(data[1:]).float().view(1, num_time_steps-1, 1)

prediction = []
input = x[:, 0, :]
for _ in range(x.shape[1]):
    input = input.view(1, 1, 1)
    pred, hidden_prev = model(input,hidden_prev)
    input = pred
    prediction.append(pred.detach().numpy().ravel()[0])

x = x.data.numpy().ravel()
y = y.data.numpy()

plt.scatter(time_steps[:-1], x.ravel(), s=90)
plt.plot(time_steps[:-1], x.ravel())

plt.scatter(time_steps[1:], prediction)
plt.show()
