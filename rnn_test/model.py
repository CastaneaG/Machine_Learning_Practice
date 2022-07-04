import torch
import torch.nn as nn

num_time_steps = 50 #元素个数(一句话的单词数)
input_size = 1 #一条曲线所以就只有1
hidden_size = 16
output_size = 1

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.rnn = nn.RNN(
            input_size= input_size,
            hidden_size= hidden_size,
            num_layers= 1,
            batch_first=True,
            # batch_first=True [b,seq,word_vec]
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p,mean=0.0,std=0.001)
        self.linear = nn.Linear(hidden_size,output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev