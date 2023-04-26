import torch



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = torch.nn.Linear(3,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, i_input):
        return self.sigmoid(self.linear(i_input))
