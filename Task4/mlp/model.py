import torch



class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,10)
        )

    def forward(self, i_input):
        x = self.flatten(i_input)
        return self.linear_stack(x)