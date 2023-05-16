class Layer(Module):
    def __init__(self,
                 i_n_features_input,
                 i_n_features_output):
        super().__init__()
        self.input_n = i_n_features_input
        self.output_n = i_n_features_output
        self.weight = torch.nn.Parameter(torch.empty(self.output_n, self.input_n))
        torch.nn.init.uniform_(self.weight, -0.1, 0.1)
        
    def forward(self, input):
        return Function.apply(input, self.weight)