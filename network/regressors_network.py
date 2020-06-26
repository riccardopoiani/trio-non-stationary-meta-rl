import torch


class RegressorNetwork(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(RegressorNetwork, self).__init__()
        self.layer_1 = torch.nn.Linear(input_size, hidden_size)
        self.layer_2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_3 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer_4 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = torch.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x

