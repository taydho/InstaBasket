import torch
import torch.nn as nn
import torch.nn.functional as F

class InstacartModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(InstacartModel, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))

        return x
