import torch
from torch import nn
import torch.nn.functional as F


class Net_arc(nn.Module):
    def __init__(self):
        super(Net_arc, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(128 * 32 * 32, 256)  # Increased the output features of fc1
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc2 = nn.Linear(256, 18)  # Output layer for classification

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        output = F.log_softmax(self.fc2(x), dim=1)  # Apply softmax to the output
        return output
