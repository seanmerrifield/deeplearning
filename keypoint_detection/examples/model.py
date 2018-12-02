import torch.nn as nn
import torch.nn.functional as F

# Define Network
class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2, 2)

        self.full1 = nn.Linear(256 * 5 * 5, 1000)
        self.dropout = nn.Dropout(p=0.4)
        self.full2 = nn.Linear(1000, 136)



    def forward(self, x):

        #Convolutional Layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = F.relu(self.full1(x))
        x = self.dropout(x)

        x = self.full2(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x