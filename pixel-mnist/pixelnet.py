import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PixelNet(nn.Module):
    def __init__(self, name, image_size, pixels):
        super(PixelNet, self).__init__()
        self.img_size = image_size
        self.pixels = pixels
        self.subset = set(np.random.choice(range(image_size), size=pixels, replace=False))
        self.subset_mask = torch.Tensor([p in self.subset for p in range(image_size)])

        if torch.cuda.is_available():
            self.subset_mask = self.subset_mask.cuda()

        self.fc0 = nn.Linear(image_size, 1000)
        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, 10)

        self.name = name
        self.accuracy = 0
    
    def forward(self, x):
        x = x.view(-1, self.img_size) # reshape x into 1 x img_size
        x *= self.subset_mask
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x) # list of log probabilities