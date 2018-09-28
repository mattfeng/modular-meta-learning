import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SeqPixelNet(nn.Module):
    def __init__(self, name, image_size, batch_size):
        super(SeqPixelNet, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.img_size = image_size
        self.INT_LAYER_SIZE = 30
        self.init_vec = torch.Tensor([np.random.normal() for i in range(self.INT_LAYER_SIZE)])

        if torch.cuda.is_available():
            self.subset_mask = self.subset_mask.cuda()

        self.accuracy = 0

        self.initial = nn.Linear(self.INT_LAYER_SIZE, 784)

        self.int1 = nn.Linear(785, 784)

        self.final1 = nn.Linear(786, 50)
        self.final2 = nn.Linear(50, 10)
    
    def forward(self, img):
        img = img.view(-1, self.img_size) # reshape img into 1 x img_size

        x = self.init_vec.clone()

        if torch.cuda.is_available():
            x = x.cuda()

        x = F.relu(self.initial(x))
        x = F.log_softmax(x) # get the probabilities
        pix_1_ix = T.argmax(x)
        pix_1 = img[:, pix_1_ix]

        x = T.cat((x, pix_1), dim=-1)

        x = self.int1(x)

        x = F.log_softmax(x) # get the probabilities
        pix_2_ix = T.argmax(x)
        pix_2 = img[:, pix_2_ix]

        x = T.cat((x, pix_1, pix_2), dim=-1)

        x = F.relu(self.final1(x))
        x = F.relu(self.final2(x))

        return F.log_softmax(x).view(1, -1) # list of log probabilities