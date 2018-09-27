import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
]) # initial transform to normalize the data

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
        transform=transforms),
    batch_size=64, shuffle=True
) # serve batches of 64 of MNIST

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms),
    batch_size=64, shuffle=True
)

class Pixel10Net(nn.Module):
    def __init__(self, image_size, pixels):
        super(Pixel10Net, self).__init__()
        self.img_size = image_size
        self.pixels = pixels

        self.fc0 = nn.Linear(image_size, 1000)
        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = x.view(-1, self.img_size) # reshape x into 1 x img_size
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)

model = Pixel10Net(image_size=28 * 28, pixels=10)
model.cuda()