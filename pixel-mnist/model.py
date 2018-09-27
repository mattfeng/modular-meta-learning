import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

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

class PixelNet(nn.Module):
    def __init__(self, name, image_size, pixels):
        super(PixelNet, self).__init__()
        self.img_size = image_size
        self.pixels = pixels
        self.subset = set(np.random.choice(range(image_size), size=pixels, replace=False))
        self.subset_mask = torch.Tensor([p in self.subset for p in range(image_size)])

        self.fc0 = nn.Linear(image_size, 1000)
        self.fc1 = nn.Linear(1000, 50)
        self.fc2 = nn.Linear(50, 10)

        self.name = name
        self.accuracy = 0
    
    def forward(self, x):
        x = x.view(-1, self.img_size) # reshape x into 1 x img_size
        x = x * self.subset_mask
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x) # list of log probabilities

def train(model, epoch):
    model.train() # put the model in training mode
    for batch_ix, (data, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()
        data, labels = Variable(data), Variable(labels)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        if batch_ix % 100 == 0:
            print("Train epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(
                epoch, batch_ix * len(data), len(train_loader.dataset),
                100. * batch_ix / len(train_loader), loss.data[0]
            ))

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the indices of the max log-prob
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    
    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    model.accuracy = correct / len(test_loader.dataset)

if __name__ == "__main__":
    models = {}
    for i in range(100):
        name = "PixelNet_{}".format(i)
        model = PixelNet(name=name, image_size=28 * 28, pixels=10)
        optimizer = optim.Adam(model.parameters())
        models[name] = (model, optimizer)
        print(model)
        print("Pixels chosen: {}".format(model.subset))
    EPOCHS = 1

    for epoch in range(1, EPOCHS + 1):
        for name, (model, optimizer) in models.items():
            if torch.cuda.is_available():
                model.cuda()

            print("Training {}: Pixels ({})".format(name, model.subset))
            train(model, epoch)
            test(model)
    
    for name, (model, _) in models.items():
        print("{:>12}: {:.1f} {}".format(name, model.accuracy, model.subset))
