import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
from seqpixelnet import SeqPixelNet

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, ))
]) # initial transform to normalize the data

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
        transform=transforms),
    batch_size=1, shuffle=True
) # serve batches of 64 of MNIST

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms),
    batch_size=1, shuffle=True
)

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
    model.accuracy = float(correct) / len(test_loader.dataset)

if __name__ == "__main__":
    name = "SeqPixelNet"
    model = SeqPixelNet(name=name, image_size=28 * 28, batch_size=64)
    optimizer = optim.Adam(model.parameters())
    print(model)

    EPOCHS = 1
    for epoch in range(1, EPOCHS + 1):
        if torch.cuda.is_available():
            model.cuda()

        train(model, epoch)
        test(model)