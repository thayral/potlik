
# svhn, 78% accuracy
#adapted from mnist and imagenet torch examples

from __future__ import print_function

import argparse
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


########################
#### The argument parser : define hyper parameters
################


parser = argparse.ArgumentParser(description='SVHN on PyTorch attempt by Theo Ayral')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')



##################################



def main():
    
    global args	
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if (args.cuda == False): 
        print("cuda is disabled or not available")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # Data Loader from torch and vision
    # Crop 28x28 and grayscale like mnist
    transform_composition=transforms.Compose([transforms.CenterCrop(28),transforms.Grayscale(), transforms.ToTensor()])


    train_data= datasets.SVHN('data/', split='train', download=True,
	               	transform=transform_composition)

    train_loader = torch.utils.data.DataLoader(train_data,
	    		batch_size=args.batch_size, shuffle=True, **kwargs)


    test_data= datasets.SVHN('data/', split='test', download=True,
	                transform=transform_composition)
	
    test_loader = torch.utils.data.DataLoader(test_data,
	    		batch_size=args.test_batch_size, shuffle=True, **kwargs)



    model = Net()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    

    for epoch in range(1, args.epochs + 1):
        train(epoch, model, optimizer, train_loader)
        test(model, optimizer, test_loader)


# This is copi-pasted from pytorch's mnist example
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(epoch, model, optimizer, train_loader):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, optimizer, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()


