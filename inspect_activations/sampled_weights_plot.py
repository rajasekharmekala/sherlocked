from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

batch_size = 32
test_batch_size = 1000
epochs = 50
lr = 0.01
momentum = 0.5
no_cuda = False
seed = 1
log_interval = 10

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True, **kwargs)


class MLPNetModified(nn.Module):
    def __init__(self, f1, f2, f3):
        super(MLPNetModified, self).__init__()
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.f1(x)
        x = self.fc2(x)
        x = self.f2(x)
        x = self.fc3(x)
        x = self.f3(x)
        return F.log_softmax(x)
    def name(self):
        return 'mlpnet'

plots_test_loss = []
plots_train_loss = []
plots_test_accuracy = []
epoch_model_parameters = []

def solve(f1, f2, f3,input_arr):
    print (str(f1).split()[1], str(f2).split()[1], str(f3).split()[1])
    model = MLPNetModified(f1, f2, f3)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    train_loss = []
    test_losses = []
    test_accuracy = []
    model_parameters= [[] for i in range(len(input_arr))]
    def train(epoch):
        model.train()
        loss_to_print = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            loss_to_print = loss.data[0]
            if batch_idx % log_interval == 0:
                train_loss.append(loss.data[0])
        print (epoch, loss_to_print)
    def test(epoch,input_arr):
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        test_loss /= len(test_loader.dataset)
        if (epoch == epochs):
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        test_losses.append(test_loss)
        test_accuracy.append(100. * correct / len(test_loader.dataset))
        x=list(model.parameters())
        for i in range(len(input_arr)):
            start_layer = input_arr[i][0]
            start_node  = input_arr[i][1]
            end_node    = input_arr[i][2]
            weight=(x[2*start_layer][end_node][start_node]).data[0]
            bias=(x[2*start_layer+1][end_node]).data[0]

            model_parameters[i].append([weight,bias])
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch,input_arr)
        epoch_model_parameters.append([str(f1).split()[1]+'_'+str(f2).split()[1]+'_'+str(f3).split()[1],model_parameters])

input_arr = []
for i in range(8):
    for j in range(5):
        input_arr.append([0,100*i,100*j])

solve(F.relu, F.relu, F.relu,input_arr)
solve(F.relu, F.sigmoid, F.relu,input_arr)
solve(F.relu, F.sigmoid, F.tanh,input_arr)

test_accuracy_last = []

for a in plots_test_accuracy:
    test_accuracy_last.append(['_'.join(a[0].split('_')[0:3]), a[1][len(a[1]) - 1]])

dic={0:"relu_relu_relu",1:"relu_sigmoid_relu",2:"relu_sigmoid_relu"}
test_accuracy_last.sort(key=lambda x: x[1])
for a in test_accuracy_last:
    print(a)
for model1 in epoch_model_parameters:
    for i in range(len(input_arr)):
        arr=np.array(model1[1][i])
        print(arr[:,0])
        print(arr[:,1])
        plt.figure()
        label_string = "L"+str(input_arr[i][0])+"-startNode-"+str(input_arr[i][1])+"-endNode-"+str(input_arr[i][2])+"-weight"
        plt.plot(arr[:,0],label=label_string)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(label_string+".png")
        plt.figure()
        label_string = "L"+str(input_arr[i][0])+"-startNode-"+str(input_arr[i][1])+"-endNode-"+str(input_arr[i][2])+"-bias"+dic[i]
        plt.plot(arr[:,1],label=label_string)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(label_string+".png")