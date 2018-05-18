#!/usr/bin/python3

import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
mpl.use("Agg")


from dataset import GTSRBTraining, make_train_valid_loader, evaluate
from network import NeuralNetwork
from preprocessing import shared_transform


learning_rate = 0.0001
momentum = 0.5
num_epochs = 40


training_data = GTSRBTraining(training_path = "train", transform = shared_transform)


#separate datasets on training 80%  and validation 20%
train_loader, valid_loader = make_train_valid_loader(training_data, valid_percentage = 0.2)

# NN architecture


conv_size = lambda w, p: (w - 3 + p*2) + 1
pooled_size = lambda w, p: int(conv_size(w, p)/2

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3) 

	
	dimension = (pooled_size(pooled_size(32, 0), 0)**2)*64
	
        self.linear1 = nn.Linear(dimension, 512)
        self.linear2 = nn.Linear(512, 43)
	self.sm = nn.Softmax()

    def forward(self, x):
        
        relu1 = F.elu(self.conv1(x.float()))
        #pooling
        mp1 = F.max_pool2d(relu1, kernel_size = 2, stride = 2)

        relu2 = F.elu(self.conv2(mp1))
        #pooling
        mp2 = F.max_pool2d(relu2, kernel_size = 2, stride = 2)

        # Fully-connected layer
        flat = mp2.view(mp2.size(0), -1)
        hidden = F.elu(self.linear1(flat))
        dropout = F.dropout(hidden)
        y = F.log_softmax(self.linear2(dropout), dim = 1)

	return y



net = NeuralNetwork()
net.cuda()

# Loss function and optimization method
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = learning_rate, momentum = momentum)


x = []
y_validation = []
y_training = []


for epoch in range(args.num_epochs):
    x.append(epoch + 1)
    print("[Epoch {}]".format(epoch + 1))

    # Train
    net.train()
    for batch_idx, (Xs, ys) in enumerate(train_loader):
        Xs, ys = Xs.cuda(), ys.cuda()

        # Forward 
        ys_hat = net(Xs)
        loss = cost(ys_hat, ys)

        # Backward 
        optimizer.zero_grad()
        loss.backward()

        # Optimize
        optimizer.step()

        # Report loss every 50 batches
        if batch_idx % 50 == 0:
            print("{} / {} => loss = {}".format(batch_idx * len(Xs), len(train_loader.dataset), loss.item()))

    if args.eval_train:
        # Evaluate on the training set
        accuracy = evaluate(net, train_loader)
        y_training.append(accuracy)
        print("Training accuracy: {}%".format(accuracy))

    # Evaluate on the validation set
    accuracy = evaluate(net, valid_loader)
    y_validation.append(accuracy)
    print("Validation accuracy: {}%".format(accuracy))

# Plot validation accuracy over epochs
np.save('valid.npy', y_validation)


# Save the learning curve and the model
timestamp = datetime.datetime.now()
identifier = "model_{}_{}_{}_{}_{}".format(timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute)

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)

plt.savefig("{}/{}.lc.png".format(directory, identifier))

model_path = "{}/{}".format(directory, identifier)
torch.save(net.state_dict(), model_path)
print("Model parameters saved to {}.".format(model_path))
