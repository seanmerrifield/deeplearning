"""
This is an example for solving image classification problems. This example
uses the Fashion MNIST dataset.

The model uses convolutional neural networks with a VGG-16 architecture.
The architecture includes the following:

- 2 convolutional layers, each followed by a max pooling layer
- 1 dropout layer to regularization
- 1 fully connected layer for classification

The model is created using Pytorch, and image processing is done using
OpenCV and Matplotlib.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

# data loading and transforming
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


### DOWNLOAD THE DATA ###

#Define a transform to read the data in as a tensor
data_transform = transforms.ToTensor()

# choose the training and test datasets
train_data = FashionMNIST(root='./data', train=True,
                                   download=True, transform=data_transform)

test_data = FashionMNIST(root='./data', train=False,
                                  download=True, transform=data_transform)


# Print out some stats about the training and test data
print('Train data, number of images: ', len(train_data))
print('Test data, number of images: ', len(test_data))


### CREATE BATCHES ###

batch_size = 20
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# specify the image classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



### VISUALIZE INPUT DATA ###

# obtain one batch of training images
images, labels = iter(train_loader).next()
images = images.numpy()

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size / 2, idx + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title(classes[labels[idx]])

plt.show()


"""
Define Network Architecture by inhereting from Pytorch's
module class. 
"""

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        #Convolutional layers take 3 parameters:
        # - Number of input channels
        # - Number of output channels
        # - Filter size (nxn)
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)

        #Pooling layers take 2 parameters:
        # - kernal_size
        # - stride
        self.pool = nn.MaxPool2d(2, 2)

        #Dropout layer with dropout probability as input
        self.dropout = nn.Dropout(p=0.5)

        #Final classification layer takes 2 parameters:
        # - Number of input channels (last conv layer width x height x depth)
        # - Number of classes
        self.linear = nn.Linear(20 * 5 * 5, 10)


    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = F.relu(self.linear(x))

        # final output
        return x


net = Net()
print(net)



### DEFINE LOSS FUNCTION AND OPTIMIZER ###

# CrossEntropyLoss is common loss function for classification problems
criterion = nn.CrossEntropyLoss()

# Use stochastic gradient descent for optimization
optimizer = optim.SGD(net.parameters(), lr=0.001)


### STARTING TEST ACCURACY ###

# Check the starting test accuracy for the untrained network
# This should be close to random guessing, which would have a
# probability of 10% correct for 10 classes

correct = 0
total = 0
for images, labels in test_loader:

    # Wrap inputs with Pytorch's variable wrapper
    images = Variable(images)

    # Run feed forward
    # outputs.data is a batch_size * num_classes tensor
    # which contains softmax probability predictions
    # for each class
    outputs = net(images)

    #Get the predicted class
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += (predicted == labels).sum()

accuracy = 100 * correct.item() / total

print("Pre-trained test accuracy is: {}%".format(accuracy))


#### Run training of the model ###

def train(n_epochs):

    loss_over_time = []

    for epoch in range(n_epochs):
        running_loss = 0.0

        for batch_i, data in enumerate(train_loader):

            # Get input images and labels
            inputs, labels = data

            # Wrap them with Pytorch variables
            inputs, labels = Variable(inputs), Variable(labels)

            # Zero the parameter (weight) gradients
            optimizer.zero_grad()

            # Run forward pass
            outputs = net(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Run backpropagation
            loss.backward()

            # Update the weights
            optimizer.step()

            # print loss statistics
            running_loss += loss.item()
            if batch_i % 1000 == 999:    # print every 1000 batches
                avg_loss = running_loss/1000
                # record and print the avg loss over the 1000 batches
                loss_over_time.append(avg_loss)
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, avg_loss))
                running_loss = 0.0

    print("Training completed")
    return loss_over_time


epochs = 30
training_loss = train(epochs)

# Save trained model
model_dir = './saved_models/'
model_name = 'model_1.pt'
# after training, save your model parameters in the dir 'saved_models'
torch.save(net.state_dict(), model_dir+model_name)


# visualize the loss as the network trained
plt.plot(training_loss)
plt.xlabel('1000\'s of batches')
plt.ylabel('loss')
plt.ylim(0, 2.5) # consistent scale
plt.show()


### Run Testing of Model ###

#Initialize tensor and lists to monitor test loss and accuracy
test_loss = torch.zeros(1)
class_correct = [0. for i in range(10)]
class_total = [0. for i in range(10)]

#set the module to evaluation model
net.eval()

for batch_i, data in enumerate(test_loader):

    # Get input images and labels
    # Note that we do not need to wrap them in Pytorch
    # Variables since we are not tracking the weight changes
    inputs, labels = data

    # Run forward pass
    outputs = net(inputs)

    # Calculate loss
    loss = criterion(outputs, labels)

    # Update average test loss
    test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))

    # get the predicted class from the maximum value in the output-list of class scores
    _, predicted = torch.max(outputs.data, 1)

    # compare predictions to true label
    # this creates a `correct` Tensor that holds the number of correctly classified images in a batch
    correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = labels.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))



# obtain one batch of test images
images, labels = iter(test_loader).next()

# get predictions
preds = np.squeeze(net(Variable(images, volatile=True)).data.max(1, keepdim=True)[1].numpy())
images = images.numpy()

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx] else "red"))

plt.show()

