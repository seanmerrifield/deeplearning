import logging
import time
from pathlib import Path
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Trainer:

    MODEL_NAME = "trained_model.pt"
    SUMMARY_FILE = "net_summary.txt"
    LOG_FILE = "log.txt"
    LOSS_PLOT = "loss.png"

    MSE = "Mean Squared Error"
    CROSS_ENTROPY = "Cross Entropy"

    def __init__(self, net, name, root_dir=None, log_path=None):

        self.name = name
        self.net = net

        if root_dir == None:    self.root = "."
        else:                   self.root = root_dir

        self.opt = None
        self.loss = None

        self.logger = self._init_log(name, log_path)



    def _init_log(self, name, log_path=None, log_name=None):
        """
        Initialize logger for training
        @param name:        Name of logger
        @param log_path:    Path to save log to
        @param log_name:    Name of log file
        @return:            None
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if log_path == None: log_path = self.root
        if log_name == None: log_name = self.LOG_FILE

        fh = logging.FileHandler(Path(log_path, log_name))
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)

        logger.addHandler(fh)

        return logger

    def _create_summary(self):
        with open(Path(self.root, self.SUMMARY_FILE), 'w+') as f:
            f.write("Training Parameters\n")
            f.write("Learning Rate: {}\n".format(self.lr))
            f.write("Batch Size: {}\n".format(self.batch_size))
            f.write("Loss Function: {}\n".format(self.type))

            f.write("Network Summary\n")
            f.write(str(self.net))

    def _log(self, level, text):
        print(text)
        if level.lower() == 'info': self.logger.info(text)
        elif level.lower() == 'warning': self.logger.warning(text)
        elif level.lower() == 'error': self.logger.error(text)
        elif level.lower() == 'debug': self.logger.debug(text)

    def _save_model(self, model_name=None):
        if model_name == None: model_name = self.MODEL_NAME

        torch.save(self.net.state_dict(), str(Path(self.root, model_name)))


    def _loader(self, set, batch_size):
        """
        Returns a Pytorch data loader based on data set and batch size
        @param set:         Dataframe
        @param batch_size:  Batch size
        @return:            Pytorch DataLoader
        """
        return DataLoader(set,
                   batch_size=batch_size,
                   shuffle=True,
                   num_workers=0)


    def train_set(self, set, batch_size):
        self.batch_size = batch_size
        self.train_loader = self._loader(set, batch_size)

    def test_set(self, set, batch_size):
        self.batch_size = batch_size
        self.test_loader = self._loader(set, batch_size)

    def net(self, net):
        """
        Sets a Pytorch network to train
        @param net:     Sets Pytorch network
        @return:        None
        """
        self.net = net

    def optimizer(self, lr=0.001):
        """
        Sets Pytorch optimizer for training
        @param lr:  Learning rate
        @return:    None
        """
        self.lr = lr
        self.opt = optim.SGD(self.net.parameters(), lr=lr)

    def loss_fn(self, type):
        """
        Sets Pytorch loss function for training
        @return:    None
        """
        if  type == self.MSE:
            self.type = type
            self.loss = nn.MSELoss()
        elif    type == self.CROSS_ENTROPY: self.loss = nn.CrossEntropyLoss()
        else:
            raise Exception

    def train(self, epochs):
        """
        Runs training on defined network
        @param epochs:  Number of epochs
        @return:        List of loss values per batch
        """

        self._create_summary()

        self._log('INFO', "Starting training over {} epochs".format(epochs))
        start = time.time()

        # prepare the net for training
        self.net.train()

        self.loss_over_time = []

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0

            # train on batches of data, assumes you already have train_loader
            for batch_i, data in enumerate(self.train_loader):
                # get the input images and their corresponding labels
                images = data['image']
                key_pts = data['keypoints']

                # flatten pts
                key_pts = key_pts.view(key_pts.size(0), -1)

                # convert variables to floats for regression loss
                key_pts = key_pts.type(torch.FloatTensor)
                images = images.type(torch.FloatTensor)

                # forward pass to get outputs
                output_pts = self.net(images)

                # calculate the loss between predicted and target keypoints
                loss = self.loss(output_pts, key_pts)

                # zero the parameter (weight) gradients
                self.opt.zero_grad()

                # backward pass to calculate the weight gradients
                loss.backward()

                # update the weights
                self.opt.step()

                # print loss statistics
                running_loss += loss.item()
                if batch_i % 10 == 9:  # print every 10 batches
                    self._log('INFO','Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 10))
                    # record and print the avg loss over the 10 batches
                    self.loss_over_time.append(running_loss)
                    running_loss = 0.0

        self._log('INFO', "Finished training")
        train_time = (start - time.time()) / 60.0

        self._log('INFO', "Training took {} minutes", train_time)

        self._save_model()
        self._log('INFO', "Saved trained model")

        self.create_loss_plot()
        self._log('INFO', "Created loss plot")

        return self.loss_over_time


    def create_loss_plot(self):
        """
        Creates line plot of loss over batches
        @return:    Pyplot with loss over batches
        """
        # visualize the loss as the network trained
        plt.plot(self.loss_over_time)
        plt.xlabel('10\'s of batches')
        plt.ylabel('loss')
        plt.ylim(0, 1)  # consistent scale
        plt.save_fig(str(Path(self.root, self.LOSS_PLOT)))
        return plt

    def sample_output(self):
        # iterate through the test dataset
        for i, sample in enumerate(self.test_loader):

            # get sample data: images and ground truth keypoints
            images = sample['image']
            key_pts = sample['keypoints']

            # convert images to FloatTensors
            images = images.type(torch.FloatTensor)

            # forward pass to get net output
            output_pts = self.net(images)

            # reshape to batch_size x 68 x 2 pts
            output_pts = output_pts.view(output_pts.size()[0], 68, -1)

            # break after first image is tested
            if i == 0:
                return images, output_pts, key_pts