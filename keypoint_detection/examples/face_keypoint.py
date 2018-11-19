import requests
import zipfile
import io
import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from data_load import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor
import helper
from model import Net
from trainer import Trainer



#Get labelled Facial Keypoint Data
URL = "https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/"
DATA_FILE = 'train-test-data.zip'
DOWNLOAD_PATH = './data/'
if len(os.listdir(DOWNLOAD_PATH)) == 0:
    r = requests.get(URL + DATA_FILE)
    assert(r.ok), 'Input data files could not be downloaded from URL'
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DOWNLOAD_PATH)





### PREPARE DATA ###

# order matters! i.e. rescaling should come before a smaller crop
data_transform = transforms.Compose([Rescale(250),RandomCrop(224),Normalize(),ToTensor()])

# create the transformed dataset
transformed_dataset = FacialKeypointsDataset(csv_file='./data/training_frames_keypoints.csv',
                                             root_dir='./data/training/',
                                             transform=data_transform)


print('Number of images: ', len(transformed_dataset))



# create the test dataset
test_dataset = FacialKeypointsDataset(csv_file='./data/test_frames_keypoints.csv',
                                             root_dir='./data/test/',
                                             transform=data_transform)


### PREPARE NETWORK ###



# train network
batch_size = 10
n_epochs = 1 # start small, and increase when you've decided on your model structure and hyperparams
lr = 0.001


run_dir = helper.create_training_dir()

net = Net()
trainer = Trainer(net, name="Training Run", root_dir=run_dir)

trainer.train_set(transformed_dataset, batch_size)
trainer.test_set(test_dataset, batch_size)
trainer.loss_fn(trainer.MSE)
trainer.optimizer(lr=lr)

### RUN INFERENCE IN TEST SET ###

# # returns: test images, test predicted keypoints, test ground truth keypoints
# test_images, test_outputs, gt_pts = trainer.sample_output()
#
# # print out the dimensions of the data to see if they make sense
# print(test_images.data.size())
# print(test_outputs.data.size())
# print(gt_pts.size())
#
# visualize_output(test_images, test_outputs, gt_pts)
#


### RUN TRAINING
trainer.train(n_epochs)

