import os

from .tfrecord import TFRecord

DATA_DIR = './data'
TRAIN_DIR = './train'


#Create Tf-Records for input data that will be used for training
train_data = TFRecord(data_dir = DATA_DIR,
                     xml_dir = os.path.join(DATA_DIR, 'train'),
                     image_dir = os.path.join(DATA_DIR, 'train'),
                     write_label_map = True
                     )

test_data = TFRecord(data_dir = DATA_DIR,
                     xml_dir = os.path.join(DATA_DIR, 'test'),
                     image_dir = os.path.join(DATA_DIR, 'test'),
                     write_label_map = False
                     )

