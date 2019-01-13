import os
import subprocess

from .tfrecord import TFRecord

#Set Input Parameters
DATA_DIR = './data'
TRAIN_DIR = './training'
MODEL_DIR = './models'


API_DIR = '../tensorflow/models/research/object_detection'
TRAIN_SCRIPT = os.path.join(API_DIR, 'legacy', 'train.py')

MODEL = 'faster_rcnn_inception_v2_pets'

model_config = MODEL + '.config'

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


# subprocess.call(['python', TRAIN_SCRIPT, 'logtostderr', 'train_dir=' + TRAIN_DIR, 'pipeline_config_path=' + MODEL_PATH])
result = subprocess.check_output(['python', TRAIN_SCRIPT, 'logtostderr', 'train_dir=' + TRAIN_DIR, 'pipeline_config_path=' + model_config])
print(result)
