import os
import subprocess
import sys
from pathlib import Path
from tfrecord import TFRecord


#Set Input Parameters
DATA_DIR = './data'
TRAIN_DIR = './training'
MODEL_DIR = './models'

#Path to Tensorflow API training script
API_DIR = Path('../tensorflow/models/research/object_detection')
TRAIN_SCRIPT = str(API_DIR / 'legacy' / 'train.py')

MODEL = 'faster_rcnn_inception_v2_coco_2018_01_28'
model_config = MODEL + '.config'
model = Path(MODEL_DIR, model_config)

#Setting training directory
train_dir = Path(TRAIN_DIR)
train_dir.mkdir(parents=True, exist_ok=True)

if len(list(train_dir.glob('run_*'))) == 0:
        run_num = 1
else:
    last_path = sorted(list(train_dir.glob('run_*')))[-1]
    last_run = int(str(last_path).split("_")[-1])
    run_num = last_run + 1

dir_name = 'run_{:02d}'.format(run_num)
run_dir = train_dir / dir_name
run_dir.mkdir()


#Create Tf-Records for input data that will be used for training
train_data = TFRecord(data_dir = DATA_DIR,
                     xml_dir = str(Path(DATA_DIR, 'train')),
                     image_dir = str(Path(DATA_DIR, 'train')),
                     write_label_map = True
                     )
#
# test_data = TFRecord(data_dir = DATA_DIR,
#                      xml_dir = str(Path((DATA_DIR, 'test')),
#                      image_dir = str(Path(DATA_DIR, 'test')),
#                      write_label_map = False
#                      )


# subprocess.call(['python', TRAIN_SCRIPT, 'logtostderr', 'train_dir=' + TRAIN_DIR, 'pipeline_config_path=' + MODEL_PATH])
result = subprocess.check_output(['python', TRAIN_SCRIPT, 'logtostderr', 'train_dir=' + str(run_dir), 'pipeline_config_path=' + str(model)])
print(result)
