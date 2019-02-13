import os
import subprocess
import sys
import tarfile
import urllib
from pathlib import Path


if Path.cwd().name == 'object_detection':
    root = '..'
else:
    root = '.'

#Path to Tensorflow API training script
API_DIR = Path(root, 'tensorflow/models/research/object_detection')
sys.path.append(str(API_DIR.parent.absolute()))
TRAIN_SCRIPT = str((API_DIR / 'legacy' / 'train.py').absolute())

from tfrecord import TFRecord
from filehandler import FileHandler

#Ensure working directory is in object_detection folder
os.chdir(str(Path(root,'object_detection').absolute()))

#Set Input Parameters
DATA_DIR = './data'
TRAIN_DIR = './training'
MODEL_DIR = './models'

IMAGE_DIR = Path(DATA_DIR, 'images')
XML_DIR = Path(DATA_DIR, 'xml')


#Model and Training Config File 
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL = 'faster_rcnn_inception_v2_coco_2018_01_28'
model_file = MODEL + '.tar.gz'
model_config = MODEL + '.config'
model = Path(MODEL_DIR, model_config)
PATH_TO_FROZEN_GRAPH = str(Path(MODEL_DIR, MODEL, 'frozen_inference_graph.pb'))


if not Path(PATH_TO_FROZEN_GRAPH).exists():
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + model_file, str(Path(MODEL_DIR, model_file)))
    tar_file = tarfile.open(str(Path(MODEL_DIR, model_file)))
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, MODEL_DIR)

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

#Split training and test data
fh = FileHandler(DATA_DIR, IMAGE_DIR, XML_DIR, train_size=0.8)
fh.split_data()


#Create Tf-Records for input data that will be used for training
train_data = TFRecord(data_dir = DATA_DIR,
                     xml_dir = str(Path(DATA_DIR, 'train')),
                     image_dir = str(Path(DATA_DIR, 'train')),
                     write_label_map = True
                     )

test_data = TFRecord(data_dir = DATA_DIR,
                     xml_dir = str(Path(DATA_DIR, 'test')),
                     image_dir = str(Path(DATA_DIR, 'test')),
                     write_label_map = False
                     )

print(str(TRAIN_SCRIPT))
print(str(run_dir))
print(str(model))

cmd = ['python', TRAIN_SCRIPT, 'logtostderr', 'train_dir=' + str(run_dir.absolute()), 'pipeline_config_path=' + str(model.absolute())]

test_cmd = ['python', TRAIN_SCRIPT, '--logtostderr', '--train_dir=' + str(run_dir.absolute()), '--pipeline_config_path=' + str(model.absolute())]
result = subprocess.Popen(cmd, stdout=subprocess.PIPE)
# result = subprocess.check_output(['python', TRAIN_SCRIPT, 'logtostderr', 'train_dir=' + str(run_dir), 'pipeline_config_path=' + str(model)])
print(result.stdout.read())
print(' '.join(test_cmd))
