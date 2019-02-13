from pathlib import Path

if Path.cwd().name == 'object_detection':   root = '..'
else: root = '.'

#Path to Tensorflow API training script
API_DIR = Path(root, 'tensorflow/models/research/object_detection')
sys.path.append(str(API_DIR.parent.absolute()))
EXPORT_SCRIPT = str((API_DIR / 'export_inference_grpah.py').absolute())


#Ensure working directory is in object_detection folder
os.chdir(str(Path(root,'object_detection'),absolute()))


#Set Input Parameters
TRAIN_DIR = './training'
RUN_DIR = str(TRAIN_DIR, 'run_04')



