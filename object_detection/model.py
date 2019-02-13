import os
import tensorflow as tf
import tarfile
import six.moves.urllib as urllib
from pathlib import Path

class Model:

    url = 'http://download.tensorflow.org/models/object_detection/'
    GRAPH_NAME = 'frozen_inference_graph.pb'

    def __init__(self, model_dir, model_name, n_classes):
        #Create model directory if it doesn't exist yet
        if not Path(model_dir).exists: Path(model_dir).mkdir(parents=True, exist_ok=True)

        self.model_dir = model_dir
        self.model_name = model_name
        self.n_classes = n_classes

        self.graph_path = str(Path(self.model_dir, self.GRAPH_NAME))
        
    def get_graph(self):
        #If model doesn't exist, download model from tensorflow api
        if not Path(self.graph_path).exists(): 
            self.download_model()
            
        return self.load_graph()

    def load_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                sess = tf.Session(graph = detection_graph)
        
            return detection_graph, sess

    def download_model(self):

        model_file = self.model_name + '.targz'
        model_path = str(Path(self.model_dir, self.model_file))

        #Get model from URL
        url = self.url + self.model_file
        opener = urllib.request.URLopener()
        opener.retrieve(url, model_path)

        #Uncompress file
        tar_file = tarfile.open(model_path)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, self.model_dir)



if __name__ == "__main__":
    MODEL_DIR = str(Path.cwd() / 'models')
    MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'

    model = Model(MODEL_DIR, MODEL_NAME, n_classes=5)
    graph, sess = model.get_graph()