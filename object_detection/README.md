# Object Detection
Object detection can be used to locate objects within images. This is done using Tensorflow's [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). 

The Tensorflow Object Detection API installation instructions are replicated here for convenience.

## Installation

1. Follow the installation instructions at the root of this repository if you haven't already. 

2. The Object Detection API was automatically added to this repository as a submodule. First update the submodule to ensure the latest version, and change local directory to the API folder
    ```sh
    git submodule update
    cd tensorflow/models/research
    ```

3. The Object Detection API uses protobufs to configure model and training parameters. The protobufs are compiled using the following commands:

* **Linux**
    ```sh
    wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
    unzip protobuf.zip
    ./bin/protoc object_detection/protos/*.proto --python_out=.
    ```

* **Mac** - Install protobufs using [homebrew](https://brew.sh/)
    ```sh
    brew install protobuf
    protoc object_detection/protos/*.proto --python_out=.
    ```
* **Windows** - For Windows the protobuf installation is a bit more involved, they need to be installed and created manually. First download the latest release of the protobuf Python installer from [here](https://github.com/protocolbuffers/protobuf/releases/download/v3.7.0/protoc-3.7.0-win64.zip). Unzip the downloaded file and then run the following commands:
    
    ```sh
    #From the folder where protobuf.exe is located
    set PATH=$PATH;%cd%
    
    #Test that able to run the protoc command is working by running
    protoc --version
    ```
    
    Now change directory to `deeplearning\tensorflow\models\research` and run the following command to create all protobufs:
    ```sh
    #From deeplearning\tensorflow\models\research
    protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
    ```
    
4. Add the Tensorflow Object Detection API libraries to `PYTHONPATH`. For Linux or Mac this command would have to be run everytime the terminal is started, but it can added to the `~/.bashrc` file, replacing pwd with an absolute path to `deeplearning/tensorflow/models/research` folder. 
* **Linux or Mac**
    ```sh
    # From deeplearning/tensorflow/models/research
    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```
* **Windows**
    ```sh
    # From deeplearning/tensorflow/models/research
    setx PYTHONPATH=$PYTHONPATH;%cd%;%cd%\slim
    ```
5. Test that the Tensorflow installation worked. If so you're good to go!
    ```sh
    python object_detection/builders/model_builder_test.py
    ```

## Training

In order to train an object detector, the API expects the input data to be in Tensorflow's tf-record format. The following steps show how an object detection model can be trained on any set of images.  

1. Label images with objects to detect. This can be easily done with software like [LabelImg](https://github.com/tzutalin/labelImg). LabelImg produces an xml file for each image that describes the bounding box and classification of each object. 

2. Copy the images and the xml files to the `data` directory. The images and xml files should be placed in `data/images` and `data/xml`, respectively. 

3. The final step is to adjust the parameters for training the model. This is all defined through a config file located in `models/faster_rcnn_inception_v2_coco_2018_01_28.config`. The object detection algorithm uses Faster RCNN pre-trained on the COCO image dataset. In the config file, adjust the following fields:
- num_classes: The number of object classes to train
- fine_tune_checkpoint: Absolute path where the pre-trained model checkpoint file is located stored (ex: `~/deeplearning/object_detection/models/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt`)
- Adjust the train_input_reader file paths to `~/deepleraning/object_detection/data/train.record` and `~/deeplearning/object_detection/data/label_map.pbtxt`.

    
5. Run the training using the `train.py` file. This will start the training of the object detection model in the `training` directory. 
    ```sh
    python train.py
    ```
    
6. Track the training progress and performance using tensorboard
    ```sh
    tensorboard --logdir=training
    ```
    
7. Save the model
    ```
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
    ```
