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

4. Add the Tensorflow Object Detection API libraries to `PYTHONPATH`. 

Note that this command would have to be run everytime the terminal is started, but it can added to the `~/.bashrc` file, replacing pwd with an absolute path to deeplearning/tensorflow/models/research folder. 
```sh
# From deeplearning/tensorflow/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

5. Test that the Tensorflow installation worked. If so you're good to go!
```sh
python object_detection/builders/model_builder_test.py
```

