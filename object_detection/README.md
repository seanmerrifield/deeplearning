# Object Detection
Object detection can be used to locate objects within images. This is done using Tensorflow's [Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). 

The Tensorflow Object Detection API installation instructions are replicated here for convenience.

##Installation

1. Follow the installation instructions at the root of this repository if you haven't already. 

2. Change local directory to the object_detection folder
```sh
cd object_detection 
```

3. The Tensorflow API uses Protobufs to configure model and training parameters. The protobufs are compiled using the following commands:

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

4. Add the Tensorflow Object Detection API libraries to the `PYTHONPATH`
```sh
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

5. Test that the Tensorflow installation worked. If so you're good to go!
```sh
python object_detection/builders/model_builder_test.py
```

