# Deep Learning
Deep Learning is a collection of projects I have made using deep neural networks built on Tensorflow. [Tensorflow](https://www.tensorflow.org/api_docs/) is a deeplearning API built by Google.

## Installation

`miniconda` is a great python package management system which can be downloaded [here](https://conda.io/miniconda.html). Install this so that you can run `conda` commands from the command line. 

Run the following commands from the command line terminal to download this repository and install the necessary python packages:

1. Clone repository and repository submodules
```sh
git clone https://github.com/seanmerrifield/deeplearning
cd deeplearning
git submodule update --init
```


2. Create a new `conda` environment
* **Linux or Mac**
```sh
conda create -n deeplearning python=3.6
source activate deeplearning
```

* **Windows**
```sh
conda create -n deeplearning python=3.6
activate deeplearning
```

3. All dependent packages are installed from the requirements text file (including Tensorflow).
```sh
pip install -r requirements.txt
```


5. And that's it!


