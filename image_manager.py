from file_manager import FileManager
import numpy as np
import pickle
import os
from pathlib import Path

class ImageManager(FileManager):
    
    def __init__(self, path=None):
        super().__init__()

        if path: self.root = Path(os.getcwd(), path)
        else:   self.root = os.getcwd()


    def get_images(self, url, file_name, save_path=None):
        """
        Downloads image dataset from url
        : url:          url to download from
        : file_name:    file_name to download
        : save_path:    path to save dataset to
        : return:       Boolean
        """

        # Retrieve file from url
        self.get_from_url(url, file_name)

        # Extract file if it doesn't exist already
        if not save_path: save_path = self.root

        # Unpack contents
        self.unpack(file_name, save_path)

        return True

    def normalize(self, x):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : x: List of image data.
        : return: Numpy array of normalize data
        """
        normalized = x / (np.max(x) - np.min(x))
        return normalized

    def one_hot_encode(self, x):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
        """
        n_classes = 10
        labels = np.array(x)
        return np.eye(n_classes)[labels]

    def _preprocess_and_save(self, features, labels, filename):
        """
        Preprocess data and save it to file
        """
        features = self.normalize(features)
        labels = self.one_hot_encode(labels)

        pickle.dump((features, labels), open(filename, 'wb'))


    def preprocess_and_save_data(self, n_batches=5):
        """
        Preprocess Training and Validation Data
        """
        valid_features = []
        valid_labels = []

        for batch_i in range(1, n_batches + 1):
            features, labels = self.load_batch(self.root, batch_i)
            validation_count = int(len(features) * 0.1)

            # Prprocess and save a batch of training data
            self._preprocess_and_save(
                features[:-validation_count],
                labels[:-validation_count],
                'preprocess_batch_' + str(batch_i) + '.p')

            # Use a portion of training batch for validation
            valid_features.extend(features[-validation_count:])
            valid_labels.extend(labels[-validation_count:])

        # Preprocess and Save all validation data
        self._preprocess_and_save(
            np.array(valid_features),
            np.array(valid_labels),
            'preprocess_validation.p')

        with open(self.root + '/test_batch', mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        # load the test data
        test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        test_labels = batch['labels']

        # Preprocess and Save all test data
        self._preprocess_and_save(
            np.array(test_features),
            np.array(test_labels),
            'preprocess_test.p')

        return True

    def load_batch(self, dir_path, batch_id):
        """
        Load a batch of the dataset
        """
        with open(dir_path + '/data_batch_' + str(batch_id), mode='rb') as file:
            batch = pickle.load(file, encoding='latin1')

        features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = batch['labels']

        return features, labels
