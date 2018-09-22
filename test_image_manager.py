import unittest
from image_manager import ImageManager
from pathlib import Path

class TestImageManager(unittest.TestCase):

    im = ImageManager(path = 'input')

    def test_download(self):

        URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        FILE = 'cifar-10-python.tar.gz'

        self.assertTrue(self.im.get_images(URL, FILE))

    def test_preprocess(self):

        n_batches = 5
        self.assertTrue(self.im.preprocess_and_save_data(n_batches))


