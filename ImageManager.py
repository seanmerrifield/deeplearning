import FileManager
class ImageManager(FileManager):
    
    def __init__(self):
        self.super().__init__()

    def get_cifar(self):
        """
        Download CIFAR 10 Dataset
        : return:       Boolean
        """
        cifar10_dataset_folder_path = 'cifar-10-batches-py'

        # Use Floyd's cifar-10 dataset if present
        floyd_cifar10_location = '/input/cifar-10/python.tar.gz'
        if isfile(floyd_cifar10_location):
            tar_gz_path = floyd_cifar10_location
        else:
            tar_gz_path = 'cifar-10-python.tar.gz'

        # Retrieve file from url
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.get_from_url(url, tar_gz_path)

        # Extract file if it doesn't exist already
        if not isdir(cifar10_dataset_folder_path):
            self.tar_extract(tar_gz_path)

        return True

 