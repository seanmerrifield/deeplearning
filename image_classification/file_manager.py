from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import tarfile

class FileManager:

    def __init__(self):
        return None

    def get_from_url(self,url,file_name, desc=None):
        """
        Download a file from provided url path
        : url:          URL string of file to download
        : file_name:    File name to download
        : desc          File description
        : return:       Tensor for image input.
        """

        class DLProgress(tqdm):
            last_block = 0

            def hook(self, block_num=1, block_size=1, total_size=None):
                self.total = total_size
                self.update((block_num - self.last_block) * block_size)
                self.last_block = block_num

        if not isfile(file_name):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc=desc) as pbar:
                urlretrieve(
                    url,
                    file_name,
                    pbar.hook)

    def tar_extract(self,file_path, save_path):

        with tarfile.open(file_path) as tar:
            tar.extractall(path=save_path)
            tar.close()

        return True

    def unpack(self, file_path, save_path):
        #Tar extract
        if '.tar' in file_path: self.tar_extract(file_path, save_path)

        #Todo: Finish unzip function
        if '.zip' in file_path: pass
        return True