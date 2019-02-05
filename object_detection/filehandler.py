from pathlib import Path
from random import shuffle
import shutil
import pandas as pd
import xml.etree.ElementTree as ET

class FileHandler:
    extensions = ['jpg', 'png', 'jpeg']

    TRAIN = 'train'
    TEST = 'test'
    def __init__(self, root_dir, image_dir, xml_dir, train_size=0.8):
        assert Path(root_dir).exists(), 'Root directory does not exist at {}'.format(root_dir)
        assert Path(image_dir).exists(), 'Directory containing images does not exist at {}'.format(image_dir)
        assert Path(xml_dir).exists(), 'Directory containing labelled object detection XML files does not exist at {}'.format(xml_dir)

        self.root = root_dir
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.train_size = train_size

        #Get image and xml data
        self.images = self.get_images()
        self.data = self.get_data(self.images)
        assert len(self.images) > 0, 'Could not find images at {}'.format(self.image_dir)
        assert len(self.data) > 0, 'Could not find associated xml data for images at {}'.format(self.xml_dir)

        #Setup training and testing directories
        self.train_dir = str(Path(self.root, self.TRAIN))
        self.test_dir = str(Path(self.root, self.TEST))
        self.clean(self.train_dir)
        self.clean(self.test_dir)




    def clean(self, path):
        dir = Path(path)
        if dir.exists(): 
            for fp in dir.glob('*'): fp.unlink()
            dir.rmdir()

        dir.mkdir()

    def get_images(self):
        
        collection = []
        for ext in self.extensions:
            images = [str(path) for path in Path(self.image_dir).glob('*.{}'.format(ext))]
            collection += images
        return collection

    def get_data(self, image_paths):
        xmls = []
        for img in image_paths:
            path = Path(img)
            file_name = path.stem + '.xml'
            xml_path = Path(self.xml_dir, file_name)
            if xml_path.exists(): 
                xmls.append( (img, str(xml_path)) )
        return xmls


    def xml_to_csv(self, path):
        xml_list = []
        for xml_file in Path(path).glob('*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text)
                        )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)

        xml_df.to_csv((path + '_labels.csv'), index=None)

    

    def split(self, data, split):
        limit = int(len(data)*split)
        return data[:limit], data[limit:]

    def copy_files(self, data, dest_path):
        for image_path, xml_path in data:
            shutil.copy(image_path, str(Path(dest_path, Path(image_path).name)) )
            shutil.copy(xml_path, str(Path(dest_path, Path(xml_path).name)) )

    def split_data(self):
        
        shuffle(self.data)
        #Split data into training and test sets
        train_data, test_data = self.split(self.data, self.train_size)
        
        #Copy files over
        self.copy_files(train_data, self.train_dir)
        self.copy_files(test_data, self.test_dir)
        
        self.xml_to_csv(self.train_dir)
        self.xml_to_csv(self.test_dir)


if __name__ == "__main__":

    root_dir = './object_detection/data'
    image_path = str(Path(root_dir, 'images'))
    xml_path = str(Path(root_dir, "xml"))

    fh = FileHandler(root_dir, image_path, xml_path)
    fh.split_data()