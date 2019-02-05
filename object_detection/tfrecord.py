from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import glob
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


import xml.etree.ElementTree as ET

class TFRecord:

    columns = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    def __init__(self, data_dir, xml_dir, image_dir, write_label_map=False):
        assert os.path.exists(data_dir), "Data directory doesn't exist at: " + data_dir
        assert os.path.exists(xml_dir), "Data directory doesn't exist at: " + xml_dir
        assert os.path.exists(image_dir), "Data directory doesn't exist at: " + image_dir

        self.data_dir = data_dir
        self.xml_dir = xml_dir
        self.image_dir = image_dir

        #Get Labelled object data
        self.csv_file = self._xml_to_csv(self.xml_dir)
        self.class_list, self.class_dict = self._get_classes(self.csv_file)

        #Write TF Record
        record_name = os.path.splitext(self.csv_file)[0] + '.record'
        self._write_record(self.csv_file, self.image_dir, record_name)

        #Create Label Map
        if write_label_map:
            self._create_label_map(self.class_list, self.data_dir)

    def _xml_to_csv(self, xml_dir):
        """
        Converts all Pascal XML files to CSV format
        :param xml_dir:     Directory path to XML files
        :return:            Returns path to written CSV is successful, None if failed
        """

        # Parse XML data
        xml_list = []
        for xml_file in glob.glob(xml_dir + '/*.xml'):
            value = self._read_pascal_xml(xml_file)
            xml_list.extend(value)

        # Write data to CSV
        if len(xml_list) > 0:
            df = pd.DataFrame(xml_list, columns=self.columns)
            csv_path = xml_dir + '.csv'
            df.to_csv(csv_path, index=None)
            return csv_path


    def _read_pascal_xml(self, path):
        """
        Parses a Pascal XML file
        :param path: Path to XML file
        :return:    list of parsed data
        """
        result = []
        tree = ET.parse(path)
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
            result.append(value)
        return result

    def _get_classes(self, csv_file):
        """
        Gets all labelled classes in dataset
        :param csv_file:    Path to CSV file with class labels
        :return:            List of class label & id tuples, dlass label to ID dictionary
        """
        classes = []

        df = pd.read_csv(csv_file)
        classes.extend(df['class'])

        #Get unique classes
        classes = list(set(df['class']))

        #Return list of Class - ID tuples, and Class-ID dict
        return [ (id,c) for id,c in enumerate(classes)], {c:id for id,c in enumerate(classes)}

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def _write_record(self, csv_file, image_dir, record_name):
        """
        Writes a TF Record
        :param csv_file:    CSV File containing Labelled object data
        :param record_name: Name of TF Record file
        :return:            None
        """

        writer = tf.python_io.TFRecordWriter(record_name)
        df = pd.read_csv(csv_file)
        grouped = self.split(df, 'filename')

        for group in grouped:
            tf_example = self._create_record(group, image_dir)
            if tf_example == None: continue
            writer.write(tf_example.SerializeToString())

        writer.close()


    def _create_record(self, group, path):
        """
        Creates a tf record object with data and path to related image
        :param group:   Labelled object data
        :param path:    Path to image directory
        :return:        TF Record object
        """
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        if image == None or image.size == 0: return None
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpeg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))

            classes.append(self.class_dict[row['class']])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example


    def _create_label_map(self, class_list, save_dir):
        with open(os.path.join(save_dir, 'label_map.pbtxt'), 'wb+') as f:
            for id, c in class_list:
                f.write('item: {\n'.encode('utf-8'))
                f.write('\tid: {}\n'.format(id + 1).encode('utf-8'))
                f.write('\tname: \'{}\'\n'.format(c).encode('utf-8'))
                f.write('}\n\n'.encode('utf-8'))

if __name__ == '__main__':
    data_dir = './data'
    image_dir = os.path.join(data_dir,'train')
    xml_dir = os.path.join(data_dir, 'train')
    tf_record = TFRecord(data_dir, xml_dir, image_dir, write_label_map=True)
