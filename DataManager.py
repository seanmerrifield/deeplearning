import pandas as pd


class DataManager:

    def __init(self):
        return self

    def read_csv(self, url):
        """
        Read csv file into a pandas dataframe
        : return: Dataframe of csv file.
        """
        self.data = pd.read_csv(url)
        return self.data

    def inspect(self):
        '''
        Return only top rows of data frame
        :return: Dataframe of top rows
        '''
        return self.data.head()

    def columns(self):
        '''
        Return a list of column names in dataframe
        :return: list of column names
        '''
        return list(self.data)

    def get(self, field):
        return self.data[field]

    def set_categories(self, fields=[]):
        """
        Convert field list to be converted into one hot encoded fields
        :param fields: list of fields (column headers) to convert to categories
        : return: Dataframe with one hot encoded fields
        """

        for category in fields:
            dummies = pd.get_dummies(self.data[category], prefix=category, drop_first=False)
            self.data = pd.concat([self.data, dummies], axis=1)

        # Drop old fields
        self.drop(fields)

        return self.data

    def set_targets(self, fields=[]):
        """
        Specify which fields should be set as targets
        :param fields: field list of targets
        :return: tuple of features dataframe and targets dataframe
        """
        self.features, self.targets = self.data.drop(fields, axis=1), self.data[fields]
        return self.features, self.targets

    def drop(self, fields=[]):
        """
        Specify fields to drop
        :param fields: list of fields (column headers) to drop
        :return: Dataframe with dropped fields
        """
        self.data = self.data.drop(fields, axis=1)
        return self.data

    def split(self, train_ratio=0.8, val_ratio=0.5):
        """
        Splits dataset into training, validation, and test sets
        :param train_ratio: proportion of dataset dedicated for training
        :param val_ratio: proportion of dataset dedicated for validation
        :return: triple dataframe of training, validation, and test sets
        """
        self.train = self.data[train_ratio * self.data.count:]
        self.test = self.data[(1 - train_ratio) * self.data.count:]

        self.val = self.data[val_ratio * self.test.count]
        self.test = self.data[(1 - val_ratio) * self.test.count]

        return self.train, self.val, self.test

    def normalize(self, fields=[]):
        """
        Normalize fields such that they have zero mean and a standard deviation of 1
        :param fields: list of fields (column headers) to normalize
        :return: Dataframe with normalized fields
        """
        scaled_features = {}
        for field in fields:
            mean, std = self.data[field].mean(), self.data[field].std()
            scaled_features[field] = [mean, std]
            self.data.loc[:, field] = (self.data[field] - mean) / std

        return self.data
