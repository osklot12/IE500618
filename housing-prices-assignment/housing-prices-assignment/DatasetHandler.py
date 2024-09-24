import numpy as np
import pandas as pd
import os

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

class DatasetHandler:
    def __init__(self, path):
        self.data = self._load_dataframe(path)
        self.train_set, self.test_set = self.__train_test_split(DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE)


    def numeric(self):
        return self.data.copy().select_dtypes(include=[np.number])


    def generate_train_test_sets(self, test_size, random_state):
        self.train_set, self.test_set = self.__train_test_split(test_size, random_state)


    def set_labels(self, label_column):
        self.train_features = self.train_set.drop(label_column, axis=1)
        self.train_labels = self.train_set[label_column]

        self.test_features = self.test_set.drop(label_column, axis=1)
        self.test_labels = self.test_set[label_column]


    def impute_median(self, median):
        imputer = SimpleImputer(strategy='median')
        imputer.fit(self.train_features)
        X = imputer.transform(housing_num)
        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


    def encode_ordinal(self):
        # Define the correct order of categories from worst to best
        kitchen_qual_order = ['Poor', 'Fair', 'Good', 'Excellent']

        # Initialize OrdinalEncoder with the correct order
        ordinal_encoder = OrdinalEncoder(categories=[kitchen_qual_order])

        # Fit and transform the "Kitchen Qual" feature
        # Assuming kitchen_qual_data is a DataFrame or array with the "Kitchen Qual" values
        transformed_kitchen_qual = ordinal_encoder.fit_transform(kitchen_qual_data)

        print(transformed_kitchen_qual)

    def __train_test_split(self, test_size, random_state):
        return train_test_split(self.data, test_size=test_size, random_state=random_state)


