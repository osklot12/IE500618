import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

def main():
    data = load_dataframe("../../datasets/diabetes/diabetes_binary_classification_data.csv")
    print(data.head())
    print(data.describe())
    print(data.info())

    histogram(data,[
        " BMI","GenHlth","Age","Education","Income"
    ])

def get_absolute_path(relative_path):
    base_path = os.path.dirname(__file__)
    absolute_path = os.path.join(base_path, relative_path)
    return absolute_path

def load_dataframe(path):
    return pd.read_csv(get_absolute_path(path))

def histogram(data, attributes):
    data[attributes].hist(figsize=(16, 12), bins=30, edgecolor='black')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()