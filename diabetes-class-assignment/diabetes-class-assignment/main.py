import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

def main():
    data = load_dataframe("../../datasets/diabetes/diabetes_binary_classification_data.csv")
    print(data.head())
    print(data.describe())
    print(data.info())

    histogram(data,[
        "BMI","GenHlth","Age","Education","Income"
    ])
    print(corr_matrix(data)["Diabetes_binary"].sort_values(ascending=False))
    heatmap(corr_matrix(data))

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


def heatmap(data):
    plt.figure(figsize=(20, 20))
    sns.heatmap(data, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.show()


def corr_matrix(data):
    return data.corr()

# Create a heatmap of correlations
#sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#plt.title('Correlation Heatmap of Selected Features')
#plt.show()

if __name__ == "__main__":
    main()