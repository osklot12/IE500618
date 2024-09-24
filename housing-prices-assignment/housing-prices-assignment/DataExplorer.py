from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix


class DataExplorer:
    def __init__(self, data_handler_):
        self.data_handler = data_handler_


    def info(self):
        return self.data_handler.data.info()


    def cat_count(self, category):
        return self.data_handler.data()[category].value_counts()


    def corr(self, target_feature):
        return self.data_handler.numeric().corr()[target_feature].sort_values(ascending=False)


    def scatter(self, attributes):
        scatter_matrix(self.data_handler.data()[attributes], figsize=(20, 20))
        plt.show()