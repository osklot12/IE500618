from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def main():
    # library config
    pd.set_option('max_info_columns', 1000)
    pd.set_option('display.max_rows', None)

    # loading data
    data = load_dataframe("../../datasets/ames-housing/AmesHousing.csv")

    # dropping irrelevant features
    data = data.drop(columns=["Order", "PID"], errors="ignore")

    # splitting dataset
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    housing = train_set.drop("SalePrice", axis=1)
    housing_labels = train_set["SalePrice"]

    # imputing median on null values for numerics
    imputed_num = impute_numerics(housing, "median")

    # imputing on null values for categorical features
    imputed_cat = impute_categories(imputed_num, [
        ["Alley", "constant", "No"],
        ["Mas Vnr Type", "constant", "No"],
        ["Bsmt Qual", "constant", "No"],
        ["Bsmt Cond", "constant", "No"],
        ["Bsmt Exposure", "constant", "None"],
        ["Bsmt Cond", "constant", "No"],
        ["BsmtFin Type 1", "constant", "No"],
        ["BsmtFin Type 2", "constant", "No"],
        ["Fireplace Qu", "constant", "No"],
        ["Garage Type", "constant", "No"],
        ["Garage Finish", "constant", "No"],
        ["Garage Qual", "constant", "No"],
        ["Garage Cond", "constant", "No"],
        ["Pool QC", "constant", "No"],
        ["Fence", "constant", "No"],
        ["Misc Feature", "constant", "No"],
    ])

    # ordinal encoding
    encoded_ordinal = encode_ordinal(imputed_cat, [
        "Kitchen Qual",
        "Land Slope",
        "Fireplace Qu",
        "Garage Finish",
        "Garage Qual",
        "Garage Cond",
        "Pool QC",
        "Exter Qual",
        "Exter Cond",
        "Bsmt Qual",
        "Bsmt Cond",
        "Bsmt Exposure",
        "BsmtFin Type 1",
        "BsmtFin Type 2",
        "Heating QC",
        "Central Air",
        "Functional",
        "Paved Drive",

    ], [
        ["Po", "Fa", "TA", "Gd", "Ex"],
        ["Gtl", "Mod", "Sev"],
        ["No", "Po", "Fa", "TA", "Gd", "Ex"],
        ["No", "Unf", "RFn", "Fin"],
        ["No", "Po", "Fa", "TA", "Gd", "Ex"],
        ["No", "Po", "Fa", "TA", "Gd", "Ex"],
        ["No", "Fa", "TA", "Gd", "Ex"],
        ["Po", "Fa", "TA", "Gd", "Ex"],
        ["Po", "Fa", "TA", "Gd", "Ex"],
        ["No", "Po", "Fa", "TA", "Gd", "Ex"],
        ["No", "Po", "Fa", "TA", "Gd", "Ex"],
        ["None", "No", "Mn", "Av", "Gd"],
        ["No", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        ["No", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
        ["Po", "Fa", "TA", "Gd", "Ex"],
        ["N", "Y"],
        ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
        ["N", "P", "Y"]
                                     ])

    # one hot encoding
    encoded_1hot = encode_one_hot(encoded_ordinal, [
        "MS SubClass", "MS Zoning",
        "Lot Shape", "Alley", "Street",
        "Land Contour", "Utilities",
        "Lot Config", "Fence",
        "Misc Feature", "Neighborhood",
        "Condition 1", "Condition 2",
        "Bldg Type", "House Style",
        "Roof Style", "Roof Matl",
        "Exterior 1st", "Exterior 2nd",
        "Mas Vnr Type", "Foundation",
        "Heating", "Electrical",
        "Garage Type", "Sale Type",
        "Sale Condition"

    ])

    # calculate correlations with preprocessed data
    processed_with_labels = pd.concat([encoded_1hot, housing_labels], axis=1)
    corr_matrix = processed_with_labels.corr()
    saleprice_corr = corr_matrix["SalePrice"].sort_values(ascending=False)
    print(saleprice_corr)


def impute_numerics(data, strategy):
    data_copy = data.copy()

    imputer = SimpleImputer(strategy=strategy)
    numerics = numeric_data(data_copy)
    imputed = imputer.fit_transform(numerics)
    imputed_df = pd.DataFrame(imputed, columns=numerics.columns, index=numerics.index)
    data_copy[numerics.columns] = imputed_df[imputer.get_feature_names_out()]

    return data_copy


def impute_categories(data, imputations):
    data_copy = data.copy()

    for column, strategy, *rest in imputations:
        if strategy == "constant":
            fill_value = rest[0] if rest else 'Missing'
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=strategy)

        imputed = imputer.fit_transform(data_copy[[column]])
        data_copy[column] = imputed.ravel()

    return data_copy


def encode_one_hot(data, columns):
    data_copy = data.copy()

    cat = data_copy[columns]
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_array = encoder.fit_transform(cat).toarray()
    encoded_feature_names = encoder.get_feature_names_out(columns)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=data_copy.index)
    data_copy = data_copy.drop(columns, axis=1)
    data_copy = pd.concat([data_copy, encoded_df], axis=1)

    return data_copy


def encode_ordinal(data, columns, cat_orders):
    data_copy = data.copy()

    cat = data_copy[columns]
    encoder = OrdinalEncoder(categories=cat_orders)
    encoded_array = encoder.fit_transform(cat)
    encoded_df = pd.DataFrame(encoded_array, columns=columns, index=data_copy.index)
    data_copy[columns] = encoded_df

    return data_copy


def array_to_dataframe(array, dataframe):
    return pd.DataFrame(array, columns=dataframe.columns, index=dataframe.index)


def numeric_data(data):
    return data.copy().select_dtypes(include=[np.number])


def categorical_data(data):
    return data.select_dtypes(include=["object"])


def cat_count(data, category):
    return data[category].value_counts()


def corr(data, target_feature):
    return numeric_data(data).corr()[target_feature].sort_values(ascending=False)


def scatter(data, attributes):
    scatter_matrix(data[attributes], figsize=(20, 20))
    plt.show()


def load_dataframe(path):
    return pd.read_csv(get_absolute_path(path))


def get_absolute_path(relative_path):
    base_path = os.path.dirname(__file__)
    absolute_path = os.path.join(base_path, relative_path)
    return absolute_path


if __name__ == "__main__":
    main()
