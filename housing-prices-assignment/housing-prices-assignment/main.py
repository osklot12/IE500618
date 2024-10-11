import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures


def main():
    # library config
    pd.set_option("max_info_columns", 1000)
    pd.set_option("display.max_rows", None)

    # loading data
    data = load_dataframe("../../datasets/ames-housing/AmesHousing.csv")

    # dropping irrelevant features
    data = data.drop(columns=["Order", "PID"], errors="ignore")

    # splitting dataset
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    housing = train_set.drop(columns=["SalePrice"], axis=1)
    housing_labels = train_set["SalePrice"]

    # accessing specific features
    log_features = get_log_features()
    nominal_features = get_nominal_features()
    ordinal_features, ordinal_categories = get_ordinal_features_categories()

    # creating preprocessing pipeline
    preprocessing = create_preprocessing_pipeline(log_features, nominal_features, ordinal_categories, ordinal_features)

    # predicting using linear regression
    # linear_reg(housing, housing_labels, preprocessing)

    # predicting using polynomial features
    # polynomial_reg(housing, housing_labels, preprocessing)

    # predicting using random forest regressor
    random_forest_reg(housing, housing_labels, preprocessing)


def create_preprocessing_pipeline(log_features, nominal_features, ordinal_categories, ordinal_features):
    log_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler()
    )

    root_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.sqrt, feature_names_out="one-to_one"),
        StandardScaler()
    )

    default_num_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    onehot_pipeline = make_pipeline(
        (SimpleImputer(strategy="constant", fill_value="No")),
        (OneHotEncoder(handle_unknown="ignore")),
    )

    ordinal_pipeline = make_pipeline(
        (SimpleImputer(strategy="constant", fill_value="No")),
        (OrdinalEncoder(categories=ordinal_categories)),
    )

    return ColumnTransformer([
        ("onehot", onehot_pipeline, nominal_features),
        ("ordinal", ordinal_pipeline, ordinal_features),
        ("log", log_pipeline, log_features),
    ],
        remainder=default_num_pipeline)


def random_forest_reg(housing, housing_labels, preprocessing):
    forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
    forest_reg.fit(housing, housing_labels)
    forest_predictions = forest_reg.predict(housing)
    forest_rmse = root_mean_squared_error(housing_labels, forest_predictions)
    print("Random forest regression RMSE on training data:\n", forest_rmse)
    forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    print("Random forest regression RMSE using cross validation:\n", pd.Series(forest_rmses).describe())


def polynomial_reg(housing, housing_labels, preprocessing):
    degree = 2
    pol_reg = make_pipeline(preprocessing, PolynomialFeatures(degree), LinearRegression())
    pol_reg.fit(housing, housing_labels)
    pol_predictions = pol_reg.predict(housing)
    pol_rmse = root_mean_squared_error(housing_labels, pol_predictions)
    print("Polynomial feature regression RSME on training data:\n", pol_rmse)
    pol_rmses = -cross_val_score(pol_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    print("Polynomial regression RMSE using cross validation:\n", pd.Series(pol_rmses).describe())


def linear_reg(housing, housing_labels, preprocessing):
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(housing, housing_labels)
    lin_predictions = lin_reg.predict(housing)
    lin_rmse = root_mean_squared_error(housing_labels, lin_predictions)
    print("Linear regression RSME on training data:\n", lin_rmse)
    linear_rmses = -cross_val_score(lin_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    print("Linear regression RMSE using cross validation:\n", pd.Series(linear_rmses).describe())


def get_log_features():
    return ["Gr Liv Area", "Lot Frontage", "Lot Area"]


def get_nominal_features():
    return [
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
    ]


def get_ordinal_features_categories():
    return ([
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
                "Paved Drive"
            ],
            [
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


def print_info(data):
    print(data.info())


def corr(data, target_feature):
    return numeric_data(data).corr()[target_feature].sort_values(ascending=False)


def numeric_data(data):
    return data.copy().select_dtypes(include=[np.number])


def load_dataframe(path):
    return pd.read_csv(get_absolute_path(path))


def get_absolute_path(relative_path):
    base_path = os.path.dirname(__file__)
    absolute_path = os.path.join(base_path, relative_path)
    return absolute_path


def histogram(data, attributes):
    data[attributes].hist(bins=50, figsize=(20, 20))
    plt.show()


if __name__ == "__main__":
    main()

# Test
# Test 2
# Test 3