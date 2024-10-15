from random import randint

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_curve, \
    roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, \
    RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, KBinsDiscretizer, StandardScaler, \
    MinMaxScaler


def main():
    data = load_dataframe("../../datasets/diabetes/diabetes_binary_classification_data.csv")

    # exploring the dataset
    explore_data(data)

    # combining heart condition features into HeartHealth
    # data["HeartHealth"] = data[["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke"]].sum(axis=1)
    # data["HeartHealth"] = data["HeartHealth"].apply(lambda x: 1 if x > 0 else 0)

    # creating a HealthyLifeStyle feature
    # data["HealthyLifestyle"] = data[["Fruits", "Veggies", "PhysActivity"]].sum(axis=1) - data["HvyAlcoholConsump"]

    #data = data.drop(columns=['HighBP', 'HighChol', 'HeartDiseaseorAttack', 'Stroke',
    #                          'Fruits', 'Veggies', 'PhysActivity', 'HvyAlcoholConsump'])

    # splitting dataset
    x_train, y_train, x_test, y_test = split_data(data)

    # creating preprocessor
    preprocessor = create_preprocessing_pipeline()

    # preprocessing data
    x_train_processed, x_test_processed = preprocess(x_train, x_test, preprocessor)

    x_train_processed_df = pd.DataFrame(x_train_processed, columns=preprocessor.get_feature_names_out())
    print(x_train_processed_df.info())

    # creating and training sgd classifier
    # model_benchmarks(SGDClassifier(random_state=42), x_train_processed, y_train)
    # model_benchmarks(
    #    RandomForestClassifier(class_weight="balanced", random_state=42, max_depth=10),
    #     x_train_processed, y_train)
    model_benchmarks(LogisticRegression(class_weight="balanced", random_state=42, max_iter=500, C=0.9),
                     x_train_processed, y_train)


def explore_data(data):
    print_data_info(data)

    # histogram(data,[
    #     "BMI", "MentHlth", "PhysHlth"
    # ])


def model_benchmarks(model, x_train, y_train):
    # getting the model name
    model_name = model.__class__.__name__

    print("Started benchmarking for " + model_name + "...")
    print()
    # training the model
    model.fit(x_train, y_train)

    # getting predictions using the full training set
    y_train_full_pred = model.predict(x_train)

    cm = confusion_matrix(y_train, y_train_full_pred)
    print(model_name + " Full Set Validation -> Confusion Matrix:")
    print(cm)
    print()

    f1 = f1_score(y_train, y_train_full_pred)
    print(model_name + " Full Set Validation -> F1 Score:")
    print(f1)
    print()

    auc = roc_auc_score(y_train, y_train_full_pred)
    print(model_name + " Full Set Validation -> AUC:")
    print(auc)
    print()

    # getting cross validation predictions
    y_train_pred = cross_val_predict(model, x_train, y_train, cv=3)

    cm = confusion_matrix(y_train, y_train_pred)
    print(model_name + " Cross Validation -> Confusion Matrix:")
    print(cm)
    print()

    f1 = f1_score(y_train, y_train_pred)
    print(model_name + " Cross Validation -> F1 Score:")
    print(f1)
    print()

    auc = roc_auc_score(y_train, y_train_pred)
    print(model_name + " Cross Validation -> AUC:")
    print(auc)
    print()

    # getting decision scores
    decision_scores = get_decision_scores(model, x_train, y_train)

    # plotting precision/recall for different thresholds
    plot_pr(decision_scores, y_train, model_name)

    # plotting roc curve
    plot_roc(decision_scores, y_train, model_name)


def get_decision_scores(model, x_train, y_train):
    if hasattr(model, "decision_function"):
        return cross_val_predict(model, x_train, y_train, cv=3, method="decision_function")
    elif hasattr(model, "predict_proba"):
        y_proba = cross_val_predict(model, x_train, y_train, cv=3, method="predict_proba")
        return y_proba[:, 1]
    else:
        raise AttributeError(f"{model.__class__.__name__} has neither decision_function nor predict_proba method.")


def plot_roc(decision_scores, y_train, model_name=""):
    # getting false positive rate and true positive rate for different threshold values
    fpr, tpr, thresholds = roc_curve(y_train, decision_scores)

    plt.figure(figsize=(10, 10))

    plt.plot(fpr, tpr, linewidth=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], "k:", label="Random classifier's ROC curve")

    plt.title(model_name + " ROC curve", fontsize=16)
    plt.ylabel("True Positive Rate (Recall)", fontsize=14)
    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=14)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pr(decision_scores, y_train, model_name=""):
    # getting precision and recall for different threshold values
    precisions, recalls, thresholds = precision_recall_curve(y_train, decision_scores)

    plt.figure(figsize=(10, 10))

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.title(model_name + " Precision and Recall vs Decision Thresholds", fontsize=16)
    plt.ylabel("Precision / Recall", fontsize=14)
    plt.xlabel("Decision Threshold", fontsize=14)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_data_info(data):
    # label of the dataset
    label = "Diabetes_binary"

    print("Dataset information:")
    print(data.info())
    print()
    print("Balance of label classes:")
    print(get_balance(data, label))
    print()
    print("Correlation between each feature and the label:")
    print(get_corr(data, label))


def get_balance(data, label):
    # checks the balance of the dataset by counting the different label classes
    return data[label].value_counts()


def get_corr(data, target):
    return corr_matrix(data)[target].sort_values(ascending=False)


def preprocess(x_train, x_test, preprocessing):
    return preprocessing.fit_transform(x_train), preprocessing.transform(x_test)


def create_preprocessing_pipeline():
    # selecting features
    log_features = ["BMI"]
    health_bin_features = ["MentHlth", "PhysHlth"]
    min_max_features = ["Age", "Education", "Income"]
    interaction_features = ["MentHlth", "PhysHlth"]

    # creating complete preprocessing pipeline
    return ColumnTransformer([
        ("log", create_log_pipeline(), log_features),
        ("health_bin", create_bin_pipeline(4), health_bin_features),
        ("minmax", MinMaxScaler(), min_max_features)
        # ("interaction", create_interaction_transformer(), interaction_features)
    ],
        remainder="passthrough")


def create_interaction_transformer():
    return PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)


def create_log_pipeline():
    return make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler()
    )


def create_binner(n_bins):
    return KBinsDiscretizer(
        n_bins=n_bins,
        encode="onehot-dense",
        strategy="uniform"
    )


def create_bin_pipeline(n_bins):
    return make_pipeline(
        create_binner(n_bins),
        MinMaxScaler()
    )


def split_data(data):
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    train_features = train_set.drop(columns=["Diabetes_binary"], axis=1)
    test_features = test_set.drop(columns=["Diabetes_binary"], axis=1)

    train_labels = train_set["Diabetes_binary"]
    test_labels = test_set["Diabetes_binary"]

    return train_features, train_labels, test_features, test_labels


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
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Heatmap of Selected Features')
# plt.show()

if __name__ == "__main__":
    main()
