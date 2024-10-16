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
    RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, KBinsDiscretizer, StandardScaler, \
    MinMaxScaler
from imblearn.over_sampling import SMOTE


def main():
    # loading the data
    data = load_dataframe("../../datasets/diabetes/diabetes_binary_classification_data.csv")

    # splitting dataset (no peaking at the test set!)
    x_train, y_train, x_test, y_test = split_data(data)

    # exploring the dataset
    train_full_set = pd.concat([x_train, y_train], axis=1)
    explore_data(train_full_set)

    # dropping insignificant features
    x_train, x_test = drop_features(x_train, x_test)

    # oversampling minority class (data is highly imbalanced)
    x_train_resampled, y_train_resampled = oversample_data(x_train, y_train)

    # creating preprocessor
    preprocessor = create_preprocessing_pipeline()

    # preprocessing data
    x_train_processed, x_test_processed = preprocess(x_train_resampled, x_test, preprocessor)

    # getting feature names after preprocessing
    features = preprocessor.get_feature_names_out()

    # random forest classifier with hyperparameters from randomized search
    rfc = RandomForestClassifier(class_weight="balanced", n_estimators=200, min_samples_split=5, min_samples_leaf=1,
                                  max_features="sqrt", max_depth=None, bootstrap=True)
    print_model_benchmarks(rfc, x_train_processed, y_train_resampled, x_test_processed, y_test, features)

    # sgd classifier
    # model_benchmarks(SGDClassifier(random_state=42), x_train_processed, y_train)

    # logistic regression
    # log_reg = LogisticRegression(class_weight="balanced", random_state=42, max_iter=500, C=0.09)
    # model_benchmarks(log_reg, x_train_processed, y_train_resampled, features)


def drop_features(x_train, x_test):
    features_to_drop = [
        "MentHlth", "PhysHlth", "AnyHealthcare", "CholCheck", "Stroke", "NoDocbcCost"
    ]

    return x_train.drop(features_to_drop, axis=1), x_test.drop(features_to_drop, axis=1)


def print_model_benchmarks(model, x_train, y_train, x_test, y_test, features):
    # getting the model name
    model_name = model.__class__.__name__

    print("Started benchmarking for " + model_name + "...")
    print()

    # training the model
    model.fit(x_train, y_train)

    # getting predictions using the full training set
    y_train_full_pred = model.predict(x_train)

    # printing score analysis using training set validation
    print_score_analysis(y_train_full_pred, y_train, model_name, "train")

    # getting cross validation predictions
    y_train_pred = cross_val_predict(model, x_train, y_train, cv=3)

    # printing score analysis using cross validation
    print_score_analysis(y_train_pred, y_train, model_name, "cross")

    # getting test set predictions
    y_test_pred = model.predict(x_test)

    # printing score analysis for the test set (generalizing)
    print_score_analysis(y_test_pred, y_test, model_name, "test")

    # printing feature significance
    print_feature_sig(features, model, model_name)

    # getting decision scores
    # decision_scores = get_decision_scores(model, x_test, y_test)

    # plotting precision/recall for different thresholds
    # plot_pr(decision_scores, y_test, model_name)

    # plotting roc curve
    # plot_roc(decision_scores, y_test, model_name)

    # print_randomized_search(model, x_train, y_train) # used to find best hyperparameters


def custom_thresh_pred(model, data, threshold):
    pred_proba = model.predict_proba(data)[:, 1]
    return (pred_proba >= threshold).astype(int)

def feature_interact(train, test, interact_columns):
    interact_data = train[interact_columns]

    # using polynomial features for feature interaction
    interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction.fit(interact_data)

    # reconstructing the original data with the new feature
    train_interact = get_feature_interact_df(train, interact_columns, interaction)
    test_interact = get_feature_interact_df(test, interact_columns, interaction)

    return train_interact, test_interact


def get_feature_interact_df(data, interact_columns, interaction):
    interact_data = data[interact_columns]

    interaction_terms = interaction.transform(interact_data)
    interaction_features = interaction.get_feature_names_out(interact_data.columns)
    interaction_df = pd.DataFrame(interaction_terms, columns=interaction_features, index=data.index)
    data = data.drop(columns=interact_columns)

    return pd.concat([data, interaction_df], axis=1)


def oversample_data(x_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(x_train, y_train)


def get_coef_weights(model, features):
    coefficients = model.coef_[0]
    feature_weights_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": coefficients
    })

    # sorting the absolute values of the coefficients
    feature_weights_df["AbsCoefficient"] = feature_weights_df["Coefficient"].abs()
    return feature_weights_df.sort_values(by="AbsCoefficient", ascending=False)


def get_feature_importance(model, features):
    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": importance
    })

    return feature_importance.sort_values(by="Importance", ascending=False)


def explore_data(data):
    print_data_info(data)

    # histogram(data,[
    #     "BMI", "MentHlth", "PhysHlth"
    # ])


def print_randomized_search(model, features, labels):
    model_name = model.__class__.__name__

    param_dist = {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 20, 30, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    print("Starting randomized search for " + model_name + "...")
    random_search.fit(features, labels)

    best_params = random_search.best_params_
    print("Best Hyperparameters:", best_params)


def print_feature_sig(features, model, model_name):
    print(model_name + " Most Significant Features:")

    if hasattr(model, "coef_"):
        print(get_coef_weights(model, features))
    elif hasattr(model, "feature_importances_"):
        print(get_feature_importance(model, features))


def print_score_analysis(labels, pred, model_name, val_mode):
    cm = confusion_matrix(labels, pred)
    print_confusion_matrix(cm, val_mode, model_name)

    f1 = f1_score(labels, pred)
    print_f1_score(f1, val_mode, model_name)

    auc = roc_auc_score(labels, pred)
    print_auc_score(auc, val_mode, model_name)


def print_auc_score(auc, val_mode, model_name):
    print(model_name + " " + get_val_mode_str(val_mode) + " -> AUC:")
    print(str(auc) + "\n")


def print_f1_score(f1, val_mode, model_name):
    print(model_name + " " + get_val_mode_str(val_mode) +" -> F1 Score:")
    print(str(f1) + "\n")


def print_confusion_matrix(cm, val_mode, model_name):
    print(model_name + " " + get_val_mode_str(val_mode) + " -> Confusion Matrix:")
    print(str(cm) + "\n")


def get_val_mode_str(val_mode):
    result = ""

    if val_mode == "train":
        result = "Training Set Validation"
    elif val_mode == "cross":
        result = "Cross Validation"
    elif val_mode == "test":
        result = "Test Set Predictions"

    return result


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
        # ("health_bin", create_bin_pipeline(4), health_bin_features),
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
