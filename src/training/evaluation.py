import click
import pandas
import cloudpickle as pickle
from pathlib import Path
from sklearn.metrics import classification_report
import json
import numpy
from matplotlib import pyplot as plt
from src.config import TARGET_MAPS


def plot_feature_importance(importances, features, output_dir):
    """
    Plot the feature importance bar chart.

    :param importances: The feature importances.
    :type importances: array-like
    :param features: The names of the features.
    :type features: array-like
    :param output_dir: The directory to save the plot.
    :type output_dir: str
    :return: None.

    Example:
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import pandas as pd
    >>> import seaborn as sns
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> clf = RandomForestClassifier()
    >>> clf.fit(X, y)
    >>> importances = clf.feature_importances_
    >>> features = iris.feature_names
    >>> output_dir = "output/"
    >>> plot_feature_importance(importances, features, output_dir)
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.barh(range(len(importances)), importances)
    plt.yticks(range(len(importances)), features, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance", fontsize=12)

    plt.savefig(output_dir)


def save_confusion_matrix(y_test, y_pred, output_file) -> None:
    """
    Save the confusion matrix to a CSV file.

    Parameters
    :param y_test: The true labels of the test data.
    :type y_test: array-like of shape (n_samples,)
    :param y_pred: The predicted labels of the test data.
    :type y_pred: array-like of shape (n_samples,)
    :param output_file: The path to save the confusion matrix CSV file.
    :type output_file: str
    :return: None

    .. notes:: The confusion matrix is saved as a CSV file with the following columns:
    - actual_class: The true class labels of the test data, mapped to their corresponding class names.
    - predicted_class: The predicted class labels of the test data, mapped to their corresponding class names.

    The class labels are mapped using the following dictionary:
    {0: "low", 1: "mid", 2: "high", 3: "lux"}

    Example
    -------
    >>> y_test = [0, 1, 2, 3, 0, 1, 2, 3]
    >>> y_pred = [0, 1, 2, 3, 1, 2, 3, 0]
    >>> output_file = "confusion_matrix.csv"
    >>> save_confusion_matrix(y_test, y_pred, output_file)
    The confusion matrix is saved as "confusion_matrix.csv" in the current directory.
    """
    plot_info = (
        pandas.DataFrame()
        .assign(actual_class=y_test)
        .assign(predicted_class=y_pred)
        .assign(actual_class=lambda x: x.actual_class.map(TARGET_MAPS))
        .assign(predicted_class=lambda x: x.predicted_class.map(TARGET_MAPS))
    )
    plot_info.to_csv(output_file, index=False)


@click.command()
@click.option(
    "--input_data",
    help="Enter where the processed data csv file is allocated",
    type=str,
)
@click.option(
    "--input_model", help="Enter where to store the ouput model artefact", type=str
)
@click.option(
    "--importance_output_graphs",
    help="Enter where the processed data csv file is allocated",
    type=str,
)
@click.option(
    "--output_reports_path",
    help="Enter where to store the ouput model artefact",
    type=str,
)
@click.option(
    "--output_reports_confusion_matrix",
    help="Enter where to store the ouput model artefact",
    type=str,
)
def main(
    input_data: str,
    input_model: str,
    importance_output_graphs: str,
    output_reports_path: str,
    output_reports_confusion_matrix: str,
) -> None:
    """
    Main Function

    This function is the entry point of the program. It takes several command line arguments and performs the following tasks:

    1. Reads the input data from the specified file paths.
    2. Loads the trained model from the specified file path.
    3. Predicts the labels for the test data using the loaded model.
    4. Computes the classification report, including accuracy, precision, recall, and F1-score.
    5. Saves the classification report as a JSON file.
    6. Computes the feature importances using the trained model.
    7. Plots the feature importance bar chart and saves it as an image file.
    8. Computes the confusion matrix using the predicted and true labels.
    9. Saves the confusion matrix as a CSV file.

    :param input_data: The file path of the processed data CSV file.
    :type input_data: str
    :param input_model: The file path to store the output model artifact.
    :type input_model: str
    :param importance_output_graphs:  The file path to store the output feature importance graph.
    :type importance_output_graphs: str
    :param output_reports_path: The file path to store the output classification report.
    :type output_reports_path: str
    :param output_reports_confusion_matrix: The file path to store the output confusion matrix.
    :type output_reports_confusion_matrix: str
    :return: None.

    .. notes::
    - The input_data should contain three CSV files: X_test.csv, X_train.csv, and y_test.csv.
    - The input_model should be a pickled trained model.
    - The importance_output_graphs should be a directory path to store the feature importance graph.
    - The output_reports_path should be a directory path to store the classification report.
    - The output_reports_confusion_matrix should be a file path to store the confusion matrix.
    """

    # convert string to ´Path´
    input_data_path = Path(input_data)
    output_reports_path = Path(output_reports_path)
    output_graphs_path = Path(importance_output_graphs)

    # Read train/test sets
    X_test = pandas.read_csv(input_data_path / "X_test.csv")
    X_train = pandas.read_csv(input_data_path / "X_train.csv")
    y_test = pandas.read_csv(input_data_path / "y_test.csv")

    # Read model
    with open(input_model, "rb") as pickle_model:
        pipeline = pickle.load(pickle_model)

    # Predict on test set
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    # Compute overall accuracy
    report = classification_report(y_test, y_pred, output_dict=True)
    # Save report as json
    with open(output_reports_path, "w") as report_file:
        json.dump(report, report_file)

    # Compute feature_importance
    importances = pipeline["clf"].feature_importances_
    indices = numpy.argsort(importances)[::-1]
    features = X_train.columns[indices]
    importances = importances[indices]
    # Plot feature_importance
    plot_feature_importance(
        importances=importances, features=features, output_dir=output_graphs_path
    )
    # Compute and save confusion matrix as DVC plot template
    save_confusion_matrix(
        y_test=y_test, y_pred=y_pred, output_file=output_reports_confusion_matrix
    )


if __name__ == "__main__":
    main()
