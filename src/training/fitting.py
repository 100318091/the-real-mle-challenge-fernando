import click
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from src.config import (
    FEATURES,
    TARGET_VAR,
    N_ESTIMATORS,
    RANDOM_STATE,
    CLASS_WEIGHT,
    N_JOBS,
    MAP_ROOM_TYPE,
    MAP_NEIGHB,
)
import cloudpickle as pickle
from pathlib import Path
import sys


def room_type_and_neigh_mapping(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Maps the values of the 'neighbourhood' and 'room_type' columns in the given DataFrame
    to their corresponding values in the 'MAP_NEIGHB' and 'MAP_ROOM_TYPE' dictionaries, respectively.

    :param df: The DataFrame containing the 'neighbourhood' and 'room_type' columns.
    :type df: pandas.DataFrame
    :return: The DataFrame with the 'neighbourhood' and 'room_type' columns mapped to their corresponding values.
    :rtype: pandas.DataFrame
    """
    return df.assign(neighbourhood=lambda x: x.neighbourhood.map(MAP_NEIGHB)).assign(
        room_type=lambda x: x.room_type.map(MAP_ROOM_TYPE)
    )


@click.command()
@click.option(
    "--input_data",
    help="Enter where the processed data csv file is allocated",
    type=str,
)
@click.option(
    "--output_model", help="Enter where to store the ouput model artefact", type=str
)
@click.option(
    "--output_train_test",
    help="Enter where to store the ouput train/test sets",
    type=str,
)
def main(input_data: str, output_train_test: str, output_model: str) -> None:
    """
    Main function for training a random forest classifier.

    :param input_data: The file path of the processed data csv file.
    :type input_data: str
    :param output_train_test: The directory path to store the output train/test sets.
    :type output_train_test: str
    :param output_model: The file path to store the output model artifact.
    :type output_model: str
    :return: None
    :rtype: None
    """

    preprocessed_data = pandas.read_csv(input_data)

    X = preprocessed_data[FEATURES]
    y = preprocessed_data[TARGET_VAR]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=1
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight=CLASS_WEIGHT,
        n_jobs=N_JOBS,
    )
    # Create a transformer function to apply to the input data also while serving
    feat_engineering = FunctionTransformer(room_type_and_neigh_mapping)
    # Create a pipeline with the feature engineering transformer and the random forest classifier
    model_pipeline = Pipeline([("feature_engineering", feat_engineering), ("clf", clf)])
    # Fit the pipeline to the training data
    model_pipeline.fit(X_train, y_train)

    # Create the output directory for the train/test sets if it does not exists
    output_train_test_path = Path(output_train_test)
    output_train_test_path.mkdir(parents=True, exist_ok=True)

    # Save the training and testing sets as CSV files in the output directory
    X_train.to_csv(Path(output_train_test) / "X_train.csv", index=False)
    y_train.to_csv(Path(output_train_test) / "y_train.csv", index=False)
    X_test.to_csv(Path(output_train_test) / "X_test.csv", index=False)
    y_test.to_csv(Path(output_train_test) / "y_test.csv", index=False)

    # Save model along with the custom function used in `FunctionTransformer`
    # using `cloudpickle``
    with open(output_model, "wb") as model_dump:
        pickle.register_pickle_by_value(sys.modules[__name__])
        pickle.dump(model_pipeline, model_dump)


if __name__ == "__main__":
    main()
