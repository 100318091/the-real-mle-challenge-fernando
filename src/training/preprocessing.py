import pandas
import logging
import click

from src.config import (
    RAW_DATA_VARS,
    BINS,
    LABELS,
)

LOGGER = logging.getLogger()

### FUNCTIONS ###


def preprocessing_bathrooms(bathrooms_text: pandas.Series) -> pandas.Series:
    """
    Function that preprocess `bathrooms_text` column to extract the number
    of baths in float format.

    Unique values are: ['1 bath', nan, '1.5 baths', '1 shared bath', '1 private bath',
       'Shared half-bath', '2 baths', '1.5 shared baths', '3 baths',
       'Half-bath', '2.5 baths', '2 shared baths', '0 baths', '4 baths',
       '0 shared baths', 'Private half-bath', '5 baths', '4.5 baths',
       '5.5 baths', '2.5 shared baths', '3.5 baths', '3 shared baths',
       '4 shared baths', '6 baths', '3.5 shared baths',
       '4.5 shared baths', '7.5 baths', '6.5 baths', '8 baths', '7 baths',
       '6 shared baths']

    Objective: ['1.0', nan, '1.5', '1.0', '1.0',
       '0.5', '2.0', '1.5', '3.0',
       '0.5', '2.5', '2.0', '0.0', '4.0',
       '0.0', '0.5', '5.0', '4.5',
       '5.5', '2.5', '3.5', '3.0',
       '4.0', '6.0', '3.5',
       '4.5', '7.5', '6.5', '8.0', '7.0',
       '6.0']


    :param bathrooms_text: Pandas string series with `bathrooms_text` information
    :type bathrooms_text: pd.Series
    :return: Pandas string series with `bathrooms` information and Dtype `Float`.
    :rtype: pd.Series
    """

    return (
        bathrooms_text
        # first replace case sensitive `half-bath`` by .5
        .replace("[hH]alf-bath", "0.5", regex=True)
        # then extract float numbers using a regular expression pattern
        .str.extract(r"(?P<bathrooms>(\d+(\.\d+)?))", expand=False).bathrooms
        # finally convert them into numeric using the robust func `to_numeric`
        .astype(dtype=float)
    )


def preprocessing_target(
    price: pandas.Series,
) -> pandas.DataFrame:
    """
    Function that preprocess `price` column to extract the only the price without the dollar
    simbol and also get ride of outliers.

    Expected input: ['$150.00', '$75.00', '$60.00', '$275.00', '$68.00', '$98.00',
       '$89.00', '$65.00', '$62.00', '$90.00', '$199.00', '$96.00', '$1.00']

    Objective: ['150', '75', '60', '275', '68', '98',
       '89', '65', '62', '90', '199', '96']


    :param price: Pandas string series with `price` information
    :type price: pd.Series
    :return: Pandas string series with `price` information and Dtype `int`.
    :rtype: pd.Series

    .. warning:: strng values without numbers and Nan or None are not allowed.
    .. todo:: Must be verify that there is not weird value like: ´$´ or 'dollars' without a number.
    NaN values should not exist at this point.
    """
    return (
        price
        # preprocressing to get only the number not the dollar sign
        .str.extract(r"(\d+).", expand=False).astype(dtype=int)
        # outlier filtering below 10 dollars
        .pipe(lambda price: price[price >= 10])
    )


def preprocess_amenities_column(df: pandas.DataFrame) -> pandas.DataFrame:
    """
    Preprocess the amenities column of a pandas DataFrame and get dummy variables from it

    :param df: The input DataFrame containing the amenities column.
    :type df: pandas.DataFrame
    :return: The modified DataFrame with the amenities column preprocessed.
    :rtype: pandas.DataFrame

    This function preprocesses the amenities column of a pandas DataFrame. It adds new columns
    to the DataFrame indicating the presence or absence of specific amenities. The amenities
    column is then dropped from the DataFrame.

    The following amenities are considered:
    - TV
    - Wifi
    - Kitchen
    - Heating
    - Elevator
    - Internet
    - Breakfast
    - Air_conditioning

    Each amenity is represented by a new column in the DataFrame, with a value of 1 indicating the
    presence of the amenity and a value of 0 indicating the absence of the amenity.

    The input DataFrame is expected to have an amenities column, which is a string containing a list
    of amenities separated by commas.

    Example usage:
    >>> df = pd.DataFrame({'amenities': ['TV, Wifi, Kitchen', 'Heating, Elevator', 'Internet, Breakfast']})
    >>> preprocessed_df = preprocess_amenities_column(df)
    >>> print(preprocessed_df)

    Output:
    TV  Wifi  Kitchen  Heating  Elevator  Internet  Breakfast  Air_conditioning
    0   1     1        1        0         0         0          0
    1   0     0        0        1         1         0          0
    2   0     0        0        0         0         1          1

    """
    return (
        df.assign(TV=lambda x: x.amenities.str.contains("TV"))
        .assign(Wifi=lambda x: x.amenities.str.contains("Wifi"))
        .assign(Kitchen=lambda x: x.amenities.str.contains("Kitchen"))
        .assign(Heating=lambda x: x.amenities.str.contains("Heating"))
        .assign(Elevator=lambda x: x.amenities.str.contains("Elevator"))
        .assign(Internet=lambda x: x.amenities.str.contains("Internet"))
        .assign(Breakfast=lambda x: x.amenities.str.contains("Breakfast"))
        .assign(Air_conditioning=lambda x: x.amenities.str.contains("Air_conditioning"))
        .astype(
            {
                "TV": int,
                "Wifi": int,
                "Kitchen": int,
                "Heating": int,
                "Elevator": int,
                "Internet": int,
                "Breakfast": int,
                "Air_conditioning": int,
            }
        )
        .drop("amenities", axis=1)
    )


def preprocessing(
    raw_data: pandas.DataFrame,
    bins: list,
    labels: list,
) -> pandas.DataFrame:
    """
    Preprocesses the raw data by applying various transformations and cleaning steps.

    :param raw_data: Pandas DataFrame containing the raw data
    :type raw_data: pd.DataFrame
    :param bins: List of bin edges for creating categories based on price, defaults to [10, 90, 180, 400, numpy.inf]
    :type bins: list, optional
    :param labels: List of labels for the created categories, defaults to [0, 1, 2, 3]
    :type labels: list, optional
    :return: Preprocessed DataFrame with transformed and cleaned data
    :rtype: pd.DataFrame
    """

    return (
        raw_data
        # preprocess bathrooms
        .assign(bathrooms=lambda x: preprocessing_bathrooms(x.bathrooms_text))
        # work only with features
        .get(RAW_DATA_VARS)
        # rename columns
        .rename(columns={"neighbourhood_group_cleansed": "neighbourhood"})
        # preprocess target variable
        .assign(price=lambda x: preprocessing_target(x.price))
        # cleaning missing values
        .dropna(axis=0)
        # feature engineering: create clusters and asign categories
        .assign(category=lambda x: pandas.cut(x.price, bins=bins, labels=labels))
        # amenities dummies are not needed for the model
        # .pipe(preprocess_amenities_column)
    )


@click.command()
@click.option(
    "--input", help="Enter where the raw data csv file is allocated", type=str
)
@click.option(
    "--output", help="Enter where to store the ouput processed data", type=str
)
def main(input: str, output: str) -> None:
    """
    This is the main function of the program. It takes input and output file paths as command line
    arguments and processes the raw data by calling the preprocessing function. The processed data is then
    saved to the specified output file.

    :param input: The file path of the raw data csv file.
    :type input: str
    :param output: The file path where the processed data will be stored.
    :type output: str

    Example Usage:
    $ python processing.py --input raw_data.csv --output processed_data.csv

    """

    raw_data = pandas.read_csv(input)

    processed_df = preprocessing(
        raw_data,
        bins=BINS,
        labels=LABELS,
    )

    processed_df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
