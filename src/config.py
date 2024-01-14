import numpy

### PREPROCESSING VARS AND DEFAULT VALUES ###
# with the first bin to 10, apartments at 10 dollar cost are not
# assign to any cluster
RAW_DATA_VARS = [
    "id",
    "neighbourhood_group_cleansed",
    "property_type",
    "room_type",
    "latitude",
    "longitude",
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "amenities",
    "price",
]
FEATURES = ["neighbourhood", "room_type", "accommodates", "bathrooms", "bedrooms"]
TARGET_VAR = ["category"]
BINS = [0, 90, 180, 400, numpy.inf]
LABELS = [0, 1, 2, 3]
# Categorical variable mapping dictionaries
MAP_ROOM_TYPE = {
    "Shared room": 1,
    "Private room": 2,
    "Entire home/apt": 3,
    "Hotel room": 4,
}
MAP_NEIGHB = {
    "Bronx": 1,
    "Queens": 2,
    "Staten Island": 3,
    "Brooklyn": 4,
    "Manhattan": 5,
}
TARGET_MAPS = {0: "low", 1: "mid", 2: "high", 3: "lux"}

### MODEL HYPERPARAMETERS ###
N_ESTIMATORS = 500
RANDOM_STATE = 0
CLASS_WEIGHT = "balanced"
N_JOBS = 4
