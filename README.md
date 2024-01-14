# The NY Estimator Problem
In this challenge, we will explore the use of Airbnb listing data to predict the price category for new listings. We want to represent a real-case scenario where the MLE is working hand-to-hand with the Data Scientists at Intelygenz.

In this case, the data scientists have handed us a set of notebooks (in `lab/analysis`) that describe the ML workflow for data preprocessing and modelling. They have also included the dataset used and the trained model.

We will use these notebooks as a baseline to create more optimized functions that can be used in an ML inference pipeline.
# The MLE Challenge
You have to fork this repository to complete the following challenges in your own `github` account. Feel free to solve the challenge however you want.

Once completed, add a `SOLUTIONS.md` file justifying your responses and don't forget to send back the solution.

If you have any doubts or questions, don't hesitate to open an issue to ask any question about any challenge.

## Challenge 1 - Refactor DEV code

The code included in `lab` has been developed by Data Scientists during the development stage of the project. Now it is time to take their solution into production, and for that we need to ensure the code is up to standard and optimised. The first challenge is to refactor the code in `lab/analysis` the best possible way to operate in production.

Not only optimisation is important at this stage, but also the code should be written and tested in a way that can be easily understood by other MLE and tested at different CI stages.

## Challenge 2 - Build an API

The next step is to build an API that make use of the trained model to define the price category for a new listing. Here is an example of an input/output payload for the API.

```json
input = {
    "id": 1001,
    "accommodates": 4,
    "room_type": "Entire home/apt",
    "beds": 2,
    "bedrooms": 1,
    "bathrooms": 2,
    "neighbourhood": "Brooklyn",
    "tv": 1,
    "elevator": 1,
    "internet": 0,
    "latitude": 40.71383,
    "longitude": -73.9658
}

output = {
    "id": 1001,
    "price_category": "High"
}
```

The key is to ensure the API is easy to use and easy to test. Feel free to architect the API in any way you like and use any framework you feel comfortable with. Just ensure it is easy to make calls to the API in a local setting.

## Challenge 3 - Dockerize your solution

Nowadays, we can't think of ML solutions in production without thinking about Docker and its benefits in terms of standardisation, scalability and performance. The objective here is to dockerize your API and ensure it is easy to deploy and run in production.

## SOLUTION: 

### Project Repository Structure

The repository is organized as follows:

#### Container Folder
Contains the Dockerfile used for building the solution.

#### Data Folder
- **dvc Folder:** All inputs and outputs for DVC, enabling automatic training and dataset saving. Managed and tracked by DVC.
- **Processed and Raw Folders:** Used for the analysis conducted in Jupyter Notebooks.

#### Evaluation Folder
Houses the plots and reports generated after the training.

#### Models Folder
Reserved for the storage of trained models.

#### Src Folder
Contains all the code:
- **Serving Folder:** Includes the API Rest definition and data models for serving the model.
- **Training Folder:** Holds all the code needed for processing, training, and evaluating the model.
- **config.py:** General configuration of the model. Any variable related to the model is defined here. Please only make changes in this file.

#### Tests Folder
Stores all tests, along with sample data for testing purposes.

#### Other Files
- **dvc.lock:** A plain text file for tracking artifact versions.
- **dvc.yaml:** Definition of the training pipeline.
- **params.yaml:** DVC pipeline parameters.
- **poetry.lock:** Lock file with the Python package versions used by this project.
- **pyproject.toml:** Poetry main file.

### Installing

A step by step series of examples that tell you how to get a development environment running

The project include a poetry toml file. The only steps to follow are:

```bash

$ python3 -m pip install poetry==1.7.1
```

And then you can setup the environment using poetry:

```bash
# Configure poetry to create venv in project folder. You can skip this step but 
# remember the path of your venv to activate it later
$ poetry config virtualenvs.in-project true
# will create a venv but you can also choose to use conda env or any other solution
$ poetry env use python3.11 
Using virtualenv: XXXXXXX
# will install all dependencies using lock file. Feel free to recreate it if needed
$ poetry install
Installing the current project: the-real-mle-challenge (0.1.0)
# activate venv (this line is for `virtualenvs.in-project=true`)
$ source .venv/bin/activate
```
Once this is done, the environment should be already ready for running the project.

## Running the tests

```bash
$ pytest tests
======================================================================================================================= test session starts =======================================================================================================================
platform darwin -- Python 3.11.7, pytest-7.4.4, pluggy-1.3.0
rootdir: /Users/fernandohernandezgant/Projects/Interviews/Intellygenz/the-real-mle-challenge-fernando
plugins: anyio-4.2.0, hydra-core-1.3.2
collected 12 items                                                                                                                                                                                                                                                

tests/test_processing.py ...........                                                                                                                                                                                                                        [ 91%]
tests/test_training.py .                                                                                                                                                                                                                                    [100%]

======================================================================================================================= 12 passed in 32.43s =======================================================================================================================
```

## Running training

```bash
$ dvc repro
Running stage 'preprocessing':                                        
> python src/training/preprocessing.py --input data/dvc/raw/listings.csv --output data/dvc/processed/preprocessed_listings.csv
/Users/fernandohernandezgant/Projects/Interviews/Intellygenz/the-real-mle-challenge-fernando/src/training/preprocessing.py:215: DtypeWarning: Columns (67) have mixed types. Specify dtype option on import or set low_memory=False.
  raw_data = pandas.read_csv(input)
                                                                                                                                                                                                                                                                   
Running stage 'training':
> python src/training/fitting.py --input_data data/dvc/processed/preprocessed_listings.csv --output_train_test data/dvc/train_test --output_model models/simple_classifier.pkl
/Users/fernandohernandezgant/Projects/Interviews/Intellygenz/the-real-mle-challenge-fernando/.venv/lib/python3.11/site-packages/sklearn/base.py:1152: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
  return fit_method(estimator, *args, **kwargs)
                                                                                                                                                                                                                                                                   
Running stage 'evaluation':
> python src/training/evaluation.py --input_data data/dvc/train_test --input_model models/simple_classifier.pkl --importance_output_graphs evaluation/graphs/importance.png --output_reports_path evaluation/reports/metrics.json --output_reports_confusion_matrix evaluation/graphs/confusion_matrix.csv
Use `dvc push` to send your updates to remote storage.
```

## Deployment

```bash
# build docker image
$ docker build -f container/Dockerfile -t real-mle-challenge:v0.1.0 .
# run it
$ docker run -d --name real-mle-challenge-fernando -p 80:80 real-mle-challenge:v0.1.0
```

## Interaction with the solution

### Only deploy the API locally 

```bash
# build docker image
$ uvicorn src.serving.api:app
INFO:     Started server process [43712]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [43712]
```

### Using docker container

After deploying the container and while it is running:

Go to [API DOC](http://localhost/redoc) for API Documentation

Go to [API Inter](http://localhost/docs) for API Interaction and test

## Built With

* [Uvicorn](https://www.uvicorn.org) - The ASGI web server
* [FastAPI](https://fastapi.tiangolo.com) - Build REST API
* [DVC](https://dvc.org) - Build ML pipelines and version tracking

## TODO

* CICD - GitHub actions - Improve
* Improve Code
* Improve Model Registry
* Include more tests
* Create Kubernetes Deployment
* Add scaling options and monitoring
... 

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Fernando Hernandez** - *Fork* - [the-real-mle-challenge-fernando](https://github.com/100318091/the-real-mle-challenge-fernando/tree/main)
* **Intellygenz** - *Initial work* - [the-real-mle-challenge](https://github.com/intelygenz/the-real-mle-challenge)