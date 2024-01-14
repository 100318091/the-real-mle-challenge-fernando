FROM python:3.11-slim
HEALTHCHECK NONE
ENV PYTHONUNBUFFERED=1
# Model path
ENV APP_MODEL_PATH=/opt/ml/model/model.pkl
# Working dir. Code will be in $APP_WRK_DIR/src
ENV APP_WRK_DIR=/opt/ml/app

# create ml_user
RUN useradd --user-group ml_user

# copy necessary files with owner ml_user
COPY --chown=ml_user:ml_user models/simple_classifier.pkl $APP_MODEL_PATH
COPY --chown=ml_user:ml_user src/config.py $APP_WRK_DIR/src/config.py
COPY --chown=ml_user:ml_user src/serving $APP_WRK_DIR/src/serving
COPY --chown=ml_user:ml_user pyproject.toml $APP_WRK_DIR/pyproject.toml
COPY --chown=ml_user:ml_user poetry.lock $APP_WRK_DIR/poetry.lock

# working dir will be $APP_WRK_DIR
WORKDIR $APP_WRK_DIR

USER root
RUN python3 -m pip install poetry
RUN poetry config virtualenvs.create false
# Install python dependencies
RUN poetry install --no-interaction --with serving --without training,dev,test

# use ml_user always as it is the reference user for ml tasks within the docker
USER ml_user

EXPOSE 80

CMD ["uvicorn", "src.serving.api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
