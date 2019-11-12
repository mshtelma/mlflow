import pytest
import numpy as np
import keras
import keras.layers as layers

import mlflow
import mlflow.keras
from distutils.version import LooseVersion

client = mlflow.tracking.MlflowClient()


@pytest.fixture
def random_train_data():
    return np.random.random((1000, 32))


@pytest.fixture()
def fit_variant():
    return "fit"


@pytest.fixture
def keras_random_data_run(random_train_data, fit_variant):
    mlflow.keras.autolog()

    def random_one_hot_labels(shape):
        n, n_class = shape
        classes = np.random.randint(0, n_class, n)
        labels = np.zeros((n, n_class))
        labels[np.arange(n), classes] = 1
        return labels

    with mlflow.start_run() as run:
        data = random_train_data
        labels = random_one_hot_labels((1000, 10))

        model = keras.Sequential()

        model.add(layers.Dense(64, activation='relu', input_shape=(32,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(lr=0.001, epsilon=1e-07),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        if fit_variant == 'fit_generator':
            def generator():
                while True:
                    yield data, labels
            model.fit_generator(generator(), epochs=10, steps_per_epoch=1)
        else:
            model.fit(data, labels, epochs=10)

    return client.get_run(run.info.run_id)


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_generator'])
def test_keras_autolog_logs_expected_data(keras_random_data_run):
    data = keras_random_data_run.data
    assert 'accuracy' in data.metrics
    assert 'loss' in data.metrics
    assert 'optimizer_name' in data.params
    assert data.params['optimizer_name'] == 'Adam'
    assert 'epsilon' in data.params
    assert data.params['epsilon'] == '1e-07'
    assert 'model_summary' in data.tags
    assert 'Total params: 6,922' in data.tags['model_summary']
    artifacts = client.list_artifacts(keras_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model_summary.txt' in artifacts


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_generator'])
def test_keras_autolog_model_can_load_from_artifact(keras_random_data_run, random_train_data):
    artifacts = client.list_artifacts(keras_random_data_run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    model = mlflow.keras.load_model("runs:/" + keras_random_data_run.info.run_id +
                                    "/model")
    model.predict(random_train_data)
