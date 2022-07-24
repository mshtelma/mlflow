import logging
import os
import sys

from cloudpickle import cloudpickle

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.steps.train import TrainStep
from mlflow.pipelines.utils.execution import get_step_output_path
from mlflow.pipelines.utils.metrics import _load_custom_metric_functions
from mlflow.pipelines.utils.tracking import apply_pipeline_tracking_config, log_code_snapshot

_logger = logging.getLogger(__name__)

AUTOML_DEFAULT_TIME_BUDGET = 2


class AutoMLTrainStep(TrainStep):
    def __init__(self, step_config, pipeline_root):
        super().__init__(step_config, pipeline_root)
        self.task_type = step_config.get("model_task_type")
        if self.task_type is None or self.task_type not in ["regression", "classification"]:
            raise MlflowException(
                "Please define 'model_task_type' step parameter! Accepted values are 'regression' or 'classification'"
            )

    def _run(self, output_directory):
        import pandas as pd
        import shutil
        from sklearn.pipeline import make_pipeline
        from mlflow.models.signature import infer_signature

        apply_pipeline_tracking_config(self.tracking_config)

        transformed_training_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="transform",
            relative_path="transformed_training_data.parquet",
        )
        train_df = pd.read_parquet(transformed_training_data_path)
        X_train, y_train = train_df.drop(columns=[self.target_col]), train_df[self.target_col]

        transformed_validation_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="transform",
            relative_path="transformed_validation_data.parquet",
        )
        validation_df = pd.read_parquet(transformed_validation_data_path)

        raw_training_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="split",
            relative_path="train.parquet",
        )
        raw_train_df = pd.read_parquet(raw_training_data_path)
        raw_X_train = raw_train_df.drop(columns=[self.target_col])

        raw_validation_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="split",
            relative_path="validation.parquet",
        )
        raw_validation_df = pd.read_parquet(raw_validation_data_path)

        transformer_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="transform",
            relative_path="transformer.pkl",
        )

        sys.path.append(self.pipeline_root)
        mlflow.autolog(disable=True)
        estimator = self._create_model_automl(X_train, y_train, self.task_type)
        mlflow.autolog(log_models=False)

        with mlflow.start_run() as run:
            estimator.fit(X_train, y_train)

            # if hasattr(estimator, "best_score_"):
            #     mlflow.log_metric("best_cv_score", estimator.best_score_)
            # if hasattr(estimator, "best_params_"):
            #     mlflow.log_params(estimator.best_params_)

            # TODO: log this as a pyfunc model
            # code_paths = [] #[os.path.join(self.pipeline_root, "steps")]
            estimator_schema = infer_signature(X_train, estimator.predict(X_train.copy()))
            logged_estimator = mlflow.sklearn.log_model(
                estimator,
                f"{self.name}/estimator",
                signature=estimator_schema,
                # code_paths=code_paths,
            )
            # Create a pipeline consisting of the transformer+model for test data evaluation
            with open(transformer_path, "rb") as f:
                transformer = cloudpickle.load(f)
            mlflow.sklearn.log_model(
                transformer,
                "transform/transformer",
            )
            model = make_pipeline(transformer, estimator)
            mlflow.autolog(disable=True)
            model_schema = infer_signature(raw_X_train, model.predict(raw_X_train.copy()))
            mlflow.autolog(log_models=False)
            model_info = mlflow.sklearn.log_model(
                model,
                f"{self.name}/model",
                signature=model_schema,
            )
            output_model_path = get_step_output_path(
                pipeline_root_path=self.pipeline_root,
                step_name=self.name,
                relative_path="model",
            )
            if os.path.exists(output_model_path) and os.path.isdir(output_model_path):
                shutil.rmtree(output_model_path)
            mlflow.sklearn.save_model(model, output_model_path)

            with open(os.path.join(output_directory, "run_id"), "w") as f:
                f.write(run.info.run_id)
            log_code_snapshot(self.pipeline_root, run.info.run_id)

            eval_metrics = {}
            for dataset_name, dataset in {
                "training": train_df,
                "validation": validation_df,
            }.items():
                eval_result = mlflow.evaluate(
                    model=logged_estimator.model_uri,
                    data=dataset,
                    targets=self.target_col,
                    model_type="regressor" if self.task_type == "regression" else "classifier",
                    evaluators="default",
                    dataset_name=dataset_name,
                    custom_metrics=_load_custom_metric_functions(
                        self.pipeline_root,
                        self.evaluation_metrics.values(),
                    ),
                    evaluator_config={
                        "log_model_explainability": False,
                    },
                )
                eval_result.save(os.path.join(output_directory, f"eval_{dataset_name}"))
                eval_metrics[dataset_name] = eval_result.metrics

        target_data = raw_validation_df[self.target_col]
        prediction_result = model.predict(raw_validation_df.drop(self.target_col, axis=1))
        pred_and_error_df = pd.DataFrame(
            {
                "target": target_data,
                "prediction": prediction_result,
                "error": prediction_result - target_data,
            }
        )
        train_predictions = model.predict(raw_train_df.drop(self.target_col, axis=1))
        worst_examples_df = BaseStep._generate_worst_examples_dataframe(
            raw_train_df, train_predictions, self.target_col
        )
        leaderboard_df = None
        try:
            leaderboard_df = self._get_leaderboard_df(run, eval_metrics)
        except Exception as e:
            _logger.warning("Failed to build model leaderboard due to unexpected failure: %s", e)

        card = self._build_step_card(
            eval_metrics=eval_metrics,
            pred_and_error_df=pred_and_error_df,
            model=model,
            model_schema=model_schema,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            worst_examples_df=worst_examples_df,
            leaderboard_df=leaderboard_df,
        )
        card.save_as_html(output_directory)
        for step_name in ("ingest", "split", "transform", "train"):
            self._log_step_card(run.info.run_id, step_name)

        return card

    def _create_model_automl(self, X, y, task_type):
        try:
            from flaml import AutoML

            automl_settings = {
                "time_budget": self.step_config.get(
                    "automl_time_budget_minutes", AUTOML_DEFAULT_TIME_BUDGET * 60
                ),
                "estimator_list": ["rf", "extra_tree", "lgbm"],
                "task": task_type,
            }
            automl = AutoML()
            automl.fit(X, y, **automl_settings)
            model = automl.model.estimator
        except ModuleNotFoundError:
            raise MlflowException(
                "Please define either train function or install FLAML to use AutoML!"
            )
        return model
