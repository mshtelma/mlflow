from typing import Dict, Any

from mlflow.exceptions import MlflowException
from .stratified_split import StratifiedSplitStep
from .train_automl import AutoMLTrainStep
from ..evaluate import EvaluateStep


class EvaluateStepExt(EvaluateStep):
    def __init__(self, step_config: Dict[str, Any], pipeline_root: str):
        super().__init__(step_config, pipeline_root)
        self.task_type = step_config.get("model_task_type")
        if self.task_type is None or self.task_type not in ["regression", "classification"]:
            raise MlflowException(
                "Please define 'model_task_type' step parameter! Accepted values are 'regression' or 'classification'"
            )
