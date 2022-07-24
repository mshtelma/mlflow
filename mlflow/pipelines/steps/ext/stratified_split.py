import importlib
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from mlflow.pipelines.cards import BaseCard
from mlflow.pipelines.step import BaseStep
from mlflow.pipelines.steps.split import (
    SplitStep,
    _OUTPUT_TRAIN_FILE_NAME,
    _OUTPUT_VALIDATION_FILE_NAME,
    _OUTPUT_TEST_FILE_NAME,
)
from mlflow.pipelines.utils.execution import get_step_output_path


class StratifiedSplitStep(SplitStep):
    def _run(self, output_directory):
        run_start_time = time.time()

        ingested_data_path = get_step_output_path(
            pipeline_root_path=self.pipeline_root,
            step_name="ingest",
            relative_path="dataset.parquet",
        )
        input_df = pd.read_parquet(ingested_data_path)
        train_ratio, validation_ratio, test_ratio = self.split_ratios
        new_val_ratio = validation_ratio / (validation_ratio + test_ratio)

        train_df, rest_df = train_test_split(
            input_df, train_size=train_ratio, stratify=input_df[self.target_col], random_state=42
        )
        validation_df, test_df = train_test_split(
            rest_df, train_size=new_val_ratio, stratify=rest_df[self.target_col], random_state=42
        )
        # Import from user function module to process dataframes
        post_split_config = self.step_config.get("post_split_method", None)
        if post_split_config is not None:
            (post_split_module_name, post_split_fn_name) = post_split_config.rsplit(".", 1)
            sys.path.append(self.pipeline_root)
            post_split = getattr(
                importlib.import_module(post_split_module_name), post_split_fn_name
            )
            (train_df, validation_df, test_df) = post_split(train_df, validation_df, test_df)

        # Output train / validation / test splits
        train_df.to_parquet(os.path.join(output_directory, _OUTPUT_TRAIN_FILE_NAME))
        validation_df.to_parquet(os.path.join(output_directory, _OUTPUT_VALIDATION_FILE_NAME))
        test_df.to_parquet(os.path.join(output_directory, _OUTPUT_TEST_FILE_NAME))

        self.run_end_time = time.time()
        self.execution_duration = self.run_end_time - run_start_time
        return self._build_profiles_and_card(train_df, validation_df, test_df)

    @classmethod
    def from_pipeline_config(cls, pipeline_config, pipeline_root):
        step_config = pipeline_config.get("steps", {}).get("split", {})
        step_config["target_col"] = pipeline_config.get("target_col")
        return cls(step_config, pipeline_root)

    @property
    def name(self):
        return "split"
