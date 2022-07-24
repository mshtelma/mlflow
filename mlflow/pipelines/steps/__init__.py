import importlib
import yaml


def make_execute_step(
    step_name: str, step_config_path: str, pipeline_root: str, output_directory: str
):
    with open(step_config_path, "r") as f:
        step_config = yaml.safe_load(f)
    step_uses = step_config.get("uses")
    if step_uses is not None:
        step_module_name, step_class_name = step_uses.rsplit(".", 1)
        step_class = getattr(importlib.import_module(step_module_name), step_class_name)
        step_class.from_step_config_path(
            step_config_path=step_config_path, pipeline_root=pipeline_root
        ).run(output_directory=output_directory)
    else:
        raise Exception("No uses defined!")
