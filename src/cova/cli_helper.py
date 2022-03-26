# -*- coding: utf-8 -*-

import json
import logging
from pathlib import Path
from typing import Dict


from cova.pipeline.pipeline import COVAAutoTune


logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")


def parse_config(config_file: str) -> Dict:
    """Parses config file with pipeline definition.

    Args:
        config_file (str): path to the config file (json format) with the pipeline configuration.

    Returns:
        dict: dictionary containing all configuration parsed from the config file.
    """
    with open(config_file, "r") as config_fn:
        config = json.load(config_fn)
    config_str = Path(config_file).read_text()

    global_definitions = config.pop("globals", None)
    if global_definitions is None:
        return config

    single_stage = global_definitions.get("single_stage", "")
    stage_config = global_definitions.get("stage_params", [])

    for key, value in global_definitions.items():
        subst_str = "$globals#{}".format(key)
        if subst_str in config_str:
            if value is None:
                value = ""
            config_str = config_str.replace(subst_str, value)

    config = json.loads(config_str)
    _ = config.pop("globals", None)
    return config, [single_stage, stage_config]


def _run(config_file: str) -> None:
    """Runs the pipeline defined in the config file.

    Args:
        config_file (str): path to the config file (json format) with the pipeline configuration.
    """
    config, (single_stage, stage_config) = parse_config(config_file)

    auto_tuner = COVAAutoTune()
    auto_tuner.load_pipeline(config, single_stage)

    if single_stage == "":
        auto_tuner.run()
    else:
        logger.info(
            "Executing single stage %s with parameters %s",
            single_stage,
            ", ".join(stage_config),
        )
        auto_tuner.run_stage(single_stage, stage_config)
