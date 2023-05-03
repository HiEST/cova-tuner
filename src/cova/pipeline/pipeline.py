import importlib.util
import inspect
import logging
import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, NewType, Optional

logger = logging.getLogger(__name__)

Args = NewType("Args", dict[str, Any])
Stage = NewType("Stage", tuple[str, Args])

PIPELINE = ["capture", "filter", "annotate", "dataset", "train"]
CONSTRUCTORS = ["COVACapture", "COVAFilter", "COVAAnnotate", "COVADataset", "COVATrain"]


class COVAFactory:
    """Factory class to load and get COVA plugins."""

    def __init__(self):
        self._plugins_by_class = {}
        self._plugins_by_module = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.join(current_dir, "plugins")
        self.load_plugins(plugins_dir)

    @staticmethod
    def _detect_class(module) -> Optional[tuple[Callable, str]]:
        """Detects the class the plugin implements and returns its constructor and its parent class."""
        for member in inspect.getmembers(module, inspect.isclass):
            # Has a COVA parent?
            for mro in reversed(member[1].mro()[1:]):
                if mro.__name__ in CONSTRUCTORS:
                    return member[1]

        return None

    @staticmethod
    def _load_plugin(plugin_file: str) -> Callable[..., Any]:
        """Loads a plugin from plugin_file containing the implementation of class_name class."""
        module_name = Path(plugin_file).stem
        spec = importlib.util.spec_from_file_location(module_name, plugin_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        constructor = COVAFactory._detect_class(module)
        if constructor is None:
            logger.warning("Could not load plugin from module %s.", module.__name__)
        else:
            logger.info(
                "Loaded plugin %s from module %s.",
                constructor.__name__,
                module.__name__,
            )
        return constructor, module.__name__

    def load_plugins(self, plugins_path: str) -> None:
        """Loads the plugins found in the plugins_path."""
        if os.path.isdir(plugins_path):
            plugins = [str(p) for p in Path(plugins_path).rglob("*.py")]
        else:
            plugins = [plugins_path]

        for plugin_file in plugins:
            try:
                plugin, module_name = COVAFactory._load_plugin(plugin_file)
            except ModuleNotFoundError:
                continue
            # Check that no other plugin had the same name.
            conflict_by_class = False
            try:
                assert self._plugins_by_class.get(plugin.__name__, None) is None
            except AttributeError:
                continue
            except AssertionError:
                conflict_by_class = True
                msg = (
                    f"Conflict in plugins by class name: {plugin.__name__} is duplicated. "
                    + "Previous will be shadowed."
                )
                logger.warning(msg)

            try:
                assert self._plugins_by_module.get(plugin.__name__, None) is None
            except AssertionError:
                msg = (
                    f"Conflict in plugins by module name: {module_name} is duplicated. "
                    + "Previous will be shadowed."
                )
                if conflict_by_class:
                    logger.error(msg)
                else:
                    logger.warning(msg)

            self._plugins_by_class[plugin.__name__] = plugin
            self._plugins_by_module[module_name] = plugin

    def get(self, plugin_name: str, kwargs):
        """Returns objects of the class defined in plugin plugin_name."""
        try:
            constructor_fn = self._plugins_by_class[plugin_name]
        except KeyError:
            try:
                constructor_fn = self._plugins_by_module[plugin_name]
            except KeyError:
                logger.error("Plugin %s not available.", plugin_name)
                sys.exit(1)

        return constructor_fn(**kwargs)


class COVAPipeline:
    """Abstract class defining pipelines in COVA."""

    @abstractmethod
    def load_pipeline(self, pipeline_config: dict) -> None:
        pass

    @abstractmethod
    def run(self):
        pass


class COVAAutoTune(COVAPipeline):
    """
    Class implementing a COVAPipeline for automatically tuning models in 5 steps:
        1. capture images
        2. filter images
        3. annotate images
        4. create dataset
        5. train model

    The pipeline executes 1-2-3 while until any of the steps returns False.
    Then, executes 4 and 5 once.
    """

    def __init__(self):
        self.factory = COVAFactory()
        self.pipeline = {}

    def load_pipeline(self, pipeline_config: dict, single_stage: Optional[str] = None) -> None:
        """Loads a pipeline as defined in the input dictionary pipeline_config,
        which the configuration for each stage defined in the pipeline.

        Args:
            pipeline_config (dict): configuration of each of the pipeline's stages.
            single_stage (str): execute only this stage. Loads only the plugin for this stage. Defaults to None.
        """

        if single_stage != "":
            stage_config = pipeline_config[single_stage]
            pipeline_config = {}
            pipeline_config[single_stage] = stage_config

        for stage, config in pipeline_config.items():
            if config.get("plugin_path", None):
                self.factory.load_plugins(config["plugin_path"])
            args = config["args"]
            self.pipeline[stage] = self.factory.get(config["plugin"], args)
            logger.info(
                "Using plugin %s for %s.",
                self.pipeline[stage].__class__.__name__,
                stage,
            )

        if single_stage == "":
            try:
                assert all([stage in self.pipeline for stage in PIPELINE])
            except AssertionError:
                logger.error("Some pipeline stages are missing in the input config.")
                sys.exit(1)

    def run(self):
        """Runs the COVA pipeline."""
        while True:
            ret, frame = self.pipeline["capture"].capture()
            if not ret:
                break

            filtered = self.pipeline["filter"].filter(frame)

            if len(filtered) == 0:
                continue

            ret = self.pipeline["annotate"].annotate(frame)
            if not ret:
                break

        for stage in ["capture", "filter"]:
            self.pipeline[stage].epilogue()

        images_path, annotations_path = self.pipeline["annotate"].epilogue()
        logger.info("images stored in %s", images_path)
        logger.info("annotations stored in %s", annotations_path)

        dataset_path = self.pipeline["dataset"].generate(images_path, annotations_path)
        self.pipeline["train"].train(dataset_path)

    def run_stage(self, stage: str, config: Optional[list] = None) -> None:
        """Runs a single stage instead of the full pipeline.

        Args:
            stage (str): Stage to run
            config (dict): Dictionary containing the configuration required by the stage, if any. Defaults to None
        """
        if stage == "annotate":
            images_path, annotations_path = self.pipeline["annotate"].epilogue()
            logger.info("images stored in %s", images_path)
            logger.info("annotations stored in %s", annotations_path)
        elif stage == "dataset":
            images_path = config[0]
            annotations_path = config[1]
            dataset_path = self.pipeline["dataset"].generate(
                images_path, annotations_path
            )
            logger.info("Dataset stored in %s", dataset_path)
        elif stage == "train":
            dataset_path = config[0]
            self.pipeline["train"].train(dataset_path)


class COVACapture(ABC):
    @abstractmethod
    def capture(self) -> None:
        """Captures next frame from stream."""
        raise NotImplementedError

    def epilogue(self) -> None:
        pass


class COVAFilter(ABC):
    @abstractmethod
    def filter(self, img) -> list[Any]:
        """Processes one image."""
        raise NotImplementedError

    def epilogue(self) -> None:
        pass


class COVAAnnotate(ABC):
    @abstractmethod
    def annotate(self, img) -> None:
        """Processes one image."""
        raise NotImplementedError

    def epilogue(self) -> None:
        pass


class COVADataset(ABC):
    @abstractmethod
    def generate(self, path) -> None:
        """Processes one image."""
        raise NotImplementedError

    def epilogue(self) -> None:
        pass


class COVATrain(ABC):
    @abstractmethod
    def train(self) -> None:
        """Abstract method for starting the training.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def epilogue(self) -> None:
        pass
