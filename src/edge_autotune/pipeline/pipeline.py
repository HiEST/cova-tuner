from abc import ABC, abstractmethod
import importlib
import inspect
import logging
import os
from pathlib import Path
import sys
from typing import NewType, Any, Tuple, Callable, Dict, List

import cv2


logger = logging.getLogger(__name__)
logging.basicConfig(level='DEBUG')

Args = NewType('Args', Dict[str, Any])
Stage = NewType('Stage', Tuple[str, Args])

PIPELINE = ['capture', 'filter', 'annotate', 'train']
CONSTRUCTORS = ['COVACapture', 'COVAFilter', 'COVAAnnotate', 'COVATrain']


class COVAPipeline:
    def __init__(self, capturer: Stage, filter: Stage,
                annotator: Stage, trainer: Stage):

        self.factory = COVAFactory()
        self.pipeline = {}

        capture_plugin, capturer_args = capturer
        filter_plugin, filter_args = filter
        annotate_plugin, annotator_args = annotator
        train_plugin, trainer_args = trainer

        self.pipeline['capture'] = self.factory.get(capture_plugin, *capturer_args)
        self.pipeline['filter'] =  self.factory.get(filter_plugin, *filter_args)
        self.pipeline['annotate'] =  self.factory.get(annotate_plugin, *annotator_args)
        self.pipeline['train'] =  self.factory.get(train_plugin, *trainer_args)

    def run(self):
        
        while True:
            ret, frame = self.pipeline['capture'].capture()
            if not ret:
                break
        
            ret, filtered = self.pipeline['filter'].filter(frame)
            if not ret:
                break
            elif not len(filtered):
                continue

            ret = self.pipeline['annotate'].annotate(filtered)
            if not ret:
                break

        for stage in PIPELINE:
            self.pipeline[stage].epilogue()
            self.pipeline[stage].epilogue()
            self.pipeline[stage].epilogue()

        self.pipeline['train'].train()


class COVAFactory:
    def __init__(self):
        self._plugins_by_class = {}
        self._plugins_by_module = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        plugins_dir = os.path.join(current_dir, 'plugins')
        self._load_plugins(plugins_dir)

    @staticmethod
    def _detect_class(module) -> Tuple[Callable, str]:
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
        logger.info(f'Loaded plugin {constructor.__name__} from module {module.__name__}.')
        return constructor, module.__name__

    def _load_plugins(self, plugins_path: str) -> None:
        """Loads the plugins found in the plugins_path."""
        if os.path.isdir(plugins_path):
            plugins = [str(p) for p in Path(plugins_path).rglob('*.py')]
        else:
            plugins = [plugins_path]

        for plugin_file in plugins:
            plugin, module_name = COVAFactory._load_plugin(plugin_file)

            # Check that no other plugin had the same name.
            conflict_by_class = False
            try:
                assert self._plugins_by_class.get(plugin.__name__, None) is None
            except AssertionError:
                conflict_by_class = True
                msg = f'Conflict in plugins by class name: {plugin.__name__} is duplicated. ' +\
                        'Previous will be shadowed.'
                logger.warning(msg)

            try:
                assert self._plugins_by_module.get(plugin.__name__, None) is None
            except AssertionError:
                msg = f'Conflict in plugins by module name: {module_name} is duplicated. ' +\
                        'Previous will be shadowed.'
                if conflict_by_class:
                    logger.error(msg)
                else:
                    logger.warning(msg)

            self._plugins_by_class[plugin.__name__] = plugin
            self._plugins_by_module[module_name] = plugin

    def get(self, plugin_name: str, *args, **kwargs):
        """Returns objects of the class defined in plugin plugin_name."""
        try:
            constructor_fn = self._plugins_by_class[plugin_name]
        except KeyError:
            try:
                constructor_fn = self._plugins_by_module[plugin_name]
            except KeyError:
                logger.error(f'Plugin {plugin_name} not available.')
                sys.exit(1)

        return constructor_fn(*args, *kwargs)


class COVACapture(ABC):
    @abstractmethod
    def capture(self) -> None:
        """Captures next frame from stream."""
        raise NotImplementedError

    def epilogue(self) -> None:
        raise NotImplementedError


class COVAFilter(ABC):
    @abstractmethod
    def filter(self, img) -> List[Any]:
        """Processes one image."""
        raise NotImplementedError

    @abstractmethod
    def epilogue(self, *args, **kwargs) -> None:
        """Processes all pending images, if any.
        The images are sent to the server. Yields annotations."""
        raise NotImplementedError


class COVAAnnotate(ABC):
    @abstractmethod
    def annotate(self, img) -> None:
        """Processes one image."""
        raise NotImplementedError

    @abstractmethod
    def epilogue(self, *args, **kwargs) -> None:
        """Processes all pending images, if any.
        The images are sent to the server. Yields annotations."""
        raise NotImplementedError


class COVATrain(ABC):
    @abstractmethod
    def train(self) -> None:
        """Processes one image."""
        raise NotImplementedError

    @abstractmethod
    def epilogue(self, *args, **kwargs) -> None:
        """Processes all pending images, if any.
        The images are sent to the server. Yields annotations."""
        raise NotImplementedError