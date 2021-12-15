"""This module implements a dummy COVAAnnotator. It does nothing."""

from cova.pipeline.pipeline import COVAAnnotate


class Dummy(COVAAnnotate):
    def annotate(self, image) -> None:
        pass

    def epilogue(self) -> None:
        pass