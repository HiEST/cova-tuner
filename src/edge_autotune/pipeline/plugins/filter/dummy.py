"""This module implements a dummy COVAFilter. It does nothing."""

from edge_autotune.pipeline.pipeline import COVAFilter


class Dummy(COVAFilter):
    def filter(self, image) -> None:
        return image

    def epilogue(self) -> None:
        pass