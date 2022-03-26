"""This module implements a dummy COVATrainer. It does nothing."""

from cova.pipeline.pipeline import COVATrain


class DummyTrainer(COVATrain):
    def train(self) -> None:
        pass

    def epilogue(self) -> None:
        pass
