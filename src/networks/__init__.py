"""
Contains A Wrapper for the PyKAN and EfficientKAN classes and defines a MLP network for regression and classification.
The EfficientKAN class is a wrapper for the efficient KAN implementation by Blealtan, available at https://github.com/Blealtan/efficient-kan.
The PyKAN class is a wrapper for the PyTorch KAN class, which is available at https://github.com/KindXiaoming/pykan
Author: Valerio Morelli (@MrPio)
"""

from .phm_network import PHMNetwork
from .pykan import PyKAN
from .effkan import EfficientKAN
from .mlp import MLP

__all__ = ["PHMNetwork", "PyKAN", "EfficientKAN", "MLP"]
__author__ = "Valerio Morelli (@MrPio)"
