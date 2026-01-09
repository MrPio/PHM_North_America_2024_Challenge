from .dataset import *
from .networks import *
from .pca import *

__author__ = 'Valerio Morelli (@MrPio)'


def to_latex(feature: str):
    feature = (feature
               .replace(" ", "\\times ")
               .replace("norm_air_density", "{ad}")
               .replace("norm_da", "{da}")
               .replace("air_density", "{ad}")
               .replace("np_ng_ratio", "\\frac{np}{ng}")
               .replace("_measured", "^{msr}")
               .replace("_target", "^{trg}")
               .replace("_margin", "^{mrg}"))
    if feature.count('^') > 1:
        feature = feature.replace('^', '_', 1)
    return f"${feature}$"

import os
import pickle
from tqdm import tqdm
from time import time_ns
import random
import math