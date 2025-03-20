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
    return f"${feature}$"
