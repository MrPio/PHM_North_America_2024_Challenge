def to_latex(feature: str):
    feature = (feature
               .replace(" ", "\\times ")
               .replace("norm_air_density", "{ad}")
               .replace("norm_da", "{da}")
               .replace("air_density", "{ad}")
               .replace("np_ng_ratio", "\\frac{np}{ng}")
               .replace("_measured", "_{msr}")
               .replace("_target", "_{trg}")
               .replace("_margin", "_{mrg}"))
    if feature.count('_') > 1:
        feature = feature.replace('_', '^', 1)
    return f"${feature}$"