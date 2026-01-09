import math
from random import random
from typing import Literal

import numpy as np
import math
from random import random
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from ..efficient_kan.kan import KAN as EffKAN
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from ..efficient_kan.kan import KAN as EffKAN
from .phm_network import PHMNetwork


class EfficientKAN(PHMNetwork):
    def __init__(
        self,
        layers,
        task: Literal["regression", "classification"],
        grid_size=2,
        use_native_loss=False,
        continual_learning=False,
        device="cpu",
    ):
        super(EfficientKAN, self).__init__(layers, task, use_native_loss, device)
        self.grid_size = grid_size
        self.model = None
        self.continual_learning = continual_learning
        self.reset()

    def reset(self):
        self.model = EffKAN(
            self.layers,
            grid_size=self.grid_size,
            spline_order=3,
            # grid_eps=0.75 if self.continual_learning else 0.02,
            # scale_base=0 if self.continual_learning else 1,
            sp_trainable=not self.continual_learning,
            sb_trainable=not self.continual_learning,
        ).to(self.device)

    def plot(self, scale=1, in_vars=None):
        """This is an implementation of mine.

        The original implementation of EfficientKAN by Blealtan lacks this feature. It's just a demo, so further refinement is needed.
        """
        base_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        def random_color():
            return random(), random(), random()

        colors = [
            base_colors[x] if x < len(base_colors) else random_color()
            for x in range(max(map(lambda l: l.in_features, self.model.layers)))
        ]
        for layer in reversed(self.model.layers):
            fig, axes = plt.subplots(
                1,
                layer.in_features * layer.out_features,
                figsize=(2 * scale * layer.in_features * layer.out_features, 2 * scale),
            )
            for i in range(layer.in_features):
                for j in range(layer.out_features):
                    x_vals = torch.linspace(-2, 2, 1000)

                    # B-Splines
                    if len(layer.grid) <= j:
                        continue
                    grid = layer.grid[j, :].unsqueeze(0)  # The knots
                    x = x_vals.unsqueeze(-1).unsqueeze(-1)
                    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(
                        x.dtype
                    )  # Determine the interval for each point
                    for k in range(1, layer.spline_order + 1):
                        bases = (
                            (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
                        ) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])

                    y_vals = F.linear(bases.squeeze(), layer.scaled_spline_weight[j, i])
                    y_vals += layer.base_activation(x_vals) * layer.base_weight[j, i]

                    alpha = math.tanh(abs(3 * layer.spline_scaler.view(layer.out_features, -1)[j, i].item()))
                    if type(axes) is np.ndarray:
                        axes[i * layer.out_features + j].plot(
                            x_vals.cpu().detach().numpy(), y_vals.cpu().detach().numpy(), alpha=alpha, color=colors[i]
                        )
                        axes[i * layer.out_features + j].grid(True)
                    else:
                        axes.plot(x_vals, y_vals, alpha=alpha, color=colors[i])
                        axes.grid(True)

        plt.show()
