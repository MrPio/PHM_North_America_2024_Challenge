from typing import Literal

from kan import KAN as Py_KAN

from .phm_network import PHMNetwork


class PyKAN(PHMNetwork):
    def __init__(
        self,
        layers,
        task: Literal["regression", "classification"],
        grid_size=2,
        k=3,
        use_native_loss=False,
        continual_learning=False,
        device="cpu",
    ):
        super(PyKAN, self).__init__(layers, task, use_native_loss, device)
        self.grid_size = grid_size
        self.k = k
        self.model = None
        self.continual_learning = continual_learning
        self.reset()

    def reset(self):
        self.model = Py_KAN(
            width=self.layers,
            grid=self.grid_size,
            k=self.k,
            noise_scale=0.1 if self.continual_learning else 0.3,
            base_fun="zero" if self.continual_learning else "silu",
            sp_trainable=not self.continual_learning,
            sb_trainable=not self.continual_learning,
            device=self.device,
        )

    def plot(self, scale=1, in_vars=None, varscale=4, figsize_base=(14, 10), rotate_in_vars=False):
        try:
            self.model.plot(
                scale=scale,
                in_vars=in_vars,
                out_vars=["$\\mu$", "$log(\\sigma)$"] if self.task == "regression" else ["$P(faulty)$"],
                varscale=varscale / self.layers[0][0],
                figsize_base=figsize_base,
                in_vars_rotation=90 if rotate_in_vars else 0,
            )
        except:
            self.model.plot(
                scale=scale,
                in_vars=in_vars,
                out_vars=["$\\mu$", "$log(\\sigma)$"] if self.task == "regression" else ["$P(faulty)$"],
                varscale=varscale / self.layers[0][0],
            )
