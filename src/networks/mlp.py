from typing import Literal

import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch import nn

from .phm_network import PHMNetwork


class MLP(PHMNetwork):
    def __init__(self, layers, task: Literal["regression", "classification"], use_native_loss=False, device="cpu"):
        super(MLP, self).__init__(layers, task, use_native_loss, device)
        self.model = None
        self.reset()

    def reset(self):
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:  # Add activation for all layers except the last one
                layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers).to(self.device)

    def plot(self, scale=1, in_vars=None, node_size=300, font_size=11, edge_width=5, use_abs=False, vmax=1):
        layers = [self.model[0].in_features]
        linears = list(filter(lambda l: type(l) == torch.nn.modules.linear.Linear, self.model))
        for l in linears:
            layers.append(l.out_features)
        # layers = [6, 50, 50, 2]
        G = nx.Graph()
        pos = {}
        node_count = 0
        layer_gap = 5
        node_gap = 0.1
        max_layer = max(layers)

        for i, layer_size in enumerate(layers):
            delta = max_layer - layer_size
            for j in range(layer_size):
                G.add_node(node_count)
                pos[node_count] = (i * layer_gap, (j + delta // 2) * node_gap)
                if i > 0:
                    for k in range(layers[i - 1]):
                        weight = linears[i - 1].weight[j, k].item()
                        G.add_edge(
                            node_count - layers[i - 1] + k - j,
                            node_count,
                            weight=abs(weight) if use_abs else weight,
                        )
                node_count += 1

        weights = [G[u][v]["weight"] for u, v in G.edges()]
        plt.figure(1, figsize=(16, 4))
        plt.hist(weights, bins="auto", edgecolor="black")
        plt.grid()
        plt.title("Weights distribution")
        plt.figure(3, figsize=(scale * 16, scale * 12))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=node_size,
            node_color="tab:blue",
            edge_cmap=plt.colormaps["viridis" if use_abs else "coolwarm"],
            edge_color=weights,
            edge_vmin=0 if use_abs else -vmax,
            edge_vmax=vmax if use_abs else vmax,
            font_size=font_size,
            font_color="white",
            width=edge_width,
        )
        plt.show()
