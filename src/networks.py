import abc
import math
from datetime import datetime
from math import ceil
from random import random
from typing import Callable
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from kan import KAN as Py_KAN
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch import nn, Tensor, optim
from torch.optim import Optimizer
from tqdm import tqdm

from . import split_dataset
from .efficient_kan.kan import KAN as EffKAN

"""
Contains A Wrapper for the PyKAN and EfficientKAN classes and defines a MLP network for regression and classification.
The EfficientKAN class is a wrapper for the efficient KAN implementation by Blealtan, available at https://github.com/Blealtan/efficient-kan.
The PyKAN class is a wrapper for the PyTorch KAN class, which is available at https://github.com/KindXiaoming/pykan
Author: Valerio Morelli (@MrPio)
"""

__all__ = ['PHMNetwork', 'PyKAN', 'EfficientKAN', 'MLP']
__author__ = 'Valerio Morelli (@MrPio)'


class PHMNetwork(nn.Module, abc.ABC):

    @staticmethod
    def score(confidence: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
        """Calculate the fault detection score as dictated by the challenge at https://data.phmsociety.org/phm2024-conference-data-challenge/"""
        pred_labels: torch.Tensor = torch.where(confidence > 0.5, 1, 0)
        confidence = torch.sigmoid(confidence)

        valid_confidence_mask = (confidence >= 0) & (confidence <= 1)
        valid_label_mask = (pred_labels == 0) | (pred_labels == 1)
        valid_mask = valid_confidence_mask & valid_label_mask

        adjusted_confidence = torch.where(
            pred_labels == 0, 1 - confidence * 2, (confidence - 0.5) * 2)
        adjusted_confidence = torch.where(
            pred_labels == true_labels, adjusted_confidence, -adjusted_confidence)

        healthy_scores = adjusted_confidence.clone()
        faulty_scores = torch.where(
            adjusted_confidence >= 0,
            adjusted_confidence,
            4 * adjusted_confidence ** 11 + adjusted_confidence
        )
        scores = torch.where(
            true_labels == 0,
            healthy_scores,
            faulty_scores
        )
        return torch.where(valid_mask, scores, -100.0)

    def __init__(self, layers: list[int] | list[list[int]], task: Literal['regression', 'classification'],
                 use_native_loss=False, device='cpu'):
        super().__init__()
        self.layers = layers
        self.task: Literal['regression', 'classification'] = task
        self.model: nn.Module
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        if self.task == 'regression':
            def normalizedGNLL(input, target, var):
                var = var.clamp(min=1e-6)
                y_max = 1 / torch.sqrt(2 * torch.pi * var)
                norm_factor = torch.where(
                    y_max > 1, y_max, torch.ones_like(y_max))
                # loss = 0.5 * torch.log(2 * torch.pi * var) + (input - target) ** 2 / (2 * var) + torch.log(norm_factor)
                loss = -1 / torch.sqrt(2 * torch.pi * var) * \
                    torch.exp(-(input - target) ** 2 / (2 * var)) / norm_factor
                return loss.mean()

            self.criterion = nn.GaussianNLLLoss().to(
                self.device) if use_native_loss else normalizedGNLL
        else:
            def classNegScore(confidence: torch.Tensor, true_labels: torch.Tensor):
                pred_labels: torch.Tensor = torch.where(confidence > 0.5, 1, 0)
                confidence = torch.sigmoid(confidence)

                valid_confidence_mask = (confidence >= 0) & (confidence <= 1)
                valid_label_mask = (pred_labels == 0) | (pred_labels == 1)
                valid_mask = valid_confidence_mask & valid_label_mask

                adjusted_confidence = torch.where(
                    pred_labels == 0, 1 - confidence * 2, (confidence - 0.5) * 2)
                adjusted_confidence = torch.where(
                    pred_labels == true_labels, adjusted_confidence, -adjusted_confidence)

                healthy_scores = adjusted_confidence.clone()
                faulty_scores = torch.where(
                    adjusted_confidence >= 0,
                    adjusted_confidence,
                    4 * adjusted_confidence ** 11 + adjusted_confidence
                )
                scores = torch.where(
                    true_labels == 0,
                    healthy_scores,
                    faulty_scores
                )
                return -torch.where(valid_mask, scores, -100.0).mean()

            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([2]).to(self.device)) if use_native_loss else classNegScore

    def forward(self, x):
        x = self.model(x)
        if self.task == 'regression':
            mu = x[:, 0]
            var = x[:, 1]
            return mu, var
        elif self.task == 'classification':
            return x

    def save(self, name: str):
        torch.save(self.model.state_dict(
        ), f'models/{name}_{self.__class__.__name__}_layers-{str(self.layers)}')

    def load(self, name: str):
        self.model.load_state_dict(
            torch.load(f'models/{name}_{self.__class__.__name__}_layers-{str(self.layers)}', weights_only=True))

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def plot(self, scale=1, in_vars=None):
        pass

    def fit(self, trainset: tuple[Tensor, Tensor], validset: tuple[Tensor, Tensor], optimizer: Optimizer,
            epochs: int, prefix: str = None, batch_size=4096, callback: Callable = None, silent=False) -> tuple[
            list[int], list[int]]:
        """Train the model on a given train set.

        After each epoch, validates it.
        """
        train_x, train_y = trainset
        valid_x, valid_y = validset

        def get_loss(batch, X_set, y_set):
            x = X_set[batch *
                      batch_size:min(X_set.size(0), (batch + 1) * batch_size)]
            y = y_set[batch *
                      batch_size:min(y_set.size(0), (batch + 1) * batch_size)]
            optimizer.zero_grad()
            if self.task == 'regression':
                mu, log_var = self(x)
                _loss = self.criterion(mu, y.view(-1), torch.exp(log_var))
            else:
                _loss = self.criterion(self(x), y.float())
            _loss.backward()
            return _loss

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
        train_losses = []
        validation_losses = []
        for epoch in range(epochs):
            epoch_train_losses = []
            epoch_validation_losses = []

            # Training
            self.model.train()
            with tqdm(range(ceil(train_x.size(0) / batch_size)), disable=silent) as pbar:
                for i in pbar:
                    loss = get_loss(i, train_x, train_y)
                    epoch_train_losses.append(loss.item())
                    optimizer.step(closure=lambda: loss)
                    pbar.set_postfix(
                        loss=loss.item(), lr=optimizer.param_groups[0]['lr'], epoch=epoch)
            scheduler.step()

            # Validation
            self.model.eval()
            for i in range(ceil(valid_x.size(0) / batch_size)):
                loss = get_loss(i, valid_x, valid_y)
                epoch_validation_losses.append(loss.item())

            train_losses.append(np.mean(epoch_train_losses))
            validation_losses.append(np.mean(epoch_validation_losses))

            if callback:
                callback()

        # Plot losses
        if prefix is not None:
            plt.figure(figsize=(24, 8))
            plt.plot(train_losses, label='Training loss')
            plt.plot(validation_losses, label='Validation loss')
            plt.legend()
            plt.savefig(
                f'img/{prefix}_{datetime.now().ctime().replace(":", "-")}.png')
            plt.close()
        return train_losses, validation_losses

    def test(self, testset: tuple[Tensor, Tensor], batch_size=4096, silent=False) -> dict:
        """Test the model on a given test set.

        Returns a dict of metrics.
        """
        test_x, test_y = testset
        self.eval()
        scores = []
        score_len = 0
        test_losses = []
        y_pred = torch.tensor([])
        for i in tqdm(range(ceil(test_x.size(0) / batch_size)), disable=silent):
            x = test_x[i *
                       batch_size:min(test_x.size(0), (i + 1) * batch_size)]
            y = test_y[i *
                       batch_size:min(test_y.size(0), (i + 1) * batch_size)]
            if self.task == 'regression':
                mu, log_var = self(x)
                loss = self.criterion(mu, y.view(-1), torch.exp(log_var))
            else:
                confidence = self(x).squeeze()
                loss = self.criterion(confidence, y.view(-1).float())
                score_t = torch.sum(PHMNetwork.score(
                    confidence, y.view(-1))).item()
                score_len += confidence.size(0)
                scores.append(score_t)
                y_pred = torch.cat(
                    (y_pred, (torch.sigmoid(confidence) > 0.5).cpu()))
            loss.backward()
            test_losses.append(loss.item())
        metrics = {'test_loss': np.mean(test_losses)}
        if self.task == 'classification':
            labels = test_y.cpu()
            metrics['avg_test_score'] = sum(scores) / score_len
            metrics['accuracy'] = accuracy_score(labels, y_pred)
            metrics['precision'] = precision_score(labels, y_pred)
            metrics['recall'] = recall_score(labels, y_pred)
            metrics['f1'] = f1_score(labels, y_pred)
            metrics['cm'] = confusion_matrix(labels, y_pred)
        return metrics

    def multi_train(self, X: pd.DataFrame, y: pd.Series, epochs, prefix: str = None, train_ratio=0.25, times=10,
                    batch_size=2048, lr=0.05, reset_each_time=True) -> pd.DataFrame:
        """Train the model multiple times with different data set random splits"""
        results = pd.DataFrame(
            columns=['test_loss'] if self.task == 'regression' else ['avg_test_score', 'accuracy', 'precision',
                                                                     'recall', 'f1'])
        normalizations_df = pd.DataFrame(
            columns=[f'{col}_mean' for col in X.columns] + [f'{col}_std' for col in X.columns] + ['y_mean', 'y_std'])
        for i in range(times):
            print(f'Time {i + 1}/{times}===========================')
            trainset, validset, testset, normalizations = split_dataset(X, y, train_ratio,
                                                                        standardize_y=False,
                                                                        # because the loss score would not be faithful to the challenge score otherwise.
                                                                        # self.task != 'classification',
                                                                        device=self.device)
            normalizations_df.loc[i] = normalizations
            if reset_each_time:
                self.reset()
            self.fit(trainset, validset, optim.Adam(self.parameters(), lr=lr), epochs, prefix=prefix,
                     batch_size=batch_size)
            metrics = self.test(testset)
            results.loc[i] = metrics

            if self.task == 'classification':
                cm_percentage = metrics['cm'].astype(
                    'float') / metrics['cm'].sum() * 100
                labels = np.array(
                    [[f'{value:.2f}%' for value in row] for row in cm_percentage])
                plt.figure(figsize=(16, 10))
                sns.heatmap(cm_percentage, annot=labels, fmt='', cmap="viridis",
                            xticklabels=["Nominal (0)", "Faulty (1)"],
                            yticklabels=["Nominal (0)", "Faulty (1)"])
                plt.xlabel("Predicted label")
                plt.ylabel("Actual label")
                plt.title("Confusion matrix")
                if prefix is not None:
                    plt.savefig(
                        f'img/{prefix}_confusion_matrix_{i}.png', bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()

            if prefix is not None:
                self.save(f'torque_target_stochastic_{prefix}_{i}')
        if prefix is not None:
            results.to_csv(f'results/{prefix}.csv', index=False)
            normalizations_df.to_csv(f'results/{prefix}_norm.csv',
                                     index=False)
        return results


class PyKAN(PHMNetwork):
    def __init__(self, layers, task: Literal['regression', 'classification'], grid_size=2, k=3, use_native_loss=False,
                 continual_learning=False,
                 device='cpu'):
        super(PyKAN, self).__init__(layers, task, use_native_loss, device)
        self.grid_size = grid_size
        self.k = k
        self.model = None
        self.continual_learning = continual_learning
        self.reset()

    def reset(self):
        self.model = Py_KAN(width=self.layers, grid=self.grid_size, k=self.k,
                            noise_scale=0.1 if self.continual_learning else 0.3,
                            base_fun='zero' if self.continual_learning else 'silu',
                            sp_trainable=not self.continual_learning,
                            sb_trainable=not self.continual_learning, device=self.device)

    def plot(self, scale=1, in_vars=None, varscale=4, figsize_base=(14, 10), rotate_in_vars=False):
        try:
            self.model.plot(scale=scale, in_vars=in_vars,
                            out_vars=['$\\mu$', '$log(\\sigma)$'] if self.task == 'regression' else [
                                '$P(faulty)$'],
                            varscale=varscale / self.layers[0][0], figsize_base=figsize_base,
                            in_vars_rotation=90 if rotate_in_vars else 0)
        except:
            self.model.plot(scale=scale, in_vars=in_vars,
                            out_vars=['$\\mu$', '$log(\\sigma)$'] if self.task == 'regression' else [
                                '$P(faulty)$'],
                            varscale=varscale / self.layers[0][0])


class EfficientKAN(PHMNetwork):
    def __init__(self, layers, task: Literal['regression', 'classification'], grid_size=2, use_native_loss=False,
                 continual_learning=False, device='cpu'):
        super(EfficientKAN, self).__init__(
            layers, task, use_native_loss, device)
        self.grid_size = grid_size
        self.model = None
        self.continual_learning = continual_learning
        self.reset()

    def reset(self):
        self.model = EffKAN(self.layers, grid_size=self.grid_size, spline_order=3,
                            # grid_eps=0.75 if self.continual_learning else 0.02,
                            # scale_base=0 if self.continual_learning else 1,
                            sp_trainable=not self.continual_learning,
                            sb_trainable=not self.continual_learning).to(self.device)

    def plot(self, scale=1, in_vars=None):
        """This is an implementation of mine.

        The original implementation of EfficientKAN by Blealtan lacks this feature. It's just a demo, so further refinement is needed.
        """
        base_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        def random_color():
            return random(), random(), random()

        colors = [base_colors[x] if x < len(base_colors) else random_color()
                  for x in range(max(map(lambda l: l.in_features, self.model.layers)))]
        for layer in reversed(self.model.layers):
            fig, axes = plt.subplots(1, layer.in_features * layer.out_features,
                                     figsize=(2 * scale * layer.in_features * layer.out_features, 2 * scale))
            for i in range(layer.in_features):
                for j in range(layer.out_features):
                    x_vals = torch.linspace(-2, 2, 1000)

                    # B-Splines
                    if len(layer.grid) <= j:
                        continue
                    grid = layer.grid[j, :].unsqueeze(0)  # The knots
                    x = x_vals.unsqueeze(-1).unsqueeze(-1)
                    bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(
                        x.dtype)  # Determine the interval for each point
                    for k in range(1, layer.spline_order + 1):
                        bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                        ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                        )

                    y_vals = F.linear(
                        bases.squeeze(), layer.scaled_spline_weight[j, i])
                    y_vals += (layer.base_activation(x_vals)
                               * layer.base_weight[j, i])

                    alpha = math.tanh(
                        abs(3 * layer.spline_scaler.view(layer.out_features, -1)[j, i].item()))
                    if type(axes) is np.ndarray:
                        axes[i * layer.out_features + j].plot(x_vals.cpu().detach().numpy(),
                                                              y_vals.cpu().detach().numpy(),
                                                              alpha=alpha, color=colors[i])
                        axes[i * layer.out_features + j].grid(True)
                    else:
                        axes.plot(x_vals, y_vals, alpha=alpha, color=colors[i])
                        axes.grid(True)

        plt.show()


class MLP(PHMNetwork):
    def __init__(self, layers, task: Literal['regression', 'classification'], use_native_loss=False, device='cpu'):
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

    def plot(self, scale=1, in_vars=None, node_size=300, font_size=11, edge_width=5):
        layers = [self.model[0].in_features]
        linears = list(filter(lambda l: type(
            l) == torch.nn.modules.linear.Linear, self.model))
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
                        G.add_edge(node_count - layers[i - 1] + k - j, node_count,
                                   weight=linears[i - 1].weight[j, k].item())
                node_count += 1

        weights = [G[u][v]['weight'] for u, v in G.edges()]
        plt.figure(1, figsize=(16, 4))
        plt.hist(weights, bins='auto', edgecolor='black')
        plt.grid()
        plt.title('Weights distribution')
        plt.figure(3, figsize=(scale * 16, scale * 12))
        nx.draw(G, pos, with_labels=True, node_size=node_size, node_color="skyblue", edge_cmap=plt.colormaps['PiYG'],
                edge_color=weights, edge_vmin=-0.5, edge_vmax=0.5, font_size=font_size, width=edge_width)
        plt.show()
