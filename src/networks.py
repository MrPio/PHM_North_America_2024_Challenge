import abc
from datetime import datetime
from math import ceil
from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
import torch
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

        adjusted_confidence = torch.where(pred_labels == 0, 1 - confidence * 2, (confidence - 0.5) * 2)
        adjusted_confidence = torch.where(pred_labels == true_labels, adjusted_confidence, -adjusted_confidence)

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

    def __init__(self, layers: list[int], task: Literal['regression', 'classification'], device=None):
        super().__init__()
        self.layers = layers
        self.task: Literal['regression', 'classification'] = task
        self.model: nn.Module
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.criterion = nn.GaussianNLLLoss(reduction='mean') if self.task == 'regression' else nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([2]).to(self.device))

    def forward(self, x):
        x = self.model(x)
        if self.task == 'regression':
            mu = x[:, 0]
            var = x[:, 1]
            return mu, var
        elif self.task == 'classification':
            return x

    def save(self, name: str):
        torch.save(self.model.state_dict(), f'models/{name}_{__class__.__name__}_layers-{"-".join(str(self.layers))}')

    def load(self, name: str):
        self.model.load_state_dict(torch.load(f'models/{name}', weights_only=True))

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def plot(self, scale=1, in_vars=None):
        pass

    def fit(self, trainset: tuple[Tensor, Tensor], validset: tuple[Tensor, Tensor], optimizer: Optimizer,
            epochs: int, prefix: str = None, batch_size=8192):
        """Train the model on a given train set.

        After each epoch, validates it.
        """
        train_x, train_y = trainset
        valid_x, valid_y = validset

        def get_loss(batch, X_set, y_set):
            x = X_set[batch * batch_size:min(X_set.size(0), (batch + 1) * batch_size)]
            y = y_set[batch * batch_size:min(y_set.size(0), (batch + 1) * batch_size)]
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
            with tqdm(range(ceil(train_x.size(0) / batch_size))) as pbar:
                for i in pbar:
                    loss = get_loss(i, train_x, train_y)
                    epoch_train_losses.append(loss.item())
                    optimizer.step(closure=lambda: loss)
                    pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'], epoch=epoch)
            scheduler.step()

            # Validation
            self.model.eval()
            for i in range(ceil(valid_x.size(0) / batch_size)):
                loss = get_loss(i, valid_x, valid_y)
                epoch_validation_losses.append(loss.item())

            train_losses.append(np.mean(epoch_train_losses))
            validation_losses.append(np.mean(epoch_validation_losses))

        # Plot losses
        plt.figure(figsize=(24, 8))
        plt.plot(train_losses, label='Training loss')
        plt.plot(validation_losses, label='Validation loss')
        plt.legend()
        if prefix is not None:
            plt.savefig(f'img/{prefix}_{datetime.now().ctime().replace(":", "-")}.png')
            plt.close()
        else:
            plt.show()

    def test(self, testset: tuple[Tensor, Tensor], batch_size=8192) -> dict:
        """Test the model on a given test set.

        Returns a dict of metrics.
        """
        test_x, test_y = testset
        self.model.eval()
        scores = []
        score_len = 0
        test_losses = []
        y_pred = torch.tensor([])
        for i in tqdm(range(ceil(test_x.size(0) / batch_size))):
            x = test_x[i * batch_size:min(test_x.size(0), (i + 1) * batch_size)]
            y = test_y[i * batch_size:min(test_y.size(0), (i + 1) * batch_size)]
            if self.task == 'regression':
                mu, log_var = self(x)
                loss = self.criterion(mu, y.view(-1), torch.exp(log_var))
            else:
                confidence = self(x).squeeze()
                score_t = torch.sum(PHMNetwork.score(confidence, y)).item()
                score_len += confidence.size(0)
                scores.append(score_t)
                y_pred = torch.cat((y_pred, (torch.sigmoid(confidence) > 0.5).cpu()))
                loss = self.criterion(confidence, y.float())
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
                    batch_size=8192) -> pd.DataFrame:
        """Train the model multiple times with different data set random splits"""
        results = pd.DataFrame(
            columns=['test_loss'] if self.task == 'regression' else ['avg_test_score', 'accuracy', 'precision',
                                                                     'recall', 'f1'])
        start_date = datetime.now()
        for i in range(times):
            print(f'Time {i + 1}/{times}===========================')
            trainset, validset, testset, normalizations = split_dataset(X, y, train_ratio, device=self.device)
            self.reset()
            self.fit(trainset, validset, optim.Adam(self.parameters(), lr=0.05), epochs, prefix=prefix,
                     batch_size=batch_size)
            metrics = self.test(testset)
            results.loc[i] = metrics

            if self.task == 'classification':
                cm_percentage = metrics['cm'].astype('float') / metrics['cm'].sum() * 100
                labels = np.array([[f'{value:.2f}%' for value in row] for row in cm_percentage])
                plt.figure(figsize=(16, 10))
                sns.heatmap(cm_percentage, annot=labels, fmt='', cmap="viridis",
                            xticklabels=["Nominal (0)", "Faulty (1)"],
                            yticklabels=["Nominal (0)", "Faulty (1)"])
                plt.xlabel("Predicted label")
                plt.ylabel("Actual label")
                plt.title("Confusion matrix")
                if prefix is not None:
                    plt.savefig(f'img/{prefix}_confusion_matrix_{i}.png', bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()

            if prefix is not None:
                self.save(f'torque_target_stochastic_{prefix}_{start_date.strftime("%H-%M-%S")}_{i}')
        if prefix is not None:
            results.to_csv(f'results/{prefix}_{start_date.strftime("%d %B %y %H-%M-%S")}.csv', index=False)
        return results


class PyKAN(PHMNetwork):
    def __init__(self, layers, task: Literal['regression', 'classification'], grid_size=2, device=None):
        super(PyKAN, self).__init__(layers, task, device)
        self.grid_size = grid_size
        self.model = None
        self.reset()

    def reset(self):
        self.model = Py_KAN(width=self.layers, grid=self.grid_size, k=3, device=self.device)

    def plot(self, scale=1, in_vars=None):
        self.model.plot(scale=1.15, in_vars=in_vars,
                        out_vars=['$\\mu$', '$\\sigma$'] if self.task == 'regression' else ['$P(faulty)$'],
                        varscale=0.45 * 8 / self.layers[0][0], figsize_base=(14, 10))


class EfficientKAN(PHMNetwork):
    def __init__(self, layers, task: Literal['regression', 'classification'], grid_size=2, device=None):
        super(EfficientKAN, self).__init__(layers, task, device)
        self.grid_size = grid_size
        self.model = None
        self.reset()

    def reset(self):
        self.model = EffKAN(self.layers, grid_size=self.grid_size, spline_order=3).to(self.device)

    def plot(self, scale=1, in_vars=None):
        pass


class MLP(PHMNetwork):
    def __init__(self, layers, task: Literal['regression', 'classification'], device=None):
        super(MLP, self).__init__(layers, task, device)
        self.model = None
        self.reset()

    def reset(self):
        layers = []
        for i in range(len(self.layers) - 1):
            layers.append(nn.Linear(self.layers[i], self.layers[i + 1]))
            if i < len(self.layers) - 2:  # Add activation for all layers except the last one
                layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers).to(self.device)

    def plot(self, scale=1, in_vars=None):
        pass
