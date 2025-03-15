import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklPCA


class PCA:
    def __init__(self, data: pd.DataFrame, loadings=None):
        self.data = data
        self.pca = sklPCA(n_components=len(self.data.columns))
        self.pca.fit_transform(self.data)
        if loadings is not None:
            self.loadings = loadings
        else:
            self.loadings = self.pca.components_[:3]

    def plot_variance(self, scale=0.5):
        plt.figure(figsize=(12 * scale, 6 * scale))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
        plt.xlabel("Number of principal components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Explained variance vs. number of principal components")
        plt.grid(True)
        plt.show()

    def plot_loadings(self, scale=0.5, is_3d=False):
        fig, ax = plt.subplots(1, 3 if is_3d else 2, figsize=(14 * scale, 6 * scale))
        for i in range(3 if is_3d else 2):
            norm = plt.Normalize(0, max(self.loadings[i]))
            colors = plt.cm.viridis(norm(abs(self.loadings[i])))
            ax[i].barh(self.data.columns, self.loadings[i], color=colors, alpha=0.85)
            ax[i].set_title(f"Feature loadings for PC{i + 1}")
            ax[i].set_xlabel("Loading value")
        plt.tight_layout()
        plt.show()

    def plot_pca(self, labels=None, scale=0.5, ax=None, c=None, alpha=0.04, azim=235, is_3d=False) -> plt.axis:
        """
        Plot a 2D or 3D PCA projection of the PCA components.
        :param labels: The 0, 1 labels of the samples.
        :param scale: The scale of the figure.
        :param ax: If provided, the axis to plot on. Otherwise, a new figure is created.
        :param c: the color of the scatter plot. Must be a dict if labels is provided.
        :param alpha:
        :param azim: the azimuth of the 3D plot.
        :param is_3d: if the plot is 3D.
        :return:
        """
        components = pd.DataFrame()
        for i in range(3 if is_3d else 2):
            components[f'PC{i + 1}'] = self.data.dot(self.loadings[i, :])

        if ax is None:
            fig = plt.figure(figsize=(12 * scale, 12 * scale))
            ax = fig.add_subplot(111, projection='3d' if is_3d else None)
            if is_3d:
                ax.view_init(elev=25, azim=azim)
            ax.grid()

        if c is None:
            c = {0: 'g', 1: 'r'}
        s = 0.5 * 750_000 / len(self.data)
        pcs = [f'PC{x + 1}' for x in range(3 if is_3d else 2)]
        if labels is None:
            ax.scatter(*[components[pc] for pc in pcs], alpha=alpha, s=s)
        else:
            for label, name in {0: 'Healthy', 1: 'Faulty'}.items():
                ax.scatter(*[components.loc[labels == label, pc] for pc in pcs],
                           c=c[label], label=name, alpha=alpha, s=s)
        return ax
